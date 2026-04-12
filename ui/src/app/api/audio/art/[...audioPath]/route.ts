/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';

/**
 * Serves embedded album art from an MP3 file's ID3v2 tag.
 * Reads only the tag header from disk (no full-file buffering).
 * Returns the raw image bytes with correct Content-Type.
 */

function synchsafeToInt(b0: number, b1: number, b2: number, b3: number) {
  return ((b0 & 0x7f) << 21) | ((b1 & 0x7f) << 14) | ((b2 & 0x7f) << 7) | (b3 & 0x7f);
}

function deUnsync(bytes: Buffer) {
  const out: number[] = [];
  for (let i = 0; i < bytes.length; i++) {
    out.push(bytes[i]);
    if (bytes[i] === 0xff && i + 1 < bytes.length && bytes[i + 1] === 0x00) i += 1;
  }
  return Buffer.from(out);
}

function readNullTerminated(buf: Buffer, start: number, wide: boolean): { text: string; next: number } {
  if (wide) {
    let i = start;
    while (i + 1 < buf.length && !(buf[i] === 0 && buf[i + 1] === 0)) i += 2;
    return { text: buf.slice(start, i).toString('utf16le'), next: i + 2 };
  }
  let i = start;
  while (i < buf.length && buf[i] !== 0) i++;
  return { text: buf.slice(start, i).toString('latin1'), next: i + 1 };
}

type ArtResult = { mime: string; data: Buffer } | null;

function extractArtFromTag(buf: Buffer): ArtResult {
  if (buf.length < 10) return null;
  if (buf[0] !== 0x49 || buf[1] !== 0x44 || buf[2] !== 0x33) return null; // "ID3"

  const verMajor = buf[3]; // 2, 3, or 4
  const flags = buf[5];
  const tagSize = synchsafeToInt(buf[6], buf[7], buf[8], buf[9]);
  const tagEnd = Math.min(10 + tagSize, buf.length);

  let tagData = buf.slice(10, tagEnd);
  if ((flags & 0x80) !== 0) tagData = deUnsync(tagData);

  let offset = 0;

  // Skip extended header
  if ((verMajor === 3 || verMajor === 4) && (flags & 0x40) !== 0 && tagData.length >= 4) {
    const extSize =
      verMajor === 4
        ? synchsafeToInt(tagData[0], tagData[1], tagData[2], tagData[3])
        : (tagData[0] << 24) | (tagData[1] << 16) | (tagData[2] << 8) | tagData[3];
    offset += 4 + Math.max(0, extSize);
  }

  while (offset < tagData.length) {
    if (tagData[offset] === 0x00) break;

    if (verMajor === 2) {
      // ID3v2.2: 3-byte frame ID, 3-byte size
      if (offset + 6 > tagData.length) break;
      const id = tagData.slice(offset, offset + 3).toString('latin1');
      const size = (tagData[offset + 3] << 16) | (tagData[offset + 4] << 8) | tagData[offset + 5];
      offset += 6;
      if (!id.trim() || size <= 0 || offset + size > tagData.length) break;

      if (id === 'PIC' && size > 6) {
        const frame = tagData.slice(offset, offset + size);
        const fmt = frame.slice(1, 4).toString('latin1').toLowerCase();
        const mime = fmt === 'png' ? 'image/png' : 'image/jpeg';
        // skip: encoding(1) + format(3) + pictureType(1) = 5, then null-terminated description
        let p = 5;
        const enc = frame[0];
        const wide = enc === 1 || enc === 2;
        const desc = readNullTerminated(frame as any, p, wide);
        p = desc.next;
        if (p < frame.length) {
          const img = frame.slice(p);
          if (img.length > 64) return { mime, data: Buffer.from(img) };
        }
      }
      offset += size;
    } else {
      // ID3v2.3/v2.4: 4-byte frame ID, 4-byte size, 2-byte flags
      if (offset + 10 > tagData.length) break;
      const id = tagData.slice(offset, offset + 4).toString('latin1');
      let size =
        verMajor === 4
          ? synchsafeToInt(tagData[offset + 4], tagData[offset + 5], tagData[offset + 6], tagData[offset + 7])
          : (tagData[offset + 4] << 24) |
            (tagData[offset + 5] << 16) |
            (tagData[offset + 6] << 8) |
            tagData[offset + 7];
      const flag2 = tagData[offset + 9];
      offset += 10;
      if (!id.trim() || size <= 0 || offset + size > tagData.length) break;

      if (id === 'APIC') {
        let frame = tagData.slice(offset, offset + size);
        if (verMajor === 4 && (flag2 & 0x02) !== 0) frame = deUnsync(frame);

        const enc = frame[0];
        // mime type: null-terminated latin1
        const mimeZ = readNullTerminated(frame as any, 1, false);
        const mime = mimeZ.text || 'image/jpeg';
        let p = mimeZ.next;
        if (p < frame.length) p += 1; // picture type byte
        const wide = enc === 1 || enc === 2;
        const desc = readNullTerminated(frame as any, p, wide);
        p = desc.next;
        if (p < frame.length) {
          const img = frame.slice(p);
          if (img.length > 64) return { mime, data: Buffer.from(img) };
        }
      }
      offset += size;
    }
  }
  return null;
}

export async function GET(request: NextRequest, { params }: { params: { audioPath: string } }) {
  const { audioPath } = await params;
  try {
    const filepath = decodeURIComponent(audioPath);

    // Security check
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();
    const allowedDirs = [datasetRoot, trainingRoot, dataRoot];
    const isAllowed = allowedDirs.some(d => filepath.startsWith(d)) && !filepath.includes('..');
    if (!isAllowed) {
      return new NextResponse('Access denied', { status: 403 });
    }

    const stat = await fs.promises.stat(filepath).catch(() => null);
    if (!stat || !stat.isFile()) {
      return new NextResponse('File not found', { status: 404 });
    }

    // Read only the ID3 tag (first min(tagSize, 4MB) bytes).
    // First read 10 bytes to get tag size, then read the full tag.
    const fd = await fs.promises.open(filepath, 'r');
    try {
      const headerBuf = Buffer.alloc(10);
      await fd.read(headerBuf, 0, 10, 0);

      if (headerBuf[0] !== 0x49 || headerBuf[1] !== 0x44 || headerBuf[2] !== 0x33) {
        return new NextResponse('No ID3 tag', { status: 404 });
      }

      const tagSize = synchsafeToInt(headerBuf[6], headerBuf[7], headerBuf[8], headerBuf[9]);
      const totalRead = Math.min(10 + tagSize, 4_000_000);

      const tagBuf = Buffer.alloc(totalRead);
      await fd.read(tagBuf, 0, totalRead, 0);

      const art = extractArtFromTag(tagBuf);
      if (!art) {
        return new NextResponse('No album art found', { status: 404 });
      }

      return new NextResponse(art.data, {
        headers: {
          'Content-Type': art.mime,
          'Content-Length': String(art.data.length),
          'Cache-Control': 'public, max-age=604800, immutable',
        },
      });
    } finally {
      await fd.close();
    }
  } catch (error) {
    console.error('Error extracting album art:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
