/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';

const contentTypeMap: { [key: string]: string } = {
  // Images
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.bmp': 'image/bmp',
  // Videos
  '.mp4': 'video/mp4',
  '.avi': 'video/x-msvideo',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.wmv': 'video/x-ms-wmv',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  // Audio
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

export async function GET(request: NextRequest, { params }: { params: { imagePath: string } }) {
  const { imagePath } = await params;
  try {
    // Decode the path
    const filepath = decodeURIComponent(imagePath);

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();

    const allowedDirs = [datasetRoot, trainingRoot, dataRoot];

    // Security check: resolve the path so any `..` segments are collapsed,
    // then ensure it's still under an allowed root. (Plain `.includes('..')`
    // false-positives on filenames that contain `..` as text, e.g. an ellipsis.)
    const resolved = path.resolve(filepath);
    const isAllowed = allowedDirs.some(
      allowedDir => resolved === allowedDir || resolved.startsWith(allowedDir + path.sep),
    );

    if (!isAllowed) {
      console.warn(`Access denied: ${resolved} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Bail out early if the client already gave up
    if (request.signal.aborted) {
      return new NextResponse(null, { status: 499 });
    }

    // Stat file (async)
    const stat = await fs.promises.stat(resolved).catch(() => null);
    if (!stat || !stat.isFile()) {
      return new NextResponse('File not found', { status: 404 });
    }

    const ext = path.extname(resolved).toLowerCase();
    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    // Weak ETag from inode/size/mtime — cheap and stable enough for revalidation
    const etag = `W/"${stat.ino.toString(36)}-${stat.size.toString(36)}-${stat.mtimeMs.toString(36)}"`;
    const cacheControl = 'public, max-age=86400, immutable';

    const ifNoneMatch = request.headers.get('if-none-match');
    if (ifNoneMatch && ifNoneMatch === etag) {
      return new NextResponse(null, {
        status: 304,
        headers: {
          ETag: etag,
          'Cache-Control': cacheControl,
        },
      });
    }

    const buildBody = (start?: number, end?: number) => {
      const nodeStream =
        start !== undefined && end !== undefined
          ? fs.createReadStream(resolved, { start, end })
          : fs.createReadStream(resolved);

      // Wire client disconnect → destroy the file stream so we don't keep
      // reading bytes for a request the browser has already cancelled.
      const onAbort = () => nodeStream.destroy();
      if (request.signal.aborted) {
        nodeStream.destroy();
      } else {
        request.signal.addEventListener('abort', onAbort, { once: true });
      }
      nodeStream.once('close', () => request.signal.removeEventListener('abort', onAbort));

      return Readable.toWeb(nodeStream) as unknown as ReadableStream;
    };

    // Support range requests for video/audio seeking
    const rangeHeader = request.headers.get('range');
    if (rangeHeader) {
      const parts = rangeHeader.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      const chunkSize = end - start + 1;

      return new NextResponse(buildBody(start, end) as any, {
        status: 206,
        headers: {
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': String(chunkSize),
          'Content-Type': contentType,
          'Cache-Control': cacheControl,
          ETag: etag,
        },
      });
    }

    return new NextResponse(buildBody() as any, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': cacheControl,
        'Accept-Ranges': 'bytes',
        ETag: etag,
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
