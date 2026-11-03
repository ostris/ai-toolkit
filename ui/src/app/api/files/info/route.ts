/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

// safetensors header: u64 LE byte length, then that many bytes of JSON.
const MAX_HEADER_BYTES = 256 * 1024 * 1024;

const DTYPE_BYTES: Record<string, number> = {
  F64: 8, I64: 8, U64: 8,
  F32: 4, I32: 4, U32: 4,
  F16: 2, BF16: 2, I16: 2, U16: 2,
  F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, I8: 1, U8: 1, BOOL: 1,
  F4: 0.5, U4: 0.5, I4: 0.5,
};

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { filePath } = body;

    if (!filePath || typeof filePath !== 'string') {
      return new NextResponse('filePath is required', { status: 400 });
    }

    const decodedFilePath = decodeURIComponent(filePath);
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot];

    const resolvedFilePath = path.resolve(decodedFilePath);
    const isAllowed = allowedDirs.some(
      allowedDir => resolvedFilePath === allowedDir || resolvedFilePath.startsWith(allowedDir + path.sep),
    );
    if (!isAllowed) {
      console.warn(`Access denied: ${resolvedFilePath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    let stat;
    try {
      stat = await fs.promises.stat(resolvedFilePath);
    } catch {
      return new NextResponse('File not found', { status: 404 });
    }
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }
    if (stat.size < 8) {
      return new NextResponse('Not a safetensors file', { status: 400 });
    }

    const handle = await fs.promises.open(resolvedFilePath, 'r');
    let header: Record<string, any>;
    let headerSize: number;
    try {
      const lenBuf = Buffer.alloc(8);
      await handle.read(lenBuf, 0, 8, 0);
      const headerLen = lenBuf.readBigUInt64LE(0);
      if (headerLen > BigInt(MAX_HEADER_BYTES) || headerLen > BigInt(stat.size - 8)) {
        return new NextResponse('Not a safetensors file (bad header length)', { status: 400 });
      }
      headerSize = Number(headerLen);
      const headerBuf = Buffer.alloc(headerSize);
      await handle.read(headerBuf, 0, headerSize, 8);
      header = JSON.parse(headerBuf.toString('utf8'));
    } catch (e) {
      return new NextResponse('Not a safetensors file (unparseable header)', { status: 400 });
    } finally {
      await handle.close();
    }

    const metadata: Record<string, string> = header.__metadata__ ?? {};
    const tensors: { name: string; dtype: string; shape: number[]; numel: number; bytes: number }[] = [];
    const dtypeCounts: Record<string, { count: number; numel: number; bytes: number }> = {};
    let totalParams = 0;
    let totalBytes = 0;

    for (const [name, info] of Object.entries(header)) {
      if (name === '__metadata__') continue;
      const t = info as { dtype: string; shape: number[]; data_offsets?: [number, number] };
      const shape = Array.isArray(t.shape) ? t.shape : [];
      const numel = shape.reduce((a, b) => a * b, 1);
      const bytes = t.data_offsets ? t.data_offsets[1] - t.data_offsets[0] : numel * (DTYPE_BYTES[t.dtype] ?? 0);
      tensors.push({ name, dtype: t.dtype, shape, numel, bytes });
      totalParams += numel;
      totalBytes += bytes;
      const d = (dtypeCounts[t.dtype] ??= { count: 0, numel: 0, bytes: 0 });
      d.count += 1;
      d.numel += numel;
      d.bytes += bytes;
    }
    tensors.sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({
      filePath: resolvedFilePath,
      fileSize: stat.size,
      headerSize,
      metadata,
      tensors,
      tensorCount: tensors.length,
      totalParams,
      totalBytes,
      dtypeCounts,
    });
  } catch (error) {
    console.error('Error reading safetensors info:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
