import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import { promisify } from 'util';
import { getDatasetsRoot } from '@/server/settings';

const brotliCompress = promisify(zlib.brotliCompress);
const gzipCompress = promisify(zlib.gzip);

export async function POST(request: Request) {
  const datasetsPath = await getDatasetsRoot();
  const body = await request.json();
  const { datasetName } = body;
  const datasetFolder = path.join(datasetsPath, datasetName);

  try {
    // Check if folder exists
    try {
      await fs.promises.access(datasetFolder);
    } catch {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    // Find all images recursively
    const imageFiles = await findImagesRecursively(datasetFolder);

    // Sort server-side so the client doesn't have to sort large lists
    imageFiles.sort((a, b) => a.localeCompare(b));

    // Send a single shared root plus each file's sub-path, rather than repeating the full
    // absolute path (and an "img_path" key) on every entry. The root carries the trailing
    // OS separator so the client rebuilds the native path with a plain concat (root + subPath),
    // keeping correct paths on Windows, macOS, and Linux without any separator logic client-side.
    const root = datasetFolder + path.sep;
    const result = imageFiles.map(imgPath => imgPath.slice(root.length));

    // Compress the payload explicitly. Even after stripping the shared prefix these lists
    // still gzip/brotli down substantially — a big win on slow connections and huge datasets.
    // The browser decompresses transparently via Content-Encoding, so the client just parses JSON.
    const json = JSON.stringify({ root, images: result });
    const acceptEncoding = request.headers.get('accept-encoding') ?? '';

    if (/\bbr\b/.test(acceptEncoding)) {
      const body = await brotliCompress(json);
      return new NextResponse(body as any, {
        headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'br' },
      });
    }
    if (/\bgzip\b/.test(acceptEncoding)) {
      const body = await gzipCompress(json);
      return new NextResponse(body as any, {
        headers: { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' },
      });
    }

    return new NextResponse(json, { headers: { 'Content-Type': 'application/json' } });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}

/**
 * Recursively finds all image files in a directory and its subdirectories
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
async function findImagesRecursively(dir: string): Promise<string[]> {
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.mp3', '.wav', '.flac', '.ogg'];
  let results: string[] = [];

  // withFileTypes avoids a separate stat per entry — a big win on large datasets.
  // Async readdir yields between directories so other requests aren't blocked.
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });

  const subdirs: string[] = [];
  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;
    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      subdirs.push(itemPath);
    } else if (entry.isFile()) {
      const ext = path.extname(name).toLowerCase();
      if (imageExtensions.includes(ext)) {
        results.push(itemPath);
      }
    }
  }

  // Recurse into subdirectories concurrently.
  const nested = await Promise.all(subdirs.map(subdir => findImagesRecursively(subdir)));
  for (const list of nested) {
    results = results.concat(list);
  }

  return results;
}
