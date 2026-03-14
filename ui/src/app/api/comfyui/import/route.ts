import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getComfyUIOutputDir } from '@/server/settings';
import { getDatasetsRoot } from '@/server/settings';

const SUPPORTED_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif',
  '.mp4', '.avi', '.mov', '.mkv', '.webm',
  '.txt', '.caption',
]);

function copyFilesRecursive(src: string, dest: string) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyFilesRecursive(srcPath, destPath);
    } else {
      const ext = path.extname(entry.name).toLowerCase();
      if (SUPPORTED_EXTENSIONS.has(ext)) {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, folder } = body;

    if (!name || typeof name !== 'string') {
      return NextResponse.json({ error: 'name is required' }, { status: 400 });
    }
    if (!folder || typeof folder !== 'string') {
      return NextResponse.json({ error: 'folder is required' }, { status: 400 });
    }

    const outputDir = await getComfyUIOutputDir();
    if (!outputDir) {
      return NextResponse.json({ error: 'COMFYUI_OUTPUT_DIR is not configured' }, { status: 400 });
    }

    const sanitizedName = name.toLowerCase().replace(/[^a-z0-9]+/g, '_');

    const sourcePath = folder === '(root)' ? outputDir : path.join(outputDir, folder);

    if (folder !== '(root)' && (folder.includes('..') || folder.includes('/'))) {
      return NextResponse.json({ error: 'Invalid folder name' }, { status: 400 });
    }

    if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isDirectory()) {
      return NextResponse.json({ error: 'Source folder not found' }, { status: 404 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const datasetPath = path.join(datasetsRoot, sanitizedName);

    if (!fs.existsSync(datasetsRoot)) {
      fs.mkdirSync(datasetsRoot, { recursive: true });
    }

    copyFilesRecursive(sourcePath, datasetPath);

    return NextResponse.json({ success: true, name: sanitizedName });
  } catch (error) {
    console.error('Error importing from ComfyUI:', error);
    return NextResponse.json({ error: 'Failed to import from ComfyUI' }, { status: 500 });
  }
}
