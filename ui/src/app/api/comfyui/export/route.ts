import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getComfyUIInputDir } from '@/server/settings';
import { getDatasetsRoot } from '@/server/settings';

function copyDirRecursive(src: string, dest: string) {
  fs.mkdirSync(dest, { recursive: true });
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { datasetName } = body;

    if (!datasetName || typeof datasetName !== 'string') {
      return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
    }

    const inputDir = await getComfyUIInputDir();
    if (!inputDir) {
      return NextResponse.json({ error: 'COMFYUI_INPUT_DIR is not configured' }, { status: 400 });
    }

    if (datasetName.includes('..')) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const datasetPath = path.join(datasetsRoot, datasetName);

    if (!fs.existsSync(datasetPath) || !fs.statSync(datasetPath).isDirectory()) {
      return NextResponse.json({ error: 'Dataset not found' }, { status: 404 });
    }

    const destPath = path.join(inputDir, datasetName);
    copyDirRecursive(datasetPath, destPath);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error exporting to ComfyUI:', error);
    return NextResponse.json({ error: 'Failed to export to ComfyUI' }, { status: 500 });
  }
}
