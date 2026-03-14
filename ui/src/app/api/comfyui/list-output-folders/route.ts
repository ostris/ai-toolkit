import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getComfyUIOutputDir } from '@/server/settings';

export async function GET() {
  try {
    const outputDir = await getComfyUIOutputDir();
    if (!outputDir) {
      return NextResponse.json({ error: 'COMFYUI_OUTPUT_DIR is not configured' }, { status: 400 });
    }

    if (!fs.existsSync(outputDir) || !fs.statSync(outputDir).isDirectory()) {
      return NextResponse.json({ error: 'ComfyUI output directory does not exist' }, { status: 400 });
    }

    const entries = fs.readdirSync(outputDir, { withFileTypes: true });
    const subfolders = entries
      .filter(e => e.isDirectory())
      .map(e => e.name)
      .sort();

    const folders = ['(root)', ...subfolders];

    return NextResponse.json({ folders });
  } catch (error) {
    console.error('Error listing ComfyUI output folders:', error);
    return NextResponse.json({ error: 'Failed to list output folders' }, { status: 500 });
  }
}
