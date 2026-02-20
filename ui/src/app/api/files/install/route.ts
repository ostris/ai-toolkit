import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder, getLoraInstallPath } from '@/server/settings';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { filePath } = body;

    if (!filePath || typeof filePath !== 'string') {
      return NextResponse.json({ error: 'filePath is required' }, { status: 400 });
    }

    const loraInstallPath = await getLoraInstallPath();
    if (!loraInstallPath) {
      return NextResponse.json({ error: 'LoRA Install Path is not configured in settings' }, { status: 400 });
    }

    // Security check: ensure source path is in allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot];

    const isAllowed =
      allowedDirs.some(allowedDir => filePath.startsWith(allowedDir)) && !filePath.includes('..');

    if (!isAllowed) {
      return NextResponse.json({ error: 'Access denied' }, { status: 403 });
    }

    // Verify source file exists
    if (!fs.existsSync(filePath)) {
      return NextResponse.json({ error: 'Source file not found' }, { status: 404 });
    }

    const stat = fs.statSync(filePath);
    if (!stat.isFile()) {
      return NextResponse.json({ error: 'Source path is not a file' }, { status: 400 });
    }

    // Verify destination directory exists
    if (!fs.existsSync(loraInstallPath)) {
      return NextResponse.json({ error: `Install directory does not exist: ${loraInstallPath}` }, { status: 400 });
    }

    const destStat = fs.statSync(loraInstallPath);
    if (!destStat.isDirectory()) {
      return NextResponse.json({ error: 'Install path is not a directory' }, { status: 400 });
    }

    const fileName = path.basename(filePath);
    const destPath = path.join(loraInstallPath, fileName);

    await fs.promises.copyFile(filePath, destPath);

    return NextResponse.json({ success: true, destPath });
  } catch (error) {
    console.error('Error installing file:', error);
    return NextResponse.json({ error: 'Failed to install file' }, { status: 500 });
  }
}
