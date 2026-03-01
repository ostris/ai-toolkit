import { NextResponse } from 'next/server';
import { execFile } from 'child_process';
import path from 'path';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { datasetName } = body;

    if (!datasetName || typeof datasetName !== 'string') {
      return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const folderPath = path.resolve(path.join(datasetsPath, datasetName));

    // Prevent path traversal
    if (!folderPath.startsWith(path.resolve(datasetsPath))) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    if (!fs.existsSync(folderPath)) {
      return NextResponse.json({ error: 'Folder not found' }, { status: 404 });
    }

    const platform = process.platform;
    let bin: string;
    if (platform === 'darwin') {
      bin = 'open';
    } else if (platform === 'win32') {
      bin = 'explorer';
    } else {
      bin = 'xdg-open';
    }

    await new Promise<void>((resolve, reject) => {
      execFile(bin, [folderPath], error => {
        if (error) {
          // explorer.exe on Windows often returns exit code 1 even on success
          if (platform === 'win32' && error.code === 1) {
            resolve();
          } else {
            reject(error);
          }
        } else {
          resolve();
        }
      });
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error opening folder:', error);
    return NextResponse.json({ error: 'Failed to open folder' }, { status: 500 });
  }
}
