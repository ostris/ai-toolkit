// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: NextRequest) {
  try {
    const datasetsPath = await getDatasetsRoot();
    if (!datasetsPath) {
      return NextResponse.json({ error: 'Datasets path not found' }, { status: 500 });
    }
    const formData = await request.formData();
    const files = formData.getAll('files');
    const datasetName = formData.get('datasetName') as string;

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    if (!datasetName) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    // Security check: normalize the path and validate it
    const normalizedPath = path.normalize(datasetName).replace(/^(\.\.(\/|\\|$))+/, '');
    const uploadDir = path.join(datasetsPath, normalizedPath);

    // Ensure paths are within datasetsPath (to prevent path traversal attacks)
    const resolvedPath = path.resolve(uploadDir);
    const resolvedBase = path.resolve(datasetsPath);
    if (!resolvedPath.startsWith(resolvedBase)) {
      return NextResponse.json({ error: 'Invalid path' }, { status: 400 });
    }

    // Recursively create directories
    await mkdir(uploadDir, { recursive: true });

    const savedFiles: string[] = [];

    // Process files sequentially to avoid overwhelming the system
    for (let i = 0; i < files.length; i++) {
      const file = files[i] as any;
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);

      // Clean filename and ensure it's unique
      const fileName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
      const filePath = path.join(uploadDir, fileName);

      await writeFile(filePath, buffer);
      savedFiles.push(fileName);
    }

    return NextResponse.json({
      message: 'Files uploaded successfully',
      files: savedFiles,
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: 'Error uploading files' }, { status: 500 });
  }
}

// Increase payload size limit (default is 4mb)
export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
  },
};
