// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
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

    // Create upload directory if it doesn't exist
    const uploadDir = join(datasetsPath, datasetName);
    await mkdir(uploadDir, { recursive: true });

    const savedFiles = await Promise.all(
      files.map(async (file: any) => {
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);

        // Clean filename and ensure it's unique
        const fileName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
        const filePath = join(uploadDir, fileName);

        await writeFile(filePath, buffer);
        return fileName;
      }),
    );

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
