// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { getDataRoot } from '@/server/settings';
import {v4 as uuidv4} from 'uuid';

export async function POST(request: NextRequest) {
  try {
    const dataRoot = await getDataRoot();
    if (!dataRoot) {
      return NextResponse.json({ error: 'Data root path not found' }, { status: 500 });
    }
    const imgRoot = join(dataRoot, 'images');


    const formData = await request.formData();
    const files = formData.getAll('files');

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    // make it recursive if it doesn't exist
    await mkdir(imgRoot, { recursive: true });
    const savedFiles = await Promise.all(
      files.map(async (file: any) => {
        const bytes = await file.arrayBuffer();
        const buffer = Buffer.from(bytes);

        const extension = file.name.split('.').pop() || 'jpg';

        // Clean filename and ensure it's unique
        const fileName = `${uuidv4()}`; // Use UUID for unique file names
        const filePath = join(imgRoot, `${fileName}.${extension}`);

        await writeFile(filePath, buffer);
        return filePath;
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
