/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { filePath } = body;

    if (!filePath || typeof filePath !== 'string') {
      return new NextResponse('filePath is required', { status: 400 });
    }

    // Decode the path
    const decodedFilePath = decodeURIComponent(filePath);

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot];

    // Security check: resolve so `..` segments collapse, then verify still under
    // an allowed root. Substring `.includes('..')` false-positives on filenames
    // containing `..` as text (e.g. an ellipsis in a filename).
    const resolvedFilePath = path.resolve(decodedFilePath);
    const isAllowed = allowedDirs.some(
      allowedDir => resolvedFilePath === allowedDir || resolvedFilePath.startsWith(allowedDir + path.sep),
    );

    if (!isAllowed) {
      console.warn(`Access denied: ${resolvedFilePath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Check if file exists
    if (!fs.existsSync(resolvedFilePath)) {
      console.warn(`File not found: ${resolvedFilePath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    // Get file info
    const stat = fs.statSync(resolvedFilePath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    fs.unlinkSync(resolvedFilePath);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting file:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
