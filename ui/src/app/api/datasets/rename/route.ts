import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { datasetName, newBaseName } = body;
    
    if (!datasetName || !newBaseName) {
      return NextResponse.json({ error: 'Dataset name and new base name are required' }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const datasetFolder = path.join(datasetsPath, datasetName);

    // Check if dataset folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Dataset '${datasetName}' not found` }, { status: 404 });
    }

    // Get all media files (images and videos) in the dataset folder
    const files = fs.readdirSync(datasetFolder);
    const mediaExtensions = [
      // Image formats
      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg',
      // Video formats
      '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'
    ];
    const mediaFiles = files.filter(file => {
      const ext = path.extname(file).toLowerCase();
      return mediaExtensions.includes(ext);
    });

    if (mediaFiles.length === 0) {
      return NextResponse.json({ error: 'No media files found in dataset' }, { status: 400 });
    }

    // Sort files to ensure consistent ordering
    mediaFiles.sort();

    const renamedFiles: { oldName: string; newName: string }[] = [];
    const errors: string[] = [];

    // Rename each media file with the new base name + counter
    for (let i = 0; i < mediaFiles.length; i++) {
      const oldFile = mediaFiles[i];
      const oldFilePath = path.join(datasetFolder, oldFile);

      // Get the original file extension
      const originalExt = path.extname(oldFile);

      // Generate new filename with 3-digit counter and original extension
      const counter = String(i + 1).padStart(3, '0');
      const newFileName = `${newBaseName}${counter}${originalExt}`;
      const newFilePath = path.join(datasetFolder, newFileName);

      try {
        // Check if new filename already exists and is different from current file
        if (fs.existsSync(newFilePath) && oldFile !== newFileName) {
          errors.push(`File ${newFileName} already exists`);
          continue;
        }

        // Only rename if the names are different
        if (oldFile !== newFileName) {
          fs.renameSync(oldFilePath, newFilePath);
          renamedFiles.push({ oldName: oldFile, newName: newFileName });

          // Also rename associated caption file if it exists
          const oldCaptionPath = path.join(datasetFolder, path.parse(oldFile).name + '.txt');
          const newCaptionPath = path.join(datasetFolder, path.parse(newFileName).name + '.txt');

          if (fs.existsSync(oldCaptionPath)) {
            fs.renameSync(oldCaptionPath, newCaptionPath);
          }
        }
      } catch (error) {
        errors.push(`Failed to rename ${oldFile}: ${error}`);
      }
    }

    return NextResponse.json({
      success: true,
      renamedFiles,
      errors,
      totalProcessed: mediaFiles.length,
      totalRenamed: renamedFiles.length
    });

  } catch (error) {
    console.error('Rename error:', error);
    return NextResponse.json({ error: 'Failed to rename files' }, { status: 500 });
  }
}
