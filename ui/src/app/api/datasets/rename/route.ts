import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

// Pattern replacement function
function applyPattern(pattern: string, index: number, totalFiles: number, datasetName: string, originalName: string, extension: string): string {
  const now = new Date();

  let result = pattern;

  // Index patterns - detect number of digits needed
  const indexMatches = pattern.match(/\{(#+)\}/g);
  if (indexMatches) {
    for (const match of indexMatches) {
      const digitCount = match.length - 2; // Remove { and }
      const paddedIndex = String(index).padStart(digitCount, '0');
      result = result.replace(match, paddedIndex);
    }
  }

  // Date/time patterns
  result = result.replace(/\{YYYY\}/g, now.getFullYear().toString());
  result = result.replace(/\{MM\}/g, String(now.getMonth() + 1).padStart(2, '0'));
  result = result.replace(/\{DD\}/g, String(now.getDate()).padStart(2, '0'));
  result = result.replace(/\{YYYYMMDD\}/g,
    `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`);
  result = result.replace(/\{HHMMSS\}/g,
    `${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`);
  result = result.replace(/\{timestamp\}/g, Math.floor(now.getTime() / 1000).toString());

  // Dataset and file patterns
  result = result.replace(/\{dataset\}/g, datasetName);
  result = result.replace(/\{original\}/g, originalName);
  result = result.replace(/\{ext\}/g, extension);

  // Random patterns
  result = result.replace(/\{random\}/g, Math.random().toString(36).substring(2, 8));
  result = result.replace(/\{uuid\}/g, crypto.randomUUID());

  // If pattern doesn't include {ext}, automatically append the extension
  if (!pattern.includes('{ext}')) {
    result += extension;
  }

  return result;
}

// Validate pattern function
function validatePattern(pattern: string): { valid: boolean; error?: string } {
  // Check if pattern contains at least one index pattern
  const hasIndex = /\{#{2,}\}/.test(pattern);
  if (!hasIndex) {
    return { valid: false, error: 'Pattern must contain at least one index pattern like {###}' };
  }

  // Check for invalid characters in filename
  const invalidChars = /[<>:"/\\|?*]/;
  if (invalidChars.test(pattern.replace(/\{[^}]+\}/g, ''))) {
    return { valid: false, error: 'Pattern contains invalid filename characters' };
  }

  return { valid: true };
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { datasetName, pattern } = body;

    if (!datasetName || !pattern) {
      return NextResponse.json({ error: 'Dataset name and pattern are required' }, { status: 400 });
    }

    // Validate the pattern
    const validation = validatePattern(pattern);
    if (!validation.valid) {
      return NextResponse.json({ error: validation.error }, { status: 400 });
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

    // Rename each media file using the pattern
    for (let i = 0; i < mediaFiles.length; i++) {
      const oldFile = mediaFiles[i];
      const oldFilePath = path.join(datasetFolder, oldFile);

      // Get the original file info
      const originalExt = path.extname(oldFile);
      const originalName = path.parse(oldFile).name;

      // Generate new filename using the pattern
      const newFileName = applyPattern(pattern, i + 1, mediaFiles.length, datasetName, originalName, originalExt);
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

// Preview endpoint to show what filenames would look like
export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const datasetName = url.searchParams.get('datasetName');
    const pattern = url.searchParams.get('pattern');

    if (!datasetName || !pattern) {
      return NextResponse.json({ error: 'Dataset name and pattern are required' }, { status: 400 });
    }

    // Validate the pattern
    const validation = validatePattern(pattern);
    if (!validation.valid) {
      return NextResponse.json({ error: validation.error }, { status: 400 });
    }

    const datasetsPath = await getDatasetsRoot();
    const datasetFolder = path.join(datasetsPath, datasetName);

    // Check if dataset folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Dataset '${datasetName}' not found` }, { status: 404 });
    }

    // Get all media files
    const files = fs.readdirSync(datasetFolder);
    const mediaExtensions = [
      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg',
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

    // Generate preview for first 5 files
    const previews = mediaFiles.slice(0, 5).map((file, index) => {
      const originalExt = path.extname(file);
      const originalName = path.parse(file).name;
      const newFileName = applyPattern(pattern, index + 1, mediaFiles.length, datasetName, originalName, originalExt);

      return {
        oldName: file,
        newName: newFileName
      };
    });

    return NextResponse.json({
      success: true,
      previews,
      totalFiles: mediaFiles.length,
      pattern
    });

  } catch (error) {
    console.error('Preview error:', error);
    return NextResponse.json({ error: 'Failed to generate preview' }, { status: 500 });
  }
}
