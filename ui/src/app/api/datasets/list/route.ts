import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

/**
 * Recursively get all subfolders (returns relative paths)
 * @param baseDir Base directory
 * @param currentPath Current relative path
 * @returns Array of relative paths to all subfolders
 */
function getAllSubfolders(baseDir: string, currentPath: string = ''): string[] {
  const results: string[] = [];
  const fullPath = path.join(baseDir, currentPath);

  if (!fs.existsSync(fullPath)) return results;

  const items = fs.readdirSync(fullPath, { withFileTypes: true });

  for (const item of items) {
    if (item.isDirectory() && !item.name.startsWith('.') && item.name !== '_controls') {
      const relativePath = currentPath ? `${currentPath}/${item.name}` : item.name;
      results.push(relativePath);
      // Recursively get subfolders
      results.push(...getAllSubfolders(baseDir, relativePath));
    }
  }

  return results;
}

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetsPath)) {
      fs.mkdirSync(datasetsPath, { recursive: true });
    }

    // Recursively get all folders (including nested)
    const folders = getAllSubfolders(datasetsPath);

    return NextResponse.json(folders);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
