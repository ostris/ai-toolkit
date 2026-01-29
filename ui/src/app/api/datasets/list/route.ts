import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetsPath)) {
      fs.mkdirSync(datasetsPath);
    }

    // find all the folders in the datasets folder (including symlinks)
    let folders = fs
      .readdirSync(datasetsPath, { withFileTypes: true })
      .filter(dirent => {
        // Check if it's a directory or a symlink pointing to a directory
        if (dirent.isDirectory()) return true;
        if (dirent.isSymbolicLink()) {
          const fullPath = `${datasetsPath}/${dirent.name}`;
          try {
            return fs.statSync(fullPath).isDirectory();
          } catch {
            return false;
          }
        }
        return false;
      })
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    return NextResponse.json(folders);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
