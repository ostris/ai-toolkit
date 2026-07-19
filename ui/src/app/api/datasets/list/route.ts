import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';

export async function GET() {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    try {
      await fs.promises.access(datasetsPath);
    } catch {
      await fs.promises.mkdir(datasetsPath);
    }

    // find all the folders in the datasets folder
    let folders = (await fs.promises.readdir(datasetsPath, { withFileTypes: true }))
      .filter(dirent => dirent.isDirectory())
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    return NextResponse.json(folders);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
