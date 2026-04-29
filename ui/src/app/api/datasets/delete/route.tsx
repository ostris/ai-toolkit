import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getDataRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name } = body;
    let datasetsPath = await getDatasetsRoot();
    let datasetPath = path.join(datasetsPath, name);

    // if folder doesnt exist, ignore
    if (!fs.existsSync(datasetPath)) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    fs.rmSync(datasetPath, { recursive: true, force: true });

    // Also delete associated notes file if it exists
    try {
      const dataRoot = await getDataRoot();
      const notesPath = path.join(dataRoot, 'notes', `${name}.txt`);
      if (fs.existsSync(notesPath)) {
        fs.unlinkSync(notesPath);
      }
    } catch (notesError) {
      console.error('Error deleting notes file:', notesError);
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to delete dataset' }, { status: 500 });
  }
}
