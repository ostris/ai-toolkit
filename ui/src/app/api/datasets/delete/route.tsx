import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

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
    fs.rmdirSync(datasetPath, { recursive: true });
    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
