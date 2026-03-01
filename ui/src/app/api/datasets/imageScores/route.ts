import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const imgPath = searchParams.get('imgPath');

  if (!imgPath || typeof imgPath !== 'string' || imgPath.trim() === '') {
    return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
  }

  // Security: prevent path traversal
  if (imgPath.includes('..')) {
    return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
  }

  const datasetsPath = await getDatasetsRoot();
  const trainingPath = await getTrainingFolder();

  // Security: ensure path is within an allowed directory
  if (!imgPath.startsWith(datasetsPath) && !imgPath.startsWith(trainingPath)) {
    return NextResponse.json({ error: 'Access denied' }, { status: 403 });
  }

  const parsed = path.parse(imgPath);
  const csvPath = path.format({ dir: parsed.dir, name: parsed.name, ext: '.csv' });

  if (!fs.existsSync(csvPath)) {
    return NextResponse.json({ scores: {} });
  }

  try {
    const content = fs.readFileSync(csvPath, 'utf-8');
    const scores: Record<string, number> = {};
    const lines = content.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      // Simple CSV parsing (metric,value)
      const commaIndex = trimmed.indexOf(',');
      if (commaIndex === -1) continue;
      const metric = trimmed.slice(0, commaIndex).trim();
      const valueStr = trimmed.slice(commaIndex + 1).trim();
      const value = parseFloat(valueStr);
      if (metric && !isNaN(value)) {
        scores[metric] = value;
      }
    }
    return NextResponse.json({ scores });
  } catch (error) {
    console.error('Error reading scores CSV:', error);
    return NextResponse.json({ error: 'Failed to read scores' }, { status: 500 });
  }
}
