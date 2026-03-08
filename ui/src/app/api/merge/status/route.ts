import { NextResponse } from 'next/server';
import fs from 'fs';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const statusFile = searchParams.get('statusFile');

  if (!statusFile) {
    return NextResponse.json({ error: 'statusFile param is required' }, { status: 400 });
  }

  try {
    if (!fs.existsSync(statusFile)) {
      return NextResponse.json({ status: 'unknown' });
    }
    const raw = fs.readFileSync(statusFile, 'utf-8');
    const data = JSON.parse(raw);
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ status: 'unknown' });
  }
}
