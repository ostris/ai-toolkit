import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';

export async function GET() {
  const presetsDir = path.join(TOOLKIT_ROOT, 'caption_presets');

  if (!fs.existsSync(presetsDir)) {
    return NextResponse.json({ presets: [] });
  }

  const files = fs.readdirSync(presetsDir, { withFileTypes: true });
  const presets = files
    .filter(f => f.isFile() && f.name.endsWith('.txt') && !f.name.startsWith('.'))
    .map(f => {
      const filePath = path.join(presetsDir, f.name);
      const content = fs.readFileSync(filePath, 'utf-8');
      const name = f.name.replace(/\.txt$/, '');
      return { name, content };
    })
    .sort((a, b) => a.name.localeCompare(b.name));

  return NextResponse.json({ presets });
}
