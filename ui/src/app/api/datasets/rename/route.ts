import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

function sanitize(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, newName } = body;
    if (!name || !newName) {
      return NextResponse.json({ error: 'name and newName are required' }, { status: 400 });
    }
    const cleaned = sanitize(newName);
    if (!cleaned) {
      return NextResponse.json({ error: 'Invalid new dataset name' }, { status: 400 });
    }

    const root = await getDatasetsRoot();
    const from = path.join(root, name);
    const to = path.join(root, cleaned);

    if (!fs.existsSync(from)) {
      return NextResponse.json({ error: `Dataset '${name}' not found` }, { status: 404 });
    }
    if (fs.existsSync(to) && path.resolve(from) !== path.resolve(to)) {
      return NextResponse.json({ error: `A dataset named '${cleaned}' already exists` }, { status: 409 });
    }

    fs.renameSync(from, to);
    return NextResponse.json({ success: true, name: cleaned });
  } catch (error) {
    console.error('Error renaming dataset:', error);
    return NextResponse.json({ error: 'Failed to rename dataset' }, { status: 500 });
  }
}
