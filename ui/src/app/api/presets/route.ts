import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const STORAGE_KEY = 'USER_PRESETS';

export async function GET() {
  try {
    const row = await prisma.settings.findUnique({ where: { key: STORAGE_KEY } });
    if (!row) return NextResponse.json({ presets: [] });
    try {
      const parsed = JSON.parse(row.value);
      return NextResponse.json({ presets: Array.isArray(parsed.presets) ? parsed.presets : [] });
    } catch {
      return NextResponse.json({ presets: [] });
    }
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load presets' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const payload = { presets: Array.isArray(body.presets) ? body.presets : [] };
    await prisma.settings.upsert({
      where: { key: STORAGE_KEY },
      update: { value: JSON.stringify(payload) },
      create: { key: STORAGE_KEY, value: JSON.stringify(payload) },
    });
    return NextResponse.json(payload);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to save presets' }, { status: 500 });
  }
}
