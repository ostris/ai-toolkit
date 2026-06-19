import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();
const STORAGE_KEY = 'PROMPT_LIBRARY';

const emptyLibrary = { prompts: [], sets: [] };

export async function GET() {
  try {
    const row = await prisma.settings.findUnique({ where: { key: STORAGE_KEY } });
    if (!row) return NextResponse.json(emptyLibrary);
    try {
      const parsed = JSON.parse(row.value);
      return NextResponse.json({
        prompts: Array.isArray(parsed.prompts) ? parsed.prompts : [],
        sets: Array.isArray(parsed.sets) ? parsed.sets : [],
      });
    } catch {
      return NextResponse.json(emptyLibrary);
    }
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load prompt library' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const payload = {
      prompts: Array.isArray(body.prompts) ? body.prompts : [],
      sets: Array.isArray(body.sets) ? body.sets : [],
    };
    await prisma.settings.upsert({
      where: { key: STORAGE_KEY },
      update: { value: JSON.stringify(payload) },
      create: { key: STORAGE_KEY, value: JSON.stringify(payload) },
    });
    return NextResponse.json(payload);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to save prompt library' }, { status: 500 });
  }
}
