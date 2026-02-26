import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const { id } = await request.json();

    if (!id) {
      return NextResponse.json({ error: 'id is required' }, { status: 400 });
    }

    const download = await prisma.videoDownload.findUnique({ where: { id } });
    if (!download) {
      return NextResponse.json({ error: 'Download not found' }, { status: 404 });
    }

    await prisma.videoDownload.update({
      where: { id },
      data: { status: 'pending', progress: 0, error: '', speed: '' },
    });

    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to requeue download' }, { status: 500 });
  }
}
