import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id } = body;

    if (!id) {
      return NextResponse.json({ error: 'ID is required' }, { status: 400 });
    }

    const download = await prisma.videoDownload.findUnique({ where: { id } });
    if (!download) {
      return NextResponse.json({ error: 'Download not found' }, { status: 404 });
    }

    await prisma.videoDownload.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to delete download' }, { status: 500 });
  }
}
