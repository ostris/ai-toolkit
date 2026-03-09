import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const folders = await prisma.galleryFolder.findMany({
      orderBy: { created_at: 'asc' },
    });
    return NextResponse.json(folders);
  } catch (error) {
    console.error('Error fetching gallery folders:', error);
    return NextResponse.json({ error: 'Failed to fetch gallery folders' }, { status: 500 });
  }
}
