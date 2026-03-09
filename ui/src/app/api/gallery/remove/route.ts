import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id } = body;

    if (!id || typeof id !== 'number') {
      return NextResponse.json({ error: 'id is required' }, { status: 400 });
    }

    await prisma.galleryFolder.delete({
      where: { id },
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error removing gallery folder:', error);
    return NextResponse.json({ error: 'Failed to remove gallery folder' }, { status: 500 });
  }
}
