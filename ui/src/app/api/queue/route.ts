import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  try {
    const queues = await prisma.queue.findMany({
      orderBy: { gpu_ids: 'asc' },
    });
    return NextResponse.json({ queues: queues });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch queue' }, { status: 500 });
  }
}
