import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

export async function GET(request: NextRequest, { params }: { params: { queueID: string } }) {
  const { queueID } = await params;

  const queue = await prisma.queue.findUnique({
    where: { gpu_ids: queueID },
  });

  if (!queue) {
    return NextResponse.json({ error: 'Queue not found' }, { status: 404 });
  }

  await prisma.queue.update({
    where: { id: queue.id },
    data: { is_running: false },
  });

  return NextResponse.json(queue);
}
