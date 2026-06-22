import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { queueID: string } }) {
  const { queueID } = await params;

  const queue = await prisma.queue.findUnique({
    where: { gpu_ids: queueID },
  });

  if (!queue) {
    // create it if it doesn't exist
    const newQueue = await prisma.queue.create({
      data: { gpu_ids: queueID, is_running: true },
    });
    return NextResponse.json(newQueue);
  }

  await prisma.queue.update({
    where: { id: queue.id },
    data: { is_running: true },
  });

  return NextResponse.json(queue);
}
