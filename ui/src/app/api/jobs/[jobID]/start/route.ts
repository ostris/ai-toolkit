import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // get highest queue position
  const highestQueuePosition = await prisma.job.aggregate({
    _max: {
      queue_position: true,
    },
  });
  const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

  await prisma.job.update({
    where: { id: jobID },
    data: { queue_position: newQueuePosition },
  });

  // make sure the queue is running
  const queue = await prisma.queue.findFirst({
    where: {
      gpu_ids: job.gpu_ids,
    },
  });

  // if queue doesn't exist, create it
  if (!queue) {
    await prisma.queue.create({
      data: {
        gpu_ids: job.gpu_ids,
        is_running: false,
      },
    });
  }

  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'queued',
      stop: false,
      return_to_queue: false,
      info: 'Job queued',
    },
  });

  // Return the response immediately
  return NextResponse.json(job);
}
