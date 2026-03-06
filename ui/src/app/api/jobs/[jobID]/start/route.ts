import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { isLtxJobConfig, isLtxOnlyMode } from '@/server/ltxOnly';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (isLtxOnlyMode()) {
    try {
      const parsedConfig = JSON.parse(job.job_config);
      if (!isLtxJobConfig(parsedConfig)) {
        return NextResponse.json(
          {
            error:
              'LTX-only mode is enabled. Non-LTX training jobs are blocked. Set AITK_ALLOW_NON_LTX=1 to override.',
          },
          { status: 400 },
        );
      }
    } catch {
      return NextResponse.json({ error: 'Invalid job config JSON' }, { status: 400 });
    }
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
