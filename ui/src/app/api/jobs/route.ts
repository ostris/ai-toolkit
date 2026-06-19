import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { isMac } from '@/helpers/basic';

const prisma = new PrismaClient();

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const status = searchParams.get('status');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return NextResponse.json(job);
    }

    const where: any = {};
    if (job_type) where.job_type = job_type;
    if (status) {
      if (status.includes(',')) {
        where.status = { in: status.split(',').map(s => s.trim()).filter(Boolean) };
      } else {
        where.status = status;
      }
    }

    const jobs = await prisma.job.findMany({
      where: Object.keys(where).length > 0 ? where : undefined,
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config, status } = body;
    let gpu_ids: string = body.gpu_ids;

    if (isMac()) {
      gpu_ids = "mps";
    }

    const extra: any = {};
    if ("job_ref" in body) {
      extra["job_ref"] = body.job_ref;
    }
    if ("job_type" in body) {
      extra["job_type"] = body.job_type;
    }

    if (id) {
      const data: any = {
        name,
        gpu_ids,
        job_config: JSON.stringify(job_config),
        ...extra,
      };
      if (status) data.status = status;
      const training = await prisma.job.update({
        where: { id },
        data,
      });
      return NextResponse.json(training);
    } else {
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
          status: status || 'stopped',
          ...extra,
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
