import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { isMac } from '@/helpers/basic';

const prisma = new PrismaClient();

function normalizeFlowGRPOJobConfig(jobConfig: any) {
  const process = jobConfig?.config?.process?.[0];
  if (process?.type !== 'flow_grpo_trainer') {
    return jobConfig;
  }
  process.grpo = process.grpo || {};
  process.train = process.train || {};
  process.sample = process.sample || {};
  process.train.disable_sampling = true;
  process.train.cache_text_embeddings = false;
  if (!process.train.noise_scheduler) {
    process.train.noise_scheduler = 'flowmatch';
  }
  if (!process.sample.sampler) {
    process.sample.sampler = 'flowmatch';
  }
  if (process.train.noise_scheduler !== 'flowmatch') {
    throw new Error(`Unsupported Flow-GRPO scheduler '${process.train.noise_scheduler}'. Supported values: flowmatch`);
  }
  if (process.sample.sampler !== 'flowmatch') {
    throw new Error(`Unsupported Flow-GRPO sampler '${process.sample.sampler}'. Supported values: flowmatch`);
  }
  process.sample.sample_every = 0;
  process.sample.samples = [];
  return jobConfig;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');

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

    const jobs = await prisma.job.findMany({
      where: job_type ? { job_type } : undefined,
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
    const { id, name } = body;
    const job_config = normalizeFlowGRPOJobConfig(body.job_config);
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
      // Update existing training
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          ...extra,
        },
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      // Create new training
      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
          ...extra,
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (typeof error?.message === 'string' && error.message.startsWith('Unsupported Flow-GRPO')) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
