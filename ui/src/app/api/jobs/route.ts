import { PrismaClient } from '@prisma/client';
import { NextRequest, NextResponse } from 'next/server';

const prisma = new PrismaClient();

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job);
    }

    const jobs = await prisma.job.findMany({
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { id, name, job_config, gpu_ids, use_multi_gpu, accelerate_config, num_gpus } = body;

    // Validate required fields
    if (!name || !job_config) {
      return NextResponse.json({ error: 'Name and job_config are required' }, { status: 400 });
    }

    // Validate GPU configuration
    if (use_multi_gpu) {
      if (!accelerate_config || num_gpus < 2) {
        return NextResponse.json({ error: 'Multi-GPU requires accelerate_config and num_gpus >= 2' }, { status: 400 });
      }
    } else {
      if (!gpu_ids) {
        return NextResponse.json({ error: 'Single GPU requires gpu_ids' }, { status: 400 });
      }
    }

    const jobData = {
      id,
      name,
      gpu_ids: gpu_ids || '',
      job_config: JSON.stringify(job_config),
      use_multi_gpu: use_multi_gpu || false,
      accelerate_config: accelerate_config ? JSON.stringify(accelerate_config) : null,
      num_gpus: num_gpus || 1,
    };

    const job = await prisma.job.upsert({
      where: { id: id || 'new' },
      update: jobData,
      create: jobData,
    });

    return NextResponse.json(job);
  } catch (error: any) {
    console.error('Error creating job:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
