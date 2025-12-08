import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

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

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config, gpu_ids } = body;

    if (id) {
      // Update existing training
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
        },
      });
      return NextResponse.json(training);
    } else {
      // Create new training
      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
