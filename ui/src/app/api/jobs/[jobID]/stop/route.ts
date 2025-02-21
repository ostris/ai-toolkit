import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  // update job status to 'running'
  await prisma.job.update({
    where: { id: jobID },
    data: {
      stop: true,
      info: 'Stopping job...',
    },
  });

  return NextResponse.json(job);
}
