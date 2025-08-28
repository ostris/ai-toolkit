import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const { status } = await request.json();

  // Validate the status
  const validStatuses = ['stopped', 'completed', 'error'];
  if (!validStatuses.includes(status)) {
    return NextResponse.json({ error: 'Invalid status' }, { status: 400 });
  }

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // Force update the job status
  const updatedJob = await prisma.job.update({
    where: { id: jobID },
    data: {
      status: status,
      stop: false,
      info: status === 'stopped' ? 'Job stopped' : 
            status === 'completed' ? 'Training completed' :
            'Job error',
    },
  });

  return NextResponse.json(updatedJob);
}
