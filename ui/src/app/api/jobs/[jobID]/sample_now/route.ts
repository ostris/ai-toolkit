import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.update({
    where: { id: jobID },
    data: {
      sample_now: true,
    },
  });

  console.log(`Job ${jobID} marked to sample on next step`);

  return NextResponse.json(job);
}
