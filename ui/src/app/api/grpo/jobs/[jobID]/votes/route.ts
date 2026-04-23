import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: Request, { params }: { params: { jobID: string } }) {
  const { searchParams } = new URL(request.url);
  const processed = searchParams.get('processed');

  try {
    const votes = await prisma.flowGRPOVote.findMany({
      where: {
        job_id: params.jobID,
        ...(processed == null ? {} : { processed: processed === 'true' }),
      },
      orderBy: {
        created_at: 'desc',
      },
    });

    return NextResponse.json({ votes });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load Flow-GRPO votes' }, { status: 500 });
  }
}
