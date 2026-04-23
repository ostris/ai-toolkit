import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: Request, { params }: { params: { jobID: string } }) {
  const { searchParams } = new URL(request.url);
  const taskID = searchParams.get('taskID');

  try {
    const candidates = await prisma.flowGRPOCandidate.findMany({
      where: {
        job_id: params.jobID,
        ...(taskID ? { vote_task_id: taskID } : {}),
      },
      orderBy: [{ created_at: 'desc' }, { order_index: 'asc' }],
    });

    return NextResponse.json({
      candidates: candidates.map(candidate => ({
        ...candidate,
        image_url: `/api/img/${encodeURIComponent(candidate.image_path)}`,
      })),
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load Flow-GRPO candidates' }, { status: 500 });
  }
}
