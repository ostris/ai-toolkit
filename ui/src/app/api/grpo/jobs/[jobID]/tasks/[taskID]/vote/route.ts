import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request, { params }: { params: { jobID: string; taskID: string } }) {
  try {
    const body = await request.json();
    const action = body.action as 'select' | 'skip';
    const selectedCandidateId = body.selectedCandidateId as string | undefined;

    const task = await prisma.flowGRPOVoteTask.findFirst({
      where: {
        id: params.taskID,
        job_id: params.jobID,
      },
      include: {
        candidates: {
          orderBy: {
            order_index: 'asc',
          },
        },
        votes: true,
      },
    });

    if (!task) {
      return NextResponse.json({ error: 'Vote task not found' }, { status: 404 });
    }
    if (task.status !== 'open') {
      return NextResponse.json({ error: 'Vote task is no longer open' }, { status: 409 });
    }
    if (task.votes.length > 0) {
      return NextResponse.json({ error: 'Vote task already has votes' }, { status: 409 });
    }

    if (action === 'skip') {
      const vote = await prisma.$transaction(async tx => {
        const created = await tx.flowGRPOVote.create({
          data: {
            job_id: params.jobID,
            vote_task_id: params.taskID,
            value: 'skip',
            reward: 0,
          },
        });
        await tx.flowGRPOVoteTask.update({
          where: { id: params.taskID },
          data: { status: 'voted' },
        });
        return created;
      });
      return NextResponse.json({ ok: true, vote });
    }

    if (action !== 'select' || !selectedCandidateId) {
      return NextResponse.json({ error: 'selectedCandidateId is required for select votes' }, { status: 400 });
    }

    const selected = task.candidates.find(candidate => candidate.id === selectedCandidateId);
    if (!selected) {
      return NextResponse.json({ error: 'Selected candidate was not found on this task' }, { status: 400 });
    }

    const votes = await prisma.$transaction(async tx => {
      const createdVotes = await Promise.all(
        task.candidates.map(candidate =>
          tx.flowGRPOVote.create({
            data: {
              job_id: params.jobID,
              vote_task_id: params.taskID,
              candidate_id: candidate.id,
              value: candidate.id === selectedCandidateId ? 'selected' : 'not_selected',
              reward: candidate.id === selectedCandidateId ? 1 : 0,
            },
          }),
        ),
      );

      await tx.flowGRPOVoteTask.update({
        where: { id: params.taskID },
        data: { status: 'voted' },
      });

      await tx.flowGRPOCandidate.updateMany({
        where: { vote_task_id: params.taskID },
        data: { status: 'voted' },
      });

      return createdVotes;
    });

    return NextResponse.json({ ok: true, votes });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to submit Flow-GRPO vote' }, { status: 500 });
  }
}
