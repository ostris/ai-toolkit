import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const scalarVotes: Record<string, number> = {
  up: 1,
  down: -1,
  skip: 0,
};

export async function POST(request: Request, { params }: { params: { jobID: string; taskID: string } }) {
  try {
    const body = await request.json();
    const rewards = Array.isArray(body.rewards) ? body.rewards : null;

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
      return NextResponse.json({ error: 'Task not found' }, { status: 404 });
    }
    if (task.status !== 'open') {
      return NextResponse.json({ error: 'Task is no longer open' }, { status: 409 });
    }
    if (task.votes.length > 0) {
      return NextResponse.json({ error: 'Task already has a vote' }, { status: 409 });
    }
    if (task.candidates.length < 2) {
      return NextResponse.json({ error: 'Flow-GRPO live tasks must contain a generated rollout group' }, { status: 409 });
    }

    const candidateById = new Map(task.candidates.map(candidate => [candidate.id, candidate]));
    const providedIds = new Set<string>();
    const submittedVotes = rewards || [{
      candidate_id: body.candidate_id,
      value: body.value,
      reward: body.reward,
    }];
    const normalizedRewards: Array<{ candidate_id: string; value: string; reward: number }> = [];
    for (const item of submittedVotes) {
      const candidateID = `${item?.candidate_id ?? ''}`.trim();
      const voteValue = `${item?.value ?? ''}`.trim().toLowerCase();
      const rewardValue = voteValue in scalarVotes ? scalarVotes[voteValue] : Number(item?.reward);
      if (!candidateID || !candidateById.has(candidateID)) {
        return NextResponse.json({ error: 'All votes must reference valid candidate IDs' }, { status: 400 });
      }
      if (!Number.isFinite(rewardValue)) {
        return NextResponse.json({ error: 'Vote must be Up, Down, Skip, or a finite reward' }, { status: 400 });
      }
      if (providedIds.has(candidateID)) {
        return NextResponse.json({ error: 'Duplicate candidate vote submitted' }, { status: 400 });
      }
      providedIds.add(candidateID);
      normalizedRewards.push({
        candidate_id: candidateID,
        value: voteValue in scalarVotes ? voteValue : 'reward',
        reward: rewardValue,
      });
    }
    if (normalizedRewards.length !== task.candidates.length) {
      return NextResponse.json({ error: 'A vote must be provided for every generated candidate' }, { status: 400 });
    }
    const votes = await prisma.$transaction(async tx => {
      const createdVotes = await Promise.all(normalizedRewards.map(item => tx.flowGRPOVote.create({
        data: {
          job_id: params.jobID,
          vote_task_id: params.taskID,
          candidate_id: item.candidate_id,
          value: item.value,
          reward: item.reward,
        },
      })));

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
