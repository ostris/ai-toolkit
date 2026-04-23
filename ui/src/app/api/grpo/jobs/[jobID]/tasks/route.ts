import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET(request: Request, { params }: { params: { jobID: string } }) {
  const { searchParams } = new URL(request.url);
  const statusParam = searchParams.get('status') || 'requested,generating,open,voted';
  const requestedLimit = parseInt(searchParams.get('limit') || '10', 10);
  const limit = Number.isFinite(requestedLimit) ? Math.max(1, Math.min(requestedLimit, 20)) : 10;
  const statuses = statusParam
    .split(',')
    .map(value => value.trim())
    .filter(Boolean);

  try {
    const tasks = await prisma.flowGRPOVoteTask.findMany({
      where: {
        job_id: params.jobID,
        ...(statuses.length === 1 ? { status: statuses[0] } : { status: { in: statuses } }),
      },
      include: {
        candidates: {
          orderBy: {
            order_index: 'asc',
          },
        },
        votes: {
          orderBy: {
            created_at: 'asc',
          },
        },
      },
      orderBy: {
        created_at: 'asc',
      },
      take: limit,
    });

    return NextResponse.json({
      tasks: tasks.map(task => ({
        ...task,
        candidates: task.candidates.map(candidate => ({
          ...candidate,
          image_url: `/api/img/${encodeURIComponent(candidate.image_path)}`,
        })),
      })),
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to load Flow-GRPO vote tasks' }, { status: 500 });
  }
}

export async function POST(request: Request, { params }: { params: { jobID: string } }) {
  try {
    const body = await request.json();
    const prompt = `${body.prompt || ''}`.trim();
    const negativePrompt = `${body.negative_prompt || ''}`.trim();
    const requestedCandidates = Math.max(2, parseInt(`${body.requested_candidates || '4'}`, 10) || 4);
    const width = Math.max(64, parseInt(`${body.width || '1024'}`, 10) || 1024);
    const height = Math.max(64, parseInt(`${body.height || '1024'}`, 10) || 1024);
    const guidanceScale = Math.max(0, parseFloat(`${body.guidance_scale || '4'}`) || 4);
    const numInferenceSteps = Math.max(1, parseInt(`${body.num_inference_steps || '30'}`, 10) || 30);
    const sampler = `${body.sampler || 'flowmatch'}`.trim() || 'flowmatch';
    const scheduler = `${body.scheduler || 'flowmatch'}`.trim() || 'flowmatch';
    const seedValue = `${body.seed ?? ''}`.trim();
    const seed = seedValue === '' ? null : parseInt(seedValue, 10);

    if (!prompt) {
      return NextResponse.json({ error: 'Prompt is required' }, { status: 400 });
    }

    const job = await prisma.job.findUnique({
      where: { id: params.jobID },
    });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const task = await prisma.flowGRPOVoteTask.create({
      data: {
        job_id: params.jobID,
        prompt,
        negative_prompt: negativePrompt,
        requested_candidates: requestedCandidates,
        width,
        height,
        seed: Number.isNaN(seed as number) ? null : seed,
        guidance_scale: guidanceScale,
        num_inference_steps: numInferenceSteps,
        sampler,
        scheduler,
        status: 'requested',
      },
    });

    return NextResponse.json({ ok: true, task });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to create Flow-GRPO vote task' }, { status: 500 });
  }
}
