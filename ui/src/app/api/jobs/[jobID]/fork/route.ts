import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { createForkedJob } from '@/server/forkManager';
import { getLatestCheckpoint, copyCheckpoint } from '@/server/checkpointManager';
import path from 'path';
import fs from 'fs';

const OUTPUT_BASE_DIR = path.join(process.cwd(), '..', 'output');

export async function POST(
  request: NextRequest,
  { params }: { params: { jobID: string } },
) {
  try {
    const { jobID } = await params;
    const body = await request.json();

    const { newJobName, gpuIds, copyCheckpoint: shouldCopyCheckpoint } = body;

    // Validate input
    if (!newJobName || typeof newJobName !== 'string') {
      return NextResponse.json(
        { success: false, error: 'newJobName is required' },
        { status: 400 },
      );
    }

    if (newJobName.trim().length === 0) {
      return NextResponse.json(
        { success: false, error: 'Job name cannot be empty' },
        { status: 400 },
      );
    }

    // Get parent job
    const parentJob = await prisma.job.findUnique({
      where: { id: jobID },
    });

    if (!parentJob) {
      return NextResponse.json(
        { success: false, error: 'Parent job not found' },
        { status: 404 },
      );
    }

    // Create forked job in database
    const forkResult = await createForkedJob({
      parentJobId: jobID,
      newJobName: newJobName.trim(),
      gpuIds: gpuIds || parentJob.gpu_ids,
    });

    if (!forkResult.success) {
      return NextResponse.json(
        { success: false, error: forkResult.error },
        { status: 400 },
      );
    }

    // Copy checkpoint to ensure fork continues from parent's training state
    if (shouldCopyCheckpoint && forkResult.jobId) {
      try {
        const parentJobOutputDir = path.join(OUTPUT_BASE_DIR, parentJob.name);
        const forkedJobOutputDir = path.join(OUTPUT_BASE_DIR, newJobName.trim());
        const forkedJobName = newJobName.trim();

        const checkpoint = await getLatestCheckpoint(parentJobOutputDir);

        if (!checkpoint || !checkpoint.exists) {
          console.warn(`[Fork] No checkpoint found in parent job: ${parentJobOutputDir}`);
        } else {
          // Create forked job output directory
          if (!fs.existsSync(forkedJobOutputDir)) {
            fs.mkdirSync(forkedJobOutputDir, { recursive: true });
          }

          // Copy checkpoint and rename to match fork job name
          await copyCheckpoint(
            checkpoint.path,
            forkedJobOutputDir,
            forkedJobName,
          );
        }
      } catch (checkpointError) {
        console.error('[Fork] Error copying checkpoint:', checkpointError);
      }
    }

    return NextResponse.json(
      {
        success: true,
        jobId: forkResult.jobId,
        message: forkResult.message,
      },
      { status: 201 },
    );
  } catch (error) {
    console.error('Error forking job:', error);
    return NextResponse.json(
      { success: false, error: 'Internal server error' },
      { status: 500 },
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: { jobID: string } },
) {
  try {
    const { jobID } = await params;

    const job = await prisma.job.findUnique({
      where: { id: jobID },
    });

    if (!job) {
      return NextResponse.json(
        { error: 'Job not found' },
        { status: 404 },
      );
    }

    // Get fork information
    let parentJob = null;
    if (job.parent_job_id) {
      parentJob = await prisma.job.findUnique({
        where: { id: job.parent_job_id },
      });
    }

    const childJobs = await prisma.job.findMany({
      where: { parent_job_id: jobID },
      orderBy: { created_at: 'desc' },
    });

    return NextResponse.json({
      job,
      parentJob,
      childJobs,
    });
  } catch (error) {
    console.error('Error getting fork info:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 },
    );
  }
}
