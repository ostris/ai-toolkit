import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';

const prisma = new PrismaClient();

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const body = await request.json();
  const { mode, newSteps, newName } = body;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (mode === 'resume') {
    // Mode 1: Resume training - same job, increase steps, change status to stopped
    // DO NOT set pretrained_lora_path - let Python code auto-detect checkpoint
    // This ensures metadata (step count) is loaded correctly
    const jobConfig = JSON.parse(job.job_config);

    // Update steps if provided
    if (newSteps && newSteps > job.step) {
      jobConfig.config.process[0].train.steps = newSteps;
    }

    // Remove any pretrained_lora_path that might exist from previous clone operations
    if (jobConfig.config.process[0].network?.pretrained_lora_path) {
      delete jobConfig.config.process[0].network.pretrained_lora_path;
    }

    // Update job to allow resumption
    const updatedJob = await prisma.job.update({
      where: { id: jobID },
      data: {
        status: 'stopped',
        stop: false,
        info: 'Ready to resume - will auto-detect latest checkpoint',
        job_config: JSON.stringify(jobConfig),
      },
    });

    console.log(`Job ${jobID} ready to resume with ${newSteps} steps`);
    return NextResponse.json(updatedJob);

  } else if (mode === 'clone') {
    // Mode 2: Clone with new name, using final checkpoint as pretrained_lora_path
    const jobConfig = JSON.parse(job.job_config);
    const oldName = jobConfig.config.name;
    const finalName = newName || `${oldName}_continued`;

    // Update job name
    jobConfig.config.name = finalName;

    // Update steps if provided
    if (newSteps) {
      jobConfig.config.process[0].train.steps = newSteps;
    }

    // Find the latest checkpoint from the old job
    const trainingFolder = jobConfig.config.process[0].training_folder;
    const oldJobFolder = path.join(trainingFolder, oldName);

    let latestCheckpoint = null;
    if (fs.existsSync(oldJobFolder)) {
      const files = fs.readdirSync(oldJobFolder);
      const checkpoints = files.filter(f =>
        f.startsWith(oldName) &&
        (f.endsWith('.safetensors') || f.endsWith('.pt'))
      );

      if (checkpoints.length > 0) {
        // Smart sorting: Find the best checkpoint
        // Priority: 1) Final file without step, 2) Highest step number, 3) Most recent
        checkpoints.sort((a, b) => {
          // Extract step number from filename (e.g., "lora_1_4000.safetensors" -> 4000)
          const stepRegex = /_(\d+)\.(safetensors|pt)$/;
          const aMatch = a.match(stepRegex);
          const bMatch = b.match(stepRegex);

          const aHasStep = !!aMatch;
          const bHasStep = !!bMatch;

          // If neither has step (both are final files like "lora_1.safetensors"), use modification time
          if (!aHasStep && !bHasStep) {
            const aPath = path.join(oldJobFolder, a);
            const bPath = path.join(oldJobFolder, b);
            return fs.statSync(bPath).mtime.getTime() - fs.statSync(aPath).mtime.getTime();
          }

          // Prefer files WITHOUT step numbers (final files) over checkpoints
          if (!aHasStep && bHasStep) return -1;  // a is final, prefer it
          if (aHasStep && !bHasStep) return 1;   // b is final, prefer it

          // Both have step numbers, use highest step
          const aStep = parseInt(aMatch![1]);
          const bStep = parseInt(bMatch![1]);
          return bStep - aStep;
        });
        latestCheckpoint = path.join(oldJobFolder, checkpoints[0]);
      }
    }

    // Set pretrained_lora_path to the latest checkpoint
    if (latestCheckpoint) {
      if (!jobConfig.config.process[0].network) {
        jobConfig.config.process[0].network = {};
      }
      jobConfig.config.process[0].network.pretrained_lora_path = latestCheckpoint;
    }

    // Create new job
    const newJob = await prisma.job.create({
      data: {
        name: finalName,
        gpu_ids: job.gpu_ids,
        job_config: JSON.stringify(jobConfig),
        status: 'stopped',
        stop: false,
        step: 0,
        info: latestCheckpoint
          ? `Starting from checkpoint: ${path.basename(latestCheckpoint)}`
          : 'Starting fresh',
        queue_position: 0,
      },
    });

    console.log(`Cloned job ${jobID} as ${newJob.id} with name ${finalName}`);
    return NextResponse.json(newJob);

  } else {
    return NextResponse.json({ error: 'Invalid mode' }, { status: 400 });
  }
}
