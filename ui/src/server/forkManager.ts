import prisma from '@/server/prisma';

interface ForkJobInput {
  parentJobId: string;
  newJobName: string;
  gpuIds?: string;
}

interface ForkJobResult {
  success: boolean;
  jobId?: string;
  error?: string;
  message?: string;
}

export async function createForkedJob(input: ForkJobInput): Promise<ForkJobResult> {
  try {
    const { parentJobId, newJobName, gpuIds } = input;

    // Get parent job
    const parentJob = await prisma.job.findUnique({
      where: { id: parentJobId },
    });

    if (!parentJob) {
      return {
        success: false,
        error: 'Parent job not found',
      };
    }

    // Check if job name already exists
    const existingJob = await prisma.job.findUnique({
      where: { name: newJobName },
    });

    if (existingJob) {
      return {
        success: false,
        error: `Job name "${newJobName}" already exists`,
      };
    }

    // Use provided GPU IDs or parent's GPU IDs
    const jobGpuIds = gpuIds || parentJob.gpu_ids;

    // Get highest queue position
    const highestQueuePosition = await prisma.job.aggregate({
      _max: {
        queue_position: true,
      },
    });
    const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

    // Parse parent job config
    let jobConfig: any;
    try {
      jobConfig = JSON.parse(parentJob.job_config);
    } catch (e) {
      return {
        success: false,
        error: 'Failed to parse parent job configuration',
      };
    }

    // Update the training name in the config to match the new job name
    if (jobConfig.config && typeof jobConfig.config === 'object') {
      jobConfig.config.name = newJobName;
    }

    // Create new forked job
    const forkedJob = await prisma.job.create({
      data: {
        name: newJobName,
        gpu_ids: jobGpuIds,
        job_config: JSON.stringify(jobConfig),
        job_type: parentJob.job_type,
        status: 'stopped',
        queue_position: newQueuePosition,
        parent_job_id: parentJobId,
        forked_at: new Date(),
        fork_from_step: parentJob.step,
      },
    });

    return {
      success: true,
      jobId: forkedJob.id,
      message: `Job forked successfully from step ${parentJob.step}`,
    };
  } catch (error) {
    console.error('Error creating forked job:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    };
  }
}

export async function getJobWithForkInfo(jobId: string) {
  try {
    const job = await prisma.job.findUnique({
      where: { id: jobId },
    });

    if (!job) {
      return null;
    }

    // Get fork info if this is a forked job
    let parentJob = null;
    if (job.parent_job_id) {
      parentJob = await prisma.job.findUnique({
        where: { id: job.parent_job_id },
      });
    }

    // Get child jobs (jobs forked from this one)
    const childJobs = await prisma.job.findMany({
      where: { parent_job_id: jobId },
      orderBy: { created_at: 'desc' },
    });

    return {
      job,
      parentJob,
      childJobs,
    };
  } catch (error) {
    console.error('Error getting job with fork info:', error);
    return null;
  }
}

export async function getForkTree(jobId: string) {
  try {
    const result = await getJobWithForkInfo(jobId);

    if (!result) {
      return null;
    }

    // Walk up the tree to find the root
    let current = result.parentJob;
    const ancestors: any[] = [];

    while (current) {
      ancestors.unshift(current);
      const parent = await prisma.job.findUnique({
        where: { id: current.parent_job_id || '' },
      });
      current = parent;
    }

    return {
      rootJob: ancestors.length > 0 ? ancestors[0] : result.job,
      ancestors,
      currentJob: result.job,
      children: result.childJobs,
    };
  } catch (error) {
    console.error('Error getting fork tree:', error);
    return null;
  }
}
