import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST() {
  try {
    // Find all jobs with status 'running'
    const runningJobs = await prisma.job.findMany({
      where: { status: 'running' },
      select: { id: true, name: true }
    });

    if (runningJobs.length === 0) {
      return NextResponse.json({ 
        message: 'No running jobs found',
        count: 0 
      });
    }

    // Update all running jobs to stopped status
    const result = await prisma.job.updateMany({
      where: { status: 'running' },
      data: { 
        status: 'stopped',
        stop: true,
        info: 'Stopped via debug panel'
      }
    });

    return NextResponse.json({
      message: `Successfully stopped ${result.count} job${result.count === 1 ? '' : 's'}`,
      count: result.count,
      stoppedJobs: runningJobs.map(job => ({ id: job.id, name: job.name }))
    });

  } catch (error) {
    console.error('Error stopping all running jobs:', error);
    return NextResponse.json(
      { error: 'Failed to stop all running jobs' }, 
      { status: 500 }
    );
  }
}