import prisma from '../prisma';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '../paths';

const execAsync = promisify(exec);

export default async function monitorJobs() {
  // Find all jobs that should be stopping
  const stoppingJobs = await prisma.job.findMany({
    where: {
      status: { in: ['running', 'stopping'] },
      stop: true,
    },
  });

  for (const job of stoppingJobs) {
    console.log(`Job ${job.id} (${job.name}) should be stopping, checking if process is still alive...`);

    // Get training folder and check for PID file
    const trainingRoot = await getTrainingFolder();
    const trainingFolder = path.join(trainingRoot, job.name);
    const pidFile = path.join(trainingFolder, 'pid.txt');

    if (fs.existsSync(pidFile)) {
      const pid = fs.readFileSync(pidFile, 'utf-8').trim();

      if (pid) {
        try {
          // Check if process is still running
          const { stdout } = await execAsync(`ps -p ${pid} -o pid=`);
          if (stdout.trim()) {
            console.log(`Process ${pid} is still running, attempting to kill...`);

            // Try graceful kill first (SIGTERM)
            try {
              process.kill(parseInt(pid), 'SIGTERM');
              console.log(`Sent SIGTERM to process ${pid}`);

              // Give it 5 seconds to die gracefully
              await new Promise(resolve => setTimeout(resolve, 5000));

              // Check if it's still alive
              try {
                const { stdout: stillAlive } = await execAsync(`ps -p ${pid} -o pid=`);
                if (stillAlive.trim()) {
                  console.log(`Process ${pid} didn't respond to SIGTERM, sending SIGKILL...`);
                  process.kill(parseInt(pid), 'SIGKILL');
                }
              } catch {
                // Process is dead, good
              }
            } catch (error: any) {
              console.error(`Error killing process ${pid}:`, error.message);
            }
          }
        } catch {
          // Process doesn't exist, that's fine
          console.log(`Process ${pid} is not running`);
        }
      }
    }

    // Update job status to stopped
    await prisma.job.update({
      where: { id: job.id },
      data: {
        status: job.return_to_queue ? 'queued' : 'stopped',
        stop: false,
        return_to_queue: false,
        info: job.return_to_queue ? 'Returned to queue' : 'Stopped',
      },
    });
    console.log(`Job ${job.id} marked as ${job.return_to_queue ? 'queued' : 'stopped'}`);
  }
}
