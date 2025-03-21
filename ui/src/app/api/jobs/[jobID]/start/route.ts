import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { TOOLKIT_ROOT } from '@/paths';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { getTrainingFolder, getHFToken } from '@/server/settings';
const isWindows = process.platform === 'win32';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // update job status to 'running'
  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      info: 'Starting job...',
    },
  });

  // setup the training
  const trainingRoot = await getTrainingFolder();

  const trainingFolder = path.join(trainingRoot, job.name);
  if (!fs.existsSync(trainingFolder)) {
    fs.mkdirSync(trainingFolder, { recursive: true });
  }

  // make the config file
  const configPath = path.join(trainingFolder, '.job_config.json');

  //log to path
  const logPath = path.join(trainingFolder, 'log.txt');

  try {
    // if the log path exists, move it to a folder called logs and rename it {num}_log.txt, looking for the highest num
    // if the log path does not exist, create it
    if (fs.existsSync(logPath)) {
      const logsFolder = path.join(trainingFolder, 'logs');
      if (!fs.existsSync(logsFolder)) {
        fs.mkdirSync(logsFolder, { recursive: true });
      }

      let num = 0;
      while (fs.existsSync(path.join(logsFolder, `${num}_log.txt`))) {
        num++;
      }

      fs.renameSync(logPath, path.join(logsFolder, `${num}_log.txt`));
    }
  } catch (e) {
    console.error('Error moving log file:', e);
  }

  // update the config dataset path
  const jobConfig = JSON.parse(job.job_config);
  jobConfig.config.process[0].sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');

  // write the config file
  fs.writeFileSync(configPath, JSON.stringify(jobConfig, null, 2));

  let pythonPath = 'python';
  // use .venv or venv if it exists
  if (fs.existsSync(path.join(TOOLKIT_ROOT, '.venv'))) {
    if (isWindows) {
      pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe');
    } else {
      pythonPath = path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python');
    }
  } else if (fs.existsSync(path.join(TOOLKIT_ROOT, 'venv'))) {
    if (isWindows) {
      pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe');
    } else {
      pythonPath = path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python');
    }
  }

  const runFilePath = path.join(TOOLKIT_ROOT, 'run.py');
  if (!fs.existsSync(runFilePath)) {
    return NextResponse.json({ error: 'run.py not found' }, { status: 500 });
  }

  const additionalEnv: any = {
    AITK_JOB_ID: jobID,
    CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
    IS_AI_TOOLKIT_UI: '1'
  };

  // HF_TOKEN
  const hfToken = await getHFToken();
  if (hfToken && hfToken.trim() !== '') {
    additionalEnv.HF_TOKEN = hfToken;
  }

  // Add the --log argument to the command
  const args = [runFilePath, configPath, '--log', logPath];

  try {
    let subprocess;

    if (isWindows) {
      // For Windows, use 'cmd.exe' to open a new command window
      subprocess = spawn('cmd.exe', ['/c', 'start', 'cmd.exe', '/k', pythonPath, ...args], {
        env: {
          ...process.env,
          ...additionalEnv,
        },
        cwd: TOOLKIT_ROOT,
        windowsHide: false,
      });
    } else {
      // For non-Windows platforms
      subprocess = spawn(pythonPath, args, {
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe'], // Changed from 'ignore' to capture output
        env: {
          ...process.env,
          ...additionalEnv,
        },
        cwd: TOOLKIT_ROOT,
      });
    }

    // Start monitoring in the background without blocking the response
    const monitorProcess = async () => {
      const startTime = Date.now();
      let errorOutput = '';
      let stdoutput = '';

      if (subprocess.stderr) {
        subprocess.stderr.on('data', data => {
          errorOutput += data.toString();
        });
        subprocess.stdout.on('data', data => {
          stdoutput += data.toString();
          // truncate to only get the last 500 characters
          if (stdoutput.length > 500) {
            stdoutput = stdoutput.substring(stdoutput.length - 500);
          }
        });
      }

      subprocess.on('exit', async code => {
        const currentTime = Date.now();
        const duration = (currentTime - startTime) / 1000;
        console.log(`Job ${jobID} exited with code ${code} after ${duration} seconds.`);
        // wait for 5 seconds to give it time to stop itself. It id still has a status of running in the db, update it to stopped
        await new Promise(resolve => setTimeout(resolve, 5000));
        const updatedJob = await prisma.job.findUnique({
          where: { id: jobID },
        });
        if (updatedJob?.status === 'running') {
          let errorString = errorOutput;
          if (errorString.trim() === '') {
            errorString = stdoutput;
          }
          await prisma.job.update({
            where: { id: jobID },
            data: {
              status: 'error',
              info: `Error launching job: ${errorString.substring(0, 500)}`,
            },
          });
        }
      });

      // Wait 30 seconds before releasing the process
      await new Promise(resolve => setTimeout(resolve, 30000));
      // Detach the process for non-Windows systems
      if (!isWindows && subprocess.unref) {
        subprocess.unref();
      }
    };

    // Start the monitoring without awaiting it
    monitorProcess().catch(err => {
      console.error(`Error in process monitoring for job ${jobID}:`, err);
    });

    // Return the response immediately
    return NextResponse.json(job);
  } catch (error: any) {
    // Handle any exceptions during process launch
    console.error('Error launching process:', error);

    await prisma.job.update({
      where: { id: jobID },
      data: {
        status: 'error',
        info: `Error launching job: ${error?.message || 'Unknown error'}`,
      },
    });

    return NextResponse.json(
      {
        error: 'Failed to launch job process',
        details: error?.message || 'Unknown error',
      },
      { status: 500 },
    );
  }
}
