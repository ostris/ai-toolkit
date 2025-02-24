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
  };

  // HF_TOKEN
  const hfToken = await getHFToken();
  if (hfToken && hfToken.trim() !== '') {
    additionalEnv.HF_TOKEN = hfToken;
  }

  let cmd = `${pythonPath} ${runFilePath} ${configPath}`;
  for (const key in additionalEnv) {
    if (os.platform() === 'win32') {
      cmd = `set ${key}=${additionalEnv[key]} && ${cmd}`;
    } else {
      cmd = `${key}=${additionalEnv[key]} ${cmd}`;
    }
  }

  console.log('Spawning command:', cmd);

  // start job
  if (isWindows) {
    // For Windows, use 'cmd.exe' to open a new command window
    const subprocess = spawn('cmd.exe', ['/c', 'start', 'cmd.exe', '/k', pythonPath, runFilePath, configPath], {
      env: {
        ...process.env,
        ...additionalEnv,
      },
      cwd: TOOLKIT_ROOT,
      windowsHide: false,
    });
    
    subprocess.unref();
  } else {
    // For non-Windows platforms, use your original approach
    const subprocess = spawn(pythonPath, [runFilePath, configPath], {
      detached: true,
      stdio: 'ignore',
      env: {
        ...process.env,
        ...additionalEnv,
      },
      cwd: TOOLKIT_ROOT,
    });
    
    subprocess.unref();
  }
  // const subprocess = spawn(pythonPath, [runFilePath, configPath], {
  //   detached: true,
  //   stdio: 'ignore',
  //   env: {
  //     ...process.env,
  //     ...additionalEnv,
  //   },
  //   cwd: TOOLKIT_ROOT,
  // });

  // subprocess.unref();

  return NextResponse.json(job);
}
