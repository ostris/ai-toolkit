import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn, execSync } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken } from '../paths';
const isWindows = process.platform === 'win32';

/**
 * Find a free port by probing with Python's socket module.
 * Falls back to a hash-based port if the probe fails.
 */
function findFreePort(pythonPath: string, fallbackSeed: string): number {
  try {
    const port = execSync(
      `"${pythonPath.replace(/"/g, '\\"')}" -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"`,
      { timeout: 5000, encoding: 'utf-8' },
    ).trim();
    const parsed = parseInt(port, 10);
    if (isNaN(parsed)) throw new Error(`Invalid port: ${port}`);
    return parsed;
  } catch {
    // Fallback: hash-based port in range 29500-39999
    let hash = 0;
    for (let i = 0; i < fallbackSeed.length; i++) {
      hash = ((hash << 5) - hash + fallbackSeed.charCodeAt(i)) | 0;
    }
    return 29500 + (Math.abs(hash) % 10500);
  }
}

/**
 * Find the accelerate binary in the venv.
 */
function findAcceleratePath(): string | null {
  const venvDirs = ['.venv', 'venv'];
  for (const venv of venvDirs) {
    const venvPath = path.join(TOOLKIT_ROOT, venv);
    if (fs.existsSync(venvPath)) {
      const accelPath = isWindows
        ? path.join(venvPath, 'Scripts', 'accelerate.exe')
        : path.join(venvPath, 'bin', 'accelerate');
      if (fs.existsSync(accelPath)) {
        return accelPath;
      }
    }
  }
  return null;
}

const startAndWatchJob = (job: Job) => {
  // starts and watches the job asynchronously
  return new Promise<void>(async (resolve, reject) => {
    const jobID = job.id;

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
      console.error(`run.py not found at path: ${runFilePath}`);
      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: run.py not found`,
        },
      });
      resolve();
      return;
    }

    // Determine if this is a multi-GPU distributed job
    const gpuIdList = job.gpu_ids
      .split(',')
      .map(s => s.trim())
      .filter(s => s.length > 0);
    const isMultiGPU = gpuIdList.length > 1;

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      IS_AI_TOOLKIT_UI: '1',
      HF_HOME: process.env.HF_HOME || path.join(process.env.HOME || '/root', '.cache', 'huggingface'),
      HF_HUB_ENABLE_HF_TRANSFER: '0',
      HF_HUB_DISABLE_XET: '1',
      HF_HUB_DOWNLOAD_TIMEOUT: '300',
    };

    // For multi-GPU on Linux, accelerate launch --gpu_ids handles device assignment.
    // Setting CUDA_VISIBLE_DEVICES alongside --gpu_ids causes conflicts (Accelerate #1848).
    // On Windows, accelerate launch is not supported, so always set CUDA_VISIBLE_DEVICES.
    if (!isMultiGPU || isWindows) {
      additionalEnv.CUDA_VISIBLE_DEVICES = `${job.gpu_ids}`;
    }

    // HF_TOKEN
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    try {
      let childProcess;

      if (isMultiGPU && !isWindows) {
        // Multi-GPU distributed training via accelerate launch
        const acceleratePath = findAcceleratePath();
        if (!acceleratePath) {
          console.error('accelerate binary not found in venv');
          await prisma.job.update({
            where: { id: jobID },
            data: {
              status: 'error',
              info: 'Error launching distributed job: accelerate binary not found in venv',
            },
          });
          resolve();
          return;
        }

        const masterPort = findFreePort(pythonPath, jobID);
        const numProcesses = gpuIdList.length;

        const launchArgs = [
          'launch',
          `--num_processes=${numProcesses}`,
          `--gpu_ids=${gpuIdList.join(',')}`,
          `--main_process_port=${masterPort}`,
          '--mixed_precision=no', // precision handled by toolkit config
          runFilePath,
          configPath,
          '--log',
          logPath,
        ];

        console.log(`Distributed launch: ${acceleratePath} ${launchArgs.join(' ')}`);
        console.log(`  GPUs: ${gpuIdList.join(',')}, port: ${masterPort}`);

        childProcess = spawn(acceleratePath, launchArgs, {
          detached: true,
          stdio: 'ignore',
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      } else if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        const args = [runFilePath, configPath, '--log', logPath];
        childProcess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
          detached: true,
          windowsHide: true,
          stdio: 'ignore', // don't tie stdio to parent
        });
      } else {
        // Single-GPU: existing path, spawn python directly
        const args = [runFilePath, configPath, '--log', logPath];
        childProcess = spawn(pythonPath, args, {
          detached: true,
          stdio: 'ignore',
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      }

      // Important: let the child run independently of this Node process.
      if (childProcess.unref) {
        childProcess.unref();
      }

      // Write pid to database and file for future management (stop/inspect).
      // For distributed jobs, this is the launcher PID (process group leader).
      const pid = childProcess.pid ?? null;
      if (pid != null) {
        try {
          await prisma.job.update({
            where: { id: jobID },
            data: { pid },
          });
        } catch (e) {
          console.error('Error updating pid in database:', e);
        }
      }
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // (No stdout/stderr listeners — logging should go to --log handled by your Python)
      // (No monitoring loop — the whole point is to let it live past this worker)
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
      resolve();
      return;
    }
    // Resolve the promise immediately after starting the process
    resolve();
  });
};

export default async function startJob(jobID: string) {
  const job: Job | null = await prisma.job.findUnique({
    where: { id: jobID },
  });
  if (!job) {
    console.error(`Job with ID ${jobID} not found`);
    return;
  }
  // update job status to 'running', this will run sync so we don't start multiple jobs.
  await prisma.job.update({
    where: { id: jobID },
    data: {
      status: 'running',
      stop: false,
      info: 'Starting job...',
    },
  });
  // start and watch the job asynchronously so the cron can continue
  startAndWatchJob(job);
}
