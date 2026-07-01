import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken } from '../paths';
import { resolvePythonPath } from '../pythonPath';
const isWindows = process.platform === 'win32';

const appendJobLog = (logPath: string, message: string) => {
  fs.appendFile(logPath, message, error => {
    if (error) console.error('Error writing to job log:', error);
  });
};

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

    const pythonPath = resolvePythonPath();

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
      return;
    }

    const additionalEnv: any = {
      AITK_JOB_ID: jobID,
      CUDA_DEVICE_ORDER: 'PCI_BUS_ID',
      CUDA_VISIBLE_DEVICES: `${job.gpu_ids}`,
      IS_AI_TOOLKIT_UI: '1',
      PYTHONUNBUFFERED: '1', // write Python output immediately so it is not lost on a crash
    };

    // HF_TOKEN
    const hfToken = await getHFToken();
    if (hfToken && hfToken.trim() !== '') {
      additionalEnv.HF_TOKEN = hfToken;
    }

    const args = [runFilePath, configPath];

    let logFd: number | null = null;
    try {
      // Capture errors that occur before run.py can initialize file logging.
      logFd = fs.openSync(logPath, 'a');
      let subprocess;

      if (isWindows) {
        // Spawn Python directly on Windows so the process can survive parent exit
        subprocess = spawn(pythonPath, args, {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
          detached: true,
          windowsHide: true,
          stdio: ['ignore', logFd, logFd], // don't tie stdio to parent; log fd passed as stdout and stderr
        });
      } else {
        // For non-Windows platforms, fully detach and ignore stdio so it survives daemon-like
        subprocess = spawn(pythonPath, args, {
          detached: true,
          stdio: ['ignore', logFd, logFd], // don't tie stdio to parent; log fd passed as stdout and stderr
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
        });
      }

      // Handle failures where the child process could not be started.
      subprocess.once('error', error => {
        const message = `Error launching job process: ${error.message}`;
        console.error(message);
        appendJobLog(logPath, `${message}\n`);
        void prisma.job
          .update({
            where: { id: jobID },
            data: { status: 'error', info: message, pid: null },
          })
          .catch(updateError => {
            console.error('Error updating job after process launch failure:', updateError);
          });
      });

      // Record abnormal termination and repair jobs Python could not update itself.
      subprocess.once('exit', (code, signal) => {
        if (code === 0) return;

        const result = signal ? `signal ${signal}` : `exit code ${code}`;
        const message = `Job process terminated with ${result}.`;
        appendJobLog(logPath, `\n${message}\n`);
        void prisma.job
          .updateMany({
            where: { id: jobID, status: 'running' },
            data: { status: 'error', info: message, pid: null },
          })
          .catch(updateError => {
            console.error('Error updating job after abnormal process exit:', updateError);
          });
      });

      // Save the PID to the database and a file for future management (stop/inspect)
      const pid = subprocess.pid ?? null;
      if (pid != null) {
        await prisma.job.update({
          where: { id: jobID },
          data: { pid },
        });
      }
      try {
        fs.writeFileSync(path.join(trainingFolder, 'pid.txt'), String(pid ?? ''), { flag: 'w' });
      } catch (e) {
        console.error('Error writing pid file:', e);
      }

      // Important: let the child run independently of this Node process.
      if (subprocess.unref) {
        subprocess.unref();
      }

      // The child remains independent; these listeners only record failures
      // while the worker is alive.
    } catch (error: any) {
      // Handle any exceptions during process launch
      console.error('Error launching process:', error);
      appendJobLog(logPath, `Error launching job process: ${error?.message || 'Unknown error'}\n`);

      await prisma.job.update({
        where: { id: jobID },
        data: {
          status: 'error',
          info: `Error launching job: ${error?.message || 'Unknown error'}`,
        },
      });
      return;
    } finally {
      if (logFd !== null) {
        fs.closeSync(logFd);
      }
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
