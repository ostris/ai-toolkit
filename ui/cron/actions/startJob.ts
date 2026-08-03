import prisma from '../prisma';
import { Job } from '@prisma/client';
import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT, getTrainingFolder, getHFToken, getModelsPath } from '../paths';
import { resolveDetachedPythonPath } from '../pythonPath';
const isWindows = process.platform === 'win32';

const appendJobLog = (logPath: string, message: string) => {
  fs.appendFile(logPath, message, error => {
    if (error) console.error('Error writing to job log:', error);
  });
};

// Windows only. Launched as `node -e <this>` so the job ends up outside the
// worker's process tree: `taskkill /T` (the dev script's `concurrently -k`,
// or any shutdown that kills the tree) walks parent/child links and would take
// a direct child down with the UI. This relay exits immediately, orphaning the
// job, and `detached` keeps the job alive once its parent is gone. Its own
// stdout/stderr are the job log, so the job inherits them as fds 1 and 2.
// Python failing to launch at all (broken venv, missing interpreter) happens
// inside the relay, so the relay -- not the worker -- is what sees that error.
// It reports it two ways: on stderr, which is the job log, and through the pid
// file, so the worker can put the real reason in the database.
const RELAY_ERROR_PREFIX = 'error:';
const WINDOWS_RELAY_SCRIPT = `
const { spawn } = require('child_process');
const fs = require('fs');
const [pidFile, command, ...args] = process.argv.slice(1);
const child = spawn(command, args, {
  detached: true,
  windowsHide: true,
  stdio: ['ignore', 1, 2],
});
child.once('error', error => {
  process.stderr.write('Error launching job process: ' + error.message + '\\n');
  try {
    fs.writeFileSync(pidFile, '${RELAY_ERROR_PREFIX}' + error.message);
  } catch (e) {
    process.stderr.write('Could not write job pid file: ' + e.message + '\\n');
  }
  process.exit(1);
});
if (child.pid) {
  fs.writeFileSync(pidFile, String(child.pid));
  child.unref();
}
`;

const RELAY_PID_TIMEOUT_MS = 30000;

type RelayResult = { pid: number | null; error?: string };

// The relay exits as soon as it has launched the job, leaving the real pid in
// pidPath. Without this we would only ever know the (already dead) relay's pid.
const readRelayPid = (relay: ChildProcess, pidPath: string): Promise<RelayResult> => {
  return new Promise(resolve => {
    let settled = false;
    const finish = (value: RelayResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };

    const timer = setTimeout(
      () => finish({ pid: null, error: 'Timed out waiting for the job process to start' }),
      RELAY_PID_TIMEOUT_MS,
    );

    relay.once('exit', () => {
      let contents: string;
      try {
        contents = fs.readFileSync(pidPath, 'utf8').trim();
      } catch {
        finish({ pid: null, error: 'Job process did not report a pid' });
        return;
      }

      if (contents.startsWith(RELAY_ERROR_PREFIX)) {
        finish({ pid: null, error: contents.slice(RELAY_ERROR_PREFIX.length) });
        return;
      }

      const pid = Number(contents);
      finish(
        Number.isInteger(pid) && pid > 0
          ? { pid }
          : { pid: null, error: 'Job process did not report a usable pid' },
      );
    });

    relay.once('error', error => finish({ pid: null, error: error.message }));
  });
};

const isProcessAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    // EPERM means it exists but belongs to someone else, which still counts.
    return e?.code === 'EPERM';
  }
};

// We cannot read an exit code off a process that is not our child, so pull the
// last thing it said instead -- for a job that dies on startup (bad venv,
// missing CUDA libs) that traceback line is the whole diagnosis.
const LOG_TAIL_BYTES = 4096;
const LOG_TAIL_MAX_CHARS = 300;

const readLogTail = (logPath: string): string | null => {
  let fd: number | null = null;
  try {
    const size = fs.statSync(logPath).size;
    const length = Math.min(size, LOG_TAIL_BYTES);
    if (length === 0) return null;

    const buffer = Buffer.alloc(length);
    fd = fs.openSync(logPath, 'r');
    fs.readSync(fd, buffer, 0, length, size - length);

    const lines = buffer.toString('utf8').split(/\r?\n/).filter(line => line.trim() !== '');
    const lastLine = lines[lines.length - 1];
    if (!lastLine) return null;
    return lastLine.length > LOG_TAIL_MAX_CHARS ? `${lastLine.slice(-LOG_TAIL_MAX_CHARS)}` : lastLine;
  } catch {
    return null;
  } finally {
    if (fd !== null) {
      try {
        fs.closeSync(fd);
      } catch {
        // nothing useful to do if the log handle will not close
      }
    }
  }
};

// The job is not our child anymore, so there is no 'exit' event to listen for.
// Poll instead, so a job that dies without updating its own row (OOM kill,
// hard crash) still gets marked as an error rather than sitting on 'running'.
const JOB_POLL_INTERVAL_MS = 2000;

const watchDetachedJob = (pid: number, jobID: string, logPath: string) => {
  const timer = setInterval(() => {
    if (isProcessAlive(pid)) return;
    clearInterval(timer);

    // A stopped or completed job writes its own ending (status row + final
    // log lines) via its KeyboardInterrupt/done handlers -- stay out of the
    // way. Only a job that vanished while still marked 'running' died without
    // getting to say anything; record that. There is no exit code to read off
    // a process that is not our child, so report the last thing it logged.
    const tail = readLogTail(logPath);
    const message = tail
      ? `Job process exited unexpectedly. Last log line: ${tail}`
      : 'Job process exited unexpectedly.';
    void prisma.job
      .updateMany({
        where: { id: jobID, status: 'running' },
        data: { status: 'error', info: message, pid: null },
      })
      .then(result => {
        if (result.count > 0) appendJobLog(logPath, `\n${message}\n`);
      })
      .catch(updateError => {
        console.error('Error updating job after process disappeared:', updateError);
      });
  }, JOB_POLL_INTERVAL_MS);

  // Never hold the worker open on account of this poll.
  if (timer.unref) timer.unref();
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

    const pythonPath = resolveDetachedPythonPath();

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

    // MODELS_PATH - one set in the env always takes precedence (it passes
    // through via process.env); only fall back to the setting if it is not set
    if (!process.env.MODELS_PATH || process.env.MODELS_PATH.trim() === '') {
      const modelsPath = await getModelsPath();
      if (modelsPath && modelsPath.trim() !== '') {
        additionalEnv.MODELS_PATH = modelsPath;
      }
    }

    const args = [runFilePath, configPath];

    // Where the Windows relay reports the job's real pid back to us.
    const relayPidPath = path.join(trainingFolder, '.job_pid');

    let logFd: number | null = null;
    try {
      // Capture errors that occur before run.py can initialize file logging.
      logFd = fs.openSync(logPath, 'a');
      let subprocess;

      if (isWindows) {
        // Launch through the relay (see WINDOWS_RELAY_SCRIPT) so the job is not
        // a descendant of this worker and survives the UI being shut down or
        // tree-killed. The relay spawns the job `detached`, which is what keeps
        // it alive once the relay exits; that in turn means DETACHED_PROCESS,
        // so pythonPath is pythonw.exe to avoid Windows handing the job a
        // console window of its own.
        try {
          fs.unlinkSync(relayPidPath);
        } catch {
          // no stale pid file to clear
        }
        subprocess = spawn(process.execPath, ['-e', WINDOWS_RELAY_SCRIPT, relayPidPath, pythonPath, ...args], {
          env: {
            ...process.env,
            ...additionalEnv,
          },
          cwd: TOOLKIT_ROOT,
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

      let pid: number | null;

      if (isWindows) {
        // The relay is gone within a few hundred ms; the pid it leaves behind is
        // the job's. Poll that pid for liveness since we get no 'exit' event.
        const relayResult = await readRelayPid(subprocess, relayPidPath);
        if (relayResult.pid == null) {
          throw new Error(relayResult.error ?? 'Job process did not report a pid');
        }
        pid = relayResult.pid;
        watchDetachedJob(pid, jobID, logPath);
      } else {
        pid = subprocess.pid ?? null;

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
      }

      // Save the PID to the database and a file for future management (stop/inspect)
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
      return_to_queue: false,
      info: 'Starting job...',
    },
  });
  // start and watch the job asynchronously so the cron can continue
  startAndWatchJob(job);
}
