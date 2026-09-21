import fs from 'fs';
import path from 'path';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';

// What the engine job writes to <job folder>/engine.json when its server is up.
export interface EngineEndpoint {
  host: string;
  port: number;
  token: string;
  pid: number;
  job_id?: string | null;
  device?: string;
  job_folder: string;
  output_folder: string;
  started_at: number;
}

export interface EngineHandle {
  jobId: string;
  jobName: string;
  gpuIds: string;
  status: string;
  endpoint: EngineEndpoint | null;
}

const isProcessAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (e: any) {
    return e?.code === 'EPERM';
  }
};

const readEndpoint = (jobFolder: string): EngineEndpoint | null => {
  try {
    const raw = fs.readFileSync(path.join(jobFolder, 'engine.json'), 'utf8');
    const ep = JSON.parse(raw) as EngineEndpoint;
    if (!ep.host || !ep.port) return null;
    // a stale file from a crashed engine must not be proxied to
    if (ep.pid && !isProcessAlive(ep.pid)) return null;
    return ep;
  } catch {
    return null;
  }
};

let cache: { ts: number; handles: EngineHandle[] } | null = null;
const CACHE_MS = 2000;

/** Every inference-engine job that is running (or starting), with its endpoint when published. */
export async function listEngines(forceFresh = false): Promise<EngineHandle[]> {
  if (!forceFresh && cache && Date.now() - cache.ts < CACHE_MS) return cache.handles;
  // every inference job: a row already marked stopped can still be a live
  // process finishing (or ignoring) its shutdown, which engine.json's pid tells us
  const jobs = await prisma.job.findMany({ where: { job_type: 'inference' }, orderBy: { updated_at: 'desc' } });
  const trainingFolder = await getTrainingFolder();
  const handles: EngineHandle[] = [];
  for (const job of jobs) {
    const active = ['running', 'queued', 'stopping'].includes(job.status);
    const endpoint = readEndpoint(path.join(trainingFolder, job.name));
    if (!active && !endpoint) continue;
    handles.push({
      jobId: job.id,
      jobName: job.name,
      gpuIds: job.gpu_ids,
      // process still alive after the row was stopped: report it as stopping
      status: !active && endpoint ? 'stopping' : job.status,
      endpoint,
    });
  }
  cache = { ts: Date.now(), handles };
  return handles;
}

/** The engine to talk to: the requested job, else the first running one with an endpoint. */
export async function resolveEngine(jobId?: string | null): Promise<EngineHandle | null> {
  const handles = await listEngines();
  if (jobId) return handles.find(h => h.jobId === jobId) || null;
  return handles.find(h => h.endpoint) || handles[0] || null;
}
