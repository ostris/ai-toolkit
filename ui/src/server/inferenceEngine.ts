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
  const jobs = await prisma.job.findMany({
    where: { job_type: 'inference', status: { in: ['running', 'queued', 'stopping'] } },
    orderBy: { updated_at: 'desc' },
  });
  const trainingFolder = await getTrainingFolder();
  const handles: EngineHandle[] = jobs.map(job => ({
    jobId: job.id,
    jobName: job.name,
    gpuIds: job.gpu_ids,
    status: job.status,
    endpoint: job.status === 'running' ? readEndpoint(path.join(trainingFolder, job.name)) : null,
  }));
  cache = { ts: Date.now(), handles };
  return handles;
}

/** The engine to talk to: the requested job, else the first running one with an endpoint. */
export async function resolveEngine(jobId?: string | null): Promise<EngineHandle | null> {
  const handles = await listEngines();
  if (jobId) return handles.find(h => h.jobId === jobId) || null;
  return handles.find(h => h.endpoint) || handles[0] || null;
}
