import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import prisma from '@/server/prisma';
import { getModelsPath, getTrainingFolder } from '@/server/settings';

export const dynamic = 'force-dynamic';

interface LoraFile {
  name: string;
  path: string;
  size: number;
  mtime: number;
}

const listSafetensors = async (dir: string): Promise<LoraFile[]> => {
  try {
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    const files = entries.filter(e => e.isFile() && e.name.endsWith('.safetensors'));
    return (
      await Promise.all(
        files.map(async e => {
          const p = path.join(dir, e.name);
          const st = await fs.promises.stat(p);
          return { name: e.name, path: p, size: st.size, mtime: st.mtimeMs };
        }),
      )
    ).sort((a, b) => b.mtime - a.mtime);
  } catch {
    return [];
  }
};

const walk = async (root: string, rel = '', depth = 0, out: (LoraFile & { relpath: string })[] = []) => {
  if (depth > 5) return out;
  let entries: fs.Dirent[];
  try {
    entries = await fs.promises.readdir(path.join(root, rel), { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    if (e.name.startsWith('.')) continue;
    const r = rel ? path.join(rel, e.name) : e.name;
    if (e.isDirectory()) {
      await walk(root, r, depth + 1, out);
    } else if (e.isFile() && e.name.endsWith('.safetensors')) {
      const p = path.join(root, r);
      const st = await fs.promises.stat(p);
      out.push({ name: e.name, relpath: r, path: p, size: st.size, mtime: st.mtimeMs });
    }
  }
  return out;
};

// LoRA files available to the Generate page: every training job's saved
// checkpoints, and everything under <MODELS_PATH>/loras.
export async function GET() {
  const trainingFolder = await getTrainingFolder();
  const jobs = await prisma.job.findMany({ where: { job_type: 'train' }, orderBy: { updated_at: 'desc' } });
  const jobEntries = (
    await Promise.all(
      jobs.map(async job => ({
        id: job.id,
        name: job.name,
        status: job.status,
        updated_at: job.updated_at,
        files: await listSafetensors(path.join(trainingFolder, job.name)),
      })),
    )
  ).filter(j => j.files.length > 0);

  const modelsPath = await getModelsPath();
  const lorasRoot = path.join(modelsPath, 'loras');
  const models = (await walk(lorasRoot)).sort((a, b) => a.relpath.localeCompare(b.relpath));

  return NextResponse.json({ jobs: jobEntries, models, lorasRoot });
}
