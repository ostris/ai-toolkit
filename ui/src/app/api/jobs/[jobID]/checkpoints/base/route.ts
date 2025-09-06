import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';

const prisma = new PrismaClient();

export async function DELETE(request: NextRequest, { params }: { params: { jobID: string } }) {
  try {
    const { jobID } = params;
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

    const status = (job.status || '').toLowerCase();
    if (status === 'running' || status === 'stopping') {
      return NextResponse.json({ error: 'Cannot remove base checkpoint while training is running.' }, { status: 409 });
    }

    const trainingRoot = await getTrainingFolder();
    const resolvedTrainingRoot = path.resolve(trainingRoot);
    const jobFolder = path.resolve(resolvedTrainingRoot, job.name);
    if (!(jobFolder + path.sep).startsWith(resolvedTrainingRoot + path.sep)) {
      return NextResponse.json({ error: 'Invalid job folder' }, { status: 400 });
    }
    const markerPath = path.join(jobFolder, '.base_checkpoint');
    if (!fs.existsSync(markerPath)) {
      return NextResponse.json({ ok: true, message: 'No base checkpoint set' });
    }

    let baseName: string | null = null;
    try {
      baseName = path.basename(fs.readFileSync(markerPath, { encoding: 'utf-8' }).trim());
    } catch {}

    // Remove marker
    try {
      fs.unlinkSync(markerPath);
    } catch (e) {
      // ignore
    }

    // Delete the actual checkpoint file as well so it no longer appears in the overview
    if (baseName) {
      const basePath = path.resolve(jobFolder, baseName);
      if (basePath.startsWith(jobFolder + path.sep) && fs.existsSync(basePath)) {
        try {
          fs.unlinkSync(basePath);
        } catch (e) {
          // ignore file deletion errors, still proceed
        }
      }
    }

    // Optionally clear parts of job_config indicating network defaults inferred from base.
    try {
      if (job.job_config) {
        const cfg = JSON.parse(job.job_config);
        const proc = cfg?.config?.process?.[0];
        if (proc && proc.network) {
          // Keep user settings; no destructive changes here. Just ensure training_folder remains.
          proc.training_folder = proc.training_folder || trainingRoot;
        }
        await prisma.job.update({ where: { id: jobID }, data: { job_config: JSON.stringify(cfg) } });
      }
    } catch (e) {
      // ignore non-fatal config issues
    }

    return NextResponse.json({ ok: true, removed: baseName });
  } catch (error) {
    console.error('Delete base checkpoint error:', error);
    return NextResponse.json({ error: 'Failed to remove base checkpoint' }, { status: 500 });
  }
}
