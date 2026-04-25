import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import archiver from 'archiver';
import { Readable, PassThrough } from 'stream';
import { getTrainingFolder, getDatasetsRoot } from '@/server/settings';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  try {
    const { jobID } = await params;

    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const trainingFolder = await getTrainingFolder();
    const datasetsRoot = await getDatasetsRoot();
    const jobOutputFolder = path.join(trainingFolder, job.name);

    // Find dataset folders referenced in job config
    const datasetAbsPaths: string[] = [];
    const datasetRelPaths: string[] = [];
    try {
      const jobConfig = JSON.parse(job.job_config);
      const datasets: { folder_path?: string }[] = jobConfig?.config?.process?.[0]?.datasets ?? [];
      for (const ds of datasets) {
        if (!ds.folder_path) continue;
        const absPath = ds.folder_path;
        const relPath = path.relative(datasetsRoot, absPath);
        if (!relPath.startsWith('..') && fs.existsSync(absPath)) {
          datasetAbsPaths.push(absPath);
          datasetRelPaths.push(relPath);
        }
      }
    } catch {
      // ignore config parse errors
    }

    const hasOutputFolder = fs.existsSync(jobOutputFolder);

    const manifest = {
      version: 1,
      exportedAt: new Date().toISOString(),
      job: {
        name: job.name,
        gpu_ids: job.gpu_ids,
        job_config: JSON.parse(job.job_config),
        status: 'stopped',
        step: job.step,
        job_type: job.job_type,
        job_ref: job.job_ref ?? null,
      },
      paths: {
        outputFolder: hasOutputFolder ? job.name : null,
        datasetFolders: datasetRelPaths,
      },
    };

    const passThrough = new PassThrough();

    // zlib level 0 = store (no compression) — fast for large binary files
    const archive = archiver('zip', { zlib: { level: 0 } });

    archive.on('error', (err) => {
      console.error('Export archive error:', err);
      passThrough.destroy(err);
    });

    archive.pipe(passThrough);

    archive.append(JSON.stringify(manifest, null, 2), { name: 'manifest.json' });

    if (hasOutputFolder) {
      archive.directory(jobOutputFolder, `output/${job.name}`);
    }

    for (let i = 0; i < datasetAbsPaths.length; i++) {
      archive.directory(datasetAbsPaths[i], `datasets/${datasetRelPaths[i]}`);
    }

    archive.finalize();

    const webStream = Readable.toWeb(passThrough) as ReadableStream;

    return new NextResponse(webStream, {
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': `attachment; filename="${encodeURIComponent(job.name)}_export.zip"`,
      },
    });
  } catch (err: any) {
    console.error('Export error:', err);
    return NextResponse.json({ error: err.message ?? 'Export failed' }, { status: 500 });
  }
}
