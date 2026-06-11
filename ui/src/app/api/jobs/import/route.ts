import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import os from 'os';
import { Readable, Transform } from 'stream';
import busboy from 'busboy';
import { getTrainingFolder, getDatasetsRoot } from '@/server/settings';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const unzipStream = require('unzip-stream');

const prisma = new PrismaClient();

// Next.js may buffer the entire request body and deliver it as a single chunk.
// Node.js stream internals reject chunks > INT32_MAX (2,147,483,647 bytes) via validateInt32.
// Split into 512 MB pieces — at most 6 pieces for a 3 GB file.
const MAX_CHUNK = 512 * 1024 * 1024;
function rechunkTransform(): Transform {
  return new Transform({
    transform(chunk: Buffer, _enc, cb) {
      if (chunk.length <= MAX_CHUNK) { this.push(chunk); cb(); return; }
      let offset = 0;
      while (offset < chunk.length) {
        this.push(chunk.subarray(offset, Math.min(offset + MAX_CHUNK, chunk.length)));
        offset += MAX_CHUNK;
      }
      cb();
    },
  });
}

async function saveUploadToTemp(request: NextRequest): Promise<{ tempPath: string }> {
  const contentType = request.headers.get('content-type') ?? '';
  const tempPath = path.join(os.tmpdir(), `aitk_import_${Date.now()}.zip`);

  await new Promise<void>((resolve, reject) => {
    const bb = busboy({ headers: { 'content-type': contentType } });

    bb.on('file', (_field, file) => {
      const ws = fs.createWriteStream(tempPath);
      file.pipe(ws);
      ws.on('finish', resolve);
      ws.on('error', reject);
      file.on('error', reject);
    });

    bb.on('error', reject);
    const src = Readable.fromWeb(request.body as import('stream/web').ReadableStream);
    src.on('error', reject);
    src.pipe(rechunkTransform()).pipe(bb);
  });

  return { tempPath };
}

// Pass 1: read manifest.json only, drain everything else
function readManifest(zipPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const parser = unzipStream.Parse();
    let resolved = false;

    parser.on('entry', (entry: any) => {
      if (entry.path === 'manifest.json') {
        const chunks: Buffer[] = [];
        entry.on('data', (c: Buffer) => chunks.push(c));
        entry.on('end', () => {
          if (!resolved) {
            resolved = true;
            try {
              resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            } catch (e) {
              reject(e);
            }
          }
        });
        entry.on('error', reject);
      } else {
        entry.autodrain();
      }
    });

    parser.on('finish', () => {
      if (!resolved) {
        reject(Object.assign(new Error('Invalid package: manifest.json not found'), { status: 400 }));
      }
    });
    parser.on('error', reject);

    fs.createReadStream(zipPath).pipe(parser);
  });
}

// Pass 2: stream each entry directly to its destination on disk
function extractFiles(
  zipPath: string,
  manifest: any,
  targetTrainingFolder: string,
  targetDatasetsRoot: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const parser = unzipStream.Parse();
    const jobName = manifest.job.name as string;
    const writes: Promise<void>[] = [];

    const pipeEntry = (entry: any, destPath: string) => {
      fs.mkdirSync(path.dirname(destPath), { recursive: true });
      const ws = fs.createWriteStream(destPath);
      writes.push(
        new Promise<void>((res, rej) => {
          ws.on('finish', res);
          ws.on('error', rej);
          entry.on('error', rej);
        }),
      );
      entry.pipe(ws);
    };

    parser.on('entry', (entry: any) => {
      const entryPath: string = entry.path;

      if (entry.type === 'Directory' || entryPath === 'manifest.json') {
        entry.autodrain();
        return;
      }

      // Output folder
      if (manifest.paths.outputFolder) {
        const outputPrefix = `output/${manifest.paths.outputFolder}/`;
        if (entryPath.startsWith(outputPrefix)) {
          const relPath = entryPath.slice(outputPrefix.length);
          if (!relPath) { entry.autodrain(); return; }
          pipeEntry(entry, path.join(targetTrainingFolder, jobName, relPath));
          return;
        }
      }

      // Dataset folders
      for (const relFolder of manifest.paths.datasetFolders as string[]) {
        const datasetPrefix = `datasets/${relFolder}/`;
        if (entryPath.startsWith(datasetPrefix)) {
          const relPath = entryPath.slice(datasetPrefix.length);
          if (!relPath) { entry.autodrain(); return; }
          pipeEntry(entry, path.join(targetDatasetsRoot, relFolder, relPath));
          return;
        }
      }

      entry.autodrain();
    });

    parser.on('finish', () => {
      Promise.all(writes).then(() => resolve()).catch(reject);
    });
    parser.on('error', reject);

    fs.createReadStream(zipPath).pipe(parser);
  });
}

async function processZip(zipPath: string) {
  const targetTrainingFolder = await getTrainingFolder();
  const targetDatasetsRoot = await getDatasetsRoot();

  const manifest = await readManifest(zipPath);

  if (!manifest.version || !manifest.job) {
    throw Object.assign(new Error('Invalid package: manifest.json has unexpected format'), { status: 400 });
  }

  const jobName = manifest.job.name as string;

  const existing = await prisma.job.findUnique({ where: { name: jobName } });
  if (existing) {
    throw Object.assign(
      new Error(`A job named "${jobName}" already exists. Please choose a different name.`),
      { status: 409 },
    );
  }

  fs.mkdirSync(path.join(targetTrainingFolder, jobName), { recursive: true });

  await extractFiles(zipPath, manifest, targetTrainingFolder, targetDatasetsRoot);

  // Rewrite absolute paths in job_config for the target machine
  const jobConfig = JSON.parse(JSON.stringify(manifest.job.job_config));
  if (Array.isArray(jobConfig?.config?.process)) {
    for (const proc of jobConfig.config.process) {
      if ('training_folder' in proc) proc.training_folder = targetTrainingFolder;
      if ('sqlite_db_path' in proc) {
        proc.sqlite_db_path = path.join(path.dirname(targetTrainingFolder), 'aitk_db.db');
      }
      if (Array.isArray(proc.datasets)) {
        for (const ds of proc.datasets) {
          if (!ds.folder_path) continue;
          for (const relFolder of manifest.paths.datasetFolders as string[]) {
            const oldPath: string = ds.folder_path;
            if (oldPath.endsWith(`/${relFolder}`) || oldPath === relFolder) {
              ds.folder_path = path.join(targetDatasetsRoot, relFolder);
              break;
            }
          }
        }
      }
    }
  }

  const { _max } = await prisma.job.aggregate({ _max: { queue_position: true } });
  const newQueuePosition = (_max.queue_position ?? 0) + 1000;

  const newJob = await prisma.job.create({
    data: {
      name: jobName,
      gpu_ids: manifest.job.gpu_ids,
      job_config: JSON.stringify(jobConfig),
      status: 'stopped',
      step: manifest.job.step ?? 0,
      job_type: manifest.job.job_type ?? 'train',
      job_ref: manifest.job.job_ref ?? null,
      queue_position: newQueuePosition,
    },
  });

  return newJob;
}

export async function POST(request: NextRequest) {
  let tempPath: string | undefined;
  try {
    const contentType = request.headers.get('content-type') ?? '';
    let zipPath: string;

    if (contentType.includes('multipart/form-data')) {
      const saved = await saveUploadToTemp(request);
      tempPath = saved.tempPath;
      zipPath = tempPath;
    } else {
      const body = await request.json();
      zipPath = body.zipPath;
      if (!zipPath || typeof zipPath !== 'string') {
        return NextResponse.json({ error: 'zipPath is required' }, { status: 400 });
      }
      if (!fs.existsSync(zipPath) || !fs.statSync(zipPath).isFile()) {
        return NextResponse.json({ error: `File not found: ${zipPath}` }, { status: 400 });
      }
    }

    const newJob = await processZip(zipPath);
    return NextResponse.json({ success: true, job: newJob });
  } catch (error: any) {
    console.error('Import error:', error);
    const status = error.status ?? 500;
    return NextResponse.json({ error: error.message ?? 'Import failed' }, { status });
  } finally {
    if (tempPath && fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }
  }
}
