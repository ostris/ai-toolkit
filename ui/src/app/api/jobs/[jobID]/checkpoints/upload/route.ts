import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { mkdir, writeFile } from 'fs/promises';
import fs from 'fs';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';

const prisma = new PrismaClient();

type InferredInfo = {
  type: 'lora' | 'dora' | 'lokr' | 'unknown';
  linear?: number;
  linear_alpha?: number;
  lokr_factor?: number;
  metadata?: Record<string, any>;
};

// Parse safetensors header without loading tensor payloads
function parseSafetensorsHeader(buffer: Buffer): { header: any; metadata: Record<string, string> | undefined } {
  if (buffer.length < 8) throw new Error('File too small to be a safetensors file');
  const headerLen = buffer.readBigUInt64LE(0);
  const headerStart = BigInt(8);
  const headerEnd = headerStart + headerLen;
  if (headerEnd > BigInt(buffer.length)) throw new Error('Invalid safetensors header length');
  const headerJson = buffer.slice(Number(headerStart), Number(headerEnd)).toString('utf-8');
  const header = JSON.parse(headerJson);
  const metadata = header['__metadata__'] as Record<string, string> | undefined;
  return { header, metadata };
}

function inferFromHeader(headerObj: any): InferredInfo {
  const info: InferredInfo = { type: 'unknown' };
  const keys = Object.keys(headerObj).filter(k => k !== '__metadata__');

  // Determine network type
  const hasLokr = keys.some(k => k.includes('lokr_')) || keys.some(k => k.startsWith('lycoris_') && k.includes('lokr'));
  const hasDoraMagnitude = keys.some(k => k.endsWith('.magnitude') || k.includes('.magnitude'));
  const isLora = keys.some(k => k.includes('lora_A') || k.includes('lora_down'));
  if (hasLokr) info.type = 'lokr';
  else if (hasDoraMagnitude) info.type = 'dora';
  else if (isLora) info.type = 'lora';

  try {
    if (info.type === 'lora' || info.type === 'dora') {
      // Find a lora_A or lora_down.weight entry and use shape[0] as rank
      const loraKey = keys.find(k => k.includes('lora_A')) || keys.find(k => k.includes('lora_down'));
      if (loraKey) {
        const shape = headerObj[loraKey]?.shape;
        if (Array.isArray(shape) && shape.length >= 2) {
          info.linear = Number(shape[0]);
          info.linear_alpha = info.linear;
        }
      }
    } else if (info.type === 'lokr') {
      // Find largest factor among lokr_w1 or lokr_w1_a
      let factor = -1;
      for (const k of keys) {
        if (k.includes('lokr_w1')) {
          const shape = headerObj[k]?.shape;
          if (Array.isArray(shape) && shape.length >= 1) {
            factor = Math.max(factor, Number(shape[0]));
          }
        }
      }
      if (factor > 0) info.lokr_factor = factor;
    }
  } catch (_) {
    // best-effort inference; ignore errors
  }

  return info;
}

function parseMetadata(metaStrings?: Record<string, string>): Record<string, any> | undefined {
  if (!metaStrings) return undefined;
  const out: Record<string, any> = {};
  for (const [k, v] of Object.entries(metaStrings)) {
    try {
      out[k] = JSON.parse(v);
    } catch {
      out[k] = v;
    }
  }
  return out;
}

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  try {
    const { jobID } = params;

    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const formData = await request.formData();
    const file = formData.get('file') as unknown as File | null;
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const bytes = Buffer.from(await file.arrayBuffer());

    // Validate header and infer params
    let headerParsed;
    try {
      headerParsed = parseSafetensorsHeader(bytes);
    } catch (e: any) {
      return NextResponse.json({ error: `Invalid safetensors file: ${e?.message || e}` }, { status: 400 });
    }
    const inferred = inferFromHeader(headerParsed.header);
    const parsedMeta = parseMetadata(headerParsed.metadata);

    const trainingRoot = await getTrainingFolder();
    // Resolve paths to prevent path traversal via job.name
    const resolvedTrainingRoot = path.resolve(trainingRoot);
    const jobFolder = path.resolve(resolvedTrainingRoot, job.name);
    if (!(jobFolder + path.sep).startsWith(resolvedTrainingRoot + path.sep)) {
      return NextResponse.json({ error: 'Invalid job folder' }, { status: 400 });
    }
    await mkdir(jobFolder, { recursive: true });

    // Enforce single base checkpoint using a marker file
    const baseMarker = path.join(jobFolder, '.base_checkpoint');
    if (fs.existsSync(baseMarker)) {
      return NextResponse.json(
        { error: 'A base checkpoint is already set. Remove it before uploading a new one.' },
        { status: 409 },
      );
    }

    // Sanitize filename and ensure it starts with job.name for resume compatibility
    const baseName = job.name.replace(/[^a-zA-Z0-9._-]/g, '_');
    const finalName = `${baseName}_base_ckpt.safetensors`;
    const savePath = path.join(jobFolder, finalName);

    await writeFile(savePath, bytes);
    // Write marker with saved filename
    try {
      await writeFile(baseMarker, finalName, { encoding: 'utf-8' });
    } catch (e) {
      console.warn('Failed to write base checkpoint marker:', e);
    }

    // Update job config network fields based on inference
    try {
      if (job.job_config) {
        const cfg = JSON.parse(job.job_config);
        const proc = cfg?.config?.process?.[0];
        if (proc) {
          proc.training_folder = proc.training_folder || trainingRoot;
          proc.network = proc.network || {};

          if (inferred.type === 'lora' || inferred.type === 'dora') {
            proc.network.type = inferred.type;
            if (typeof inferred.linear === 'number' && inferred.linear > 0) {
              proc.network.linear = inferred.linear;
              proc.network.linear_alpha = inferred.linear_alpha ?? inferred.linear;
            }
          } else if (inferred.type === 'lokr') {
            proc.network.type = 'lokr';
            if (typeof inferred.lokr_factor === 'number' && inferred.lokr_factor > 0) {
              proc.network.lokr_full_rank = true;
              proc.network.lokr_factor = inferred.lokr_factor;
            }
          }

          // Ensure network_kwargs exists
          proc.network.network_kwargs = proc.network.network_kwargs || { ignore_if_contains: [] };

          await prisma.job.update({
            where: { id: jobID },
            data: { job_config: JSON.stringify(cfg) },
          });
        }
      }
    } catch (e) {
      // non-fatal
      console.warn('Failed to update job config from uploaded weights', e);
    }

    return NextResponse.json({
      message: 'Checkpoint uploaded',
      saved_as: finalName,
      inferred,
      metadata: parsedMeta,
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: 'Error uploading checkpoint' }, { status: 500 });
  }
}
