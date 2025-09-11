/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import archiver from 'archiver';
import { getTrainingFolder } from '@/server/settings';

export const runtime = 'nodejs'; // ensure Node APIs are available
export const dynamic = 'force-dynamic'; // long-running, non-cached

type PostBody = {
  zipTarget: 'samples'; //only samples for now
  jobName: string;
};

async function resolveSafe(p: string) {
  // resolve symlinks + normalize
  return await fsp.realpath(p);
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as PostBody;
    if (!body || !body.jobName) {
      return NextResponse.json({ error: 'jobName is required' }, { status: 400 });
    }

    const trainingRoot = await resolveSafe(await getTrainingFolder());
    const folderPath = await resolveSafe(path.join(trainingRoot, body.jobName, 'samples'));
    const outputPath = path.resolve(trainingRoot, body.jobName, 'samples.zip');

    // Must be a directory
    let stat: fs.Stats;
    try {
      stat = await fsp.stat(folderPath);
    } catch {
      return new NextResponse('Folder not found', { status: 404 });
    }
    if (!stat.isDirectory()) {
      return new NextResponse('Not a directory', { status: 400 });
    }

    // delete current one if it exists
    if (fs.existsSync(outputPath)) {
      await fsp.unlink(outputPath);
    }

    // Create write stream & archive
    await new Promise<void>((resolve, reject) => {
      const output = fs.createWriteStream(outputPath);
      const archive = archiver('zip', { zlib: { level: 9 } });

      output.on('close', () => resolve());
      output.on('error', reject);
      archive.on('error', reject);

      archive.pipe(output);

      // Add the directory contents (place them under the folder's base name in the zip)
      const rootName = path.basename(folderPath);
      archive.directory(folderPath, rootName);

      archive.finalize().catch(reject);
    });

    // Return the absolute path so your existing /api/files/[...filePath] can serve it
    // Example download URL (client-side): `/api/files/${encodeURIComponent(resolvedOutPath)}`
    return NextResponse.json({
      ok: true,
      zipPath: outputPath,
      fileName: path.basename(outputPath),
    });
  } catch (err) {
    console.error('Zip error:', err);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
