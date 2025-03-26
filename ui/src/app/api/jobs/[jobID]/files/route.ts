import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const prisma = new PrismaClient();

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  if (!fs.existsSync(jobFolder)) {
    return NextResponse.json({ files: [] });
  }

  // find all safetensors files in the job folder
  let files = fs
    .readdirSync(jobFolder)
    .filter(file => {
      return file.endsWith('.safetensors');
    })
    .map(file => {
      return path.join(jobFolder, file);
    })
    .sort();

  // get the file size for each file
  const fileObjects = files.map(file => {
    const stats = fs.statSync(file);
    return {
      path: file,
      size: stats.size,
    };
  });

  return NextResponse.json({ files: fileObjects });
}
