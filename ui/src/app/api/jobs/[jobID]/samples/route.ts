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

  // setup the training
  const trainingFolder = await getTrainingFolder();

  const samplesFolder = path.join(trainingFolder, job.name, 'samples');
  if (!fs.existsSync(samplesFolder)) {
    return NextResponse.json({ samples: [] });
  }

  // find all img (png, jpg, jpeg) files in the samples folder
  const samples = fs
    .readdirSync(samplesFolder)
    .filter(file => {
      return file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || file.endsWith('.webp');
    })
    .map(file => {
      return path.join(samplesFolder, file);
    })
    .sort();

  return NextResponse.json({ samples });
}
