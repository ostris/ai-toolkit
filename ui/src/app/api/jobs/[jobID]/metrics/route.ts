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
  const metricsPath = path.join(jobFolder, `metrics_${job.name}.jsonl`);

  if (!fs.existsSync(metricsPath)) {
    return NextResponse.json({ metrics: [] });
  }

  try {
    // Read the JSONL file
    const fileContent = fs.readFileSync(metricsPath, 'utf-8');
    const lines = fileContent.trim().split('\n').filter(line => line.trim());

    // Get last 1000 entries (or all if less)
    const recentLines = lines.slice(-1000);

    // Parse each line as JSON
    const metrics = recentLines.map(line => {
      try {
        return JSON.parse(line);
      } catch (e) {
        console.error('Error parsing metrics line:', e);
        return null;
      }
    }).filter(m => m !== null);

    return NextResponse.json({ metrics });
  } catch (error) {
    console.error('Error reading metrics file:', error);
    return NextResponse.json({ metrics: [], error: 'Error reading metrics file' });
  }
}
