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

    // Parse each line as JSON
    const allMetrics = lines.map(line => {
      try {
        return JSON.parse(line);
      } catch (e) {
        console.error('Error parsing metrics line:', e);
        return null;
      }
    }).filter(m => m !== null);

    // Downsample to max 500 points for chart performance
    // Always include first and last, evenly distribute the rest
    let metrics = allMetrics;
    if (allMetrics.length > 500) {
      const step = Math.floor(allMetrics.length / 499); // 499 + first = 500
      metrics = allMetrics.filter((_, idx) => idx === 0 || idx === allMetrics.length - 1 || idx % step === 0);

      // Ensure we don't exceed 500 points
      if (metrics.length > 500) {
        metrics = metrics.slice(0, 500);
      }
    }

    return NextResponse.json({ metrics, total: allMetrics.length });
  } catch (error) {
    console.error('Error reading metrics file:', error);
    return NextResponse.json({ metrics: [], error: 'Error reading metrics file' });
  }
}
