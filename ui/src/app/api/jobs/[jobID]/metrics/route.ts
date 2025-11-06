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

    // Extract switch_boundary_every from job config for MoE expert inference
    let switchBoundaryEvery = 100; // Default fallback
    try {
      const jobConfig = typeof job.job_config === 'string' ? JSON.parse(job.job_config) : job.job_config;
      switchBoundaryEvery = jobConfig?.config?.process?.[0]?.train?.switch_boundary_every || 100;
    } catch (e) {
      console.error('Error parsing job config for switch_boundary_every:', e);
    }

    // Helper to infer expert from step number (for MoE training)
    const inferExpert = (step: number): string => {
      const blockIndex = Math.floor(step / switchBoundaryEvery);
      return blockIndex % 2 === 0 ? 'high_noise' : 'low_noise';
    };

    // Separate metrics by expert BEFORE downsampling
    // This prevents adding a step for one expert from changing which steps are included for the other expert
    const highNoiseMetrics = allMetrics.filter(m => {
      const expert = m.expert || inferExpert(m.step);
      return expert === 'high_noise';
    });
    const lowNoiseMetrics = allMetrics.filter(m => {
      const expert = m.expert || inferExpert(m.step);
      return expert === 'low_noise';
    });

    // Downsample each expert separately to max 250 points (500 total across both experts)
    const downsampleExpert = (expertMetrics: any[], maxPoints: number) => {
      if (expertMetrics.length <= maxPoints) return expertMetrics;

      const lastIdx = expertMetrics.length - 1;
      const step = Math.floor(expertMetrics.length / (maxPoints - 2)); // Leave room for first and last

      // Get evenly distributed middle points
      const middleIndices = new Set<number>();
      for (let i = step; i < lastIdx; i += step) {
        middleIndices.add(i);
        if (middleIndices.size >= maxPoints - 2) break;
      }

      // Always include first and last
      return expertMetrics.filter((_, idx) =>
        idx === 0 || idx === lastIdx || middleIndices.has(idx)
      );
    };

    const downsampledHighNoise = downsampleExpert(highNoiseMetrics, 250);
    const downsampledLowNoise = downsampleExpert(lowNoiseMetrics, 250);

    // Merge back together and sort by step
    const metrics = [...downsampledHighNoise, ...downsampledLowNoise].sort((a, b) => a.step - b.step);

    return NextResponse.json({
      metrics,
      total: allMetrics.length,
      switchBoundaryEvery
    });
  } catch (error) {
    console.error('Error reading metrics file:', error);
    return NextResponse.json({ metrics: [], error: 'Error reading metrics file' });
  }
}
