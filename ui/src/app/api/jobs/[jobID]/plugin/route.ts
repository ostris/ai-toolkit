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
  const pluginPath = path.join(jobFolder, 'plugin.html');

  if (!fs.existsSync(pluginPath)) {
    return NextResponse.json({ exists: false, html: null });
  }

  // lightweight existence check used to decide if the Plugin tab should show
  if (request.nextUrl.searchParams.get('check') === '1') {
    return NextResponse.json({ exists: true, html: null });
  }

  // serve the raw html so it can be loaded directly as an iframe src
  let html = '';
  try {
    html = fs.readFileSync(pluginPath, 'utf-8');
  } catch (error) {
    console.error('Error reading plugin file:', error);
    return NextResponse.json({ error: 'Error reading plugin file' }, { status: 500 });
  }
  return new NextResponse(html, {
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-store',
    },
  });
}
