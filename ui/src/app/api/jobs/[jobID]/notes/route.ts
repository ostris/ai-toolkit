import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

const getNotesPath = async (jobID: string): Promise<string | null> => {
  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return null;
  }

  const trainingFolder = await getTrainingFolder();
  return path.join(trainingFolder, job.name, 'notes.md');
};

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const notesPath = await getNotesPath(jobID);

  if (!notesPath) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  let content = '';
  try {
    content = await fs.promises.readFile(notesPath, 'utf-8');
  } catch {
    // no notes yet, start empty
  }

  return NextResponse.json({ content });
}

export async function POST(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const notesPath = await getNotesPath(jobID);

  if (!notesPath) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const body = await request.json();
  const content = typeof body?.content === 'string' ? body.content : '';

  try {
    // the job folder may not exist yet if the job has never run
    await fs.promises.mkdir(path.dirname(notesPath), { recursive: true });
    await fs.promises.writeFile(notesPath, content, 'utf-8');
  } catch (error) {
    console.error('Error saving notes file:', error);
    return NextResponse.json({ error: 'Error saving notes file' }, { status: 500 });
  }

  return NextResponse.json({ success: true });
}
