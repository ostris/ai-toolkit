import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDataRoot } from '@/server/settings';

function getNotesDir(dataRoot: string) {
  return path.join(dataRoot, 'notes');
}

function getNotesFilePath(dataRoot: string, datasetName: string) {
  return path.join(getNotesDir(dataRoot), `${datasetName}.txt`);
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const datasetName = searchParams.get('datasetName');

    const dataRoot = await getDataRoot();

    // If no datasetName, return list of all datasets that have notes
    if (!datasetName) {
      const notesDir = getNotesDir(dataRoot);
      const datasetsWithNotes: string[] = [];
      if (fs.existsSync(notesDir)) {
        for (const file of fs.readdirSync(notesDir)) {
          if (file.endsWith('.txt')) {
            datasetsWithNotes.push(file.slice(0, -4));
          }
        }
      }
      return NextResponse.json({ datasetsWithNotes });
    }

    const notesPath = getNotesFilePath(dataRoot, datasetName);

    let notes = '';
    if (fs.existsSync(notesPath)) {
      notes = fs.readFileSync(notesPath, 'utf-8');
    }

    return NextResponse.json({ notes });
  } catch (error) {
    console.error('Error reading notes:', error);
    return NextResponse.json({ error: 'Failed to read notes' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { datasetName, notes } = body;

    if (!datasetName || typeof datasetName !== 'string') {
      return NextResponse.json({ error: 'datasetName is required' }, { status: 400 });
    }

    const dataRoot = await getDataRoot();
    const notesDir = getNotesDir(dataRoot);
    const notesPath = getNotesFilePath(dataRoot, datasetName);

    if (!notes || notes.trim() === '') {
      // Delete the file if notes are empty
      if (fs.existsSync(notesPath)) {
        fs.unlinkSync(notesPath);
      }
    } else {
      // Create notes directory if needed
      if (!fs.existsSync(notesDir)) {
        fs.mkdirSync(notesDir, { recursive: true });
      }
      fs.writeFileSync(notesPath, notes, 'utf-8');
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error saving notes:', error);
    return NextResponse.json({ error: 'Failed to save notes' }, { status: 500 });
  }
}
