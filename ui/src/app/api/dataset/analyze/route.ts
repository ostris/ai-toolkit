import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs/promises';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    const { path: datasetPath } = await request.json();

    if (!datasetPath) {
      return NextResponse.json({ error: 'Dataset path is required' }, { status: 400 });
    }

    // Validate path exists
    try {
      await fs.access(datasetPath);
    } catch (error) {
      return NextResponse.json({ error: 'Dataset path does not exist' }, { status: 404 });
    }

    // Run Python script to analyze dataset
    const scriptPath = path.join(process.cwd(), '..', 'toolkit', 'dataset_analyzer.py');

    try {
      const { stdout, stderr } = await execAsync(`python3 "${scriptPath}" "${datasetPath}"`, {
        timeout: 30000,
      });

      if (stderr && !stdout) {
        console.error('Dataset analysis error:', stderr);
        return NextResponse.json({ error: 'Failed to analyze dataset' }, { status: 500 });
      }

      const result = JSON.parse(stdout.trim());

      if (result.error) {
        return NextResponse.json({ error: result.error }, { status: 400 });
      }

      return NextResponse.json(result);
    } catch (execError: any) {
      console.error('Dataset analysis execution error:', execError);
      return NextResponse.json(
        { error: execError.message || 'Failed to execute dataset analysis' },
        { status: 500 }
      );
    }
  } catch (error: any) {
    console.error('Dataset analysis error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
