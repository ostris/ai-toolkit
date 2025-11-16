import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

/**
 * GET /api/jobs/[jobID]/analysis
 *
 * Retrieves performance analysis for a training job, including:
 * - Summary statistics (memory, duration, steps)
 * - Priority-ranked recommendations
 * - Log errors and warnings
 * - Health score (good/warning/critical)
 */
export async function GET(
  request: NextRequest,
  { params }: { params: { jobID: string } }
) {
  try {
    const jobID = params.jobID;

    // Run Python analysis script
    const scriptPath = path.join(process.cwd(), '..', 'toolkit', 'monitoring', 'api_helper.py');
    const dbPath = path.join(process.cwd(), '..', 'aitk_db.db');

    const { stdout, stderr } = await execAsync(
      `python3 "${scriptPath}" analyze "${jobID}" "${dbPath}"`,
      {
        timeout: 30000,
        cwd: path.join(process.cwd(), '..'),
      }
    );

    if (stderr && !stderr.includes('Warning')) {
      console.error('Analysis script error:', stderr);
    }

    const result = JSON.parse(stdout);

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 404 });
    }

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Analysis API error:', error);

    // If no analysis data exists, return empty response
    if (error.message?.includes('No analysis data')) {
      return NextResponse.json({
        summary: null,
        recommendations: [],
        log_errors: [],
        health_score: 'unknown',
        metrics_timeline: [],
      });
    }

    return NextResponse.json(
      { error: error.message || 'Failed to fetch analysis' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/jobs/[jobID]/analysis
 *
 * Triggers analysis for a completed training job.
 * Can be called after training completes to generate recommendations.
 */
export async function POST(
  request: NextRequest,
  { params }: { params: { jobID: string } }
) {
  try {
    const jobID = params.jobID;
    const body = await request.json();
    const jobName = body.jobName || '';

    const scriptPath = path.join(process.cwd(), '..', 'toolkit', 'monitoring', 'api_helper.py');
    const dbPath = path.join(process.cwd(), '..', 'aitk_db.db');
    const outputFolder = path.join(process.cwd(), '..', 'output');

    const { stdout, stderr } = await execAsync(
      `python3 "${scriptPath}" generate "${jobID}" "${dbPath}" "${jobName}" "${outputFolder}"`,
      {
        timeout: 60000,
        cwd: path.join(process.cwd(), '..'),
      }
    );

    if (stderr && !stderr.includes('Warning')) {
      console.error('Analysis generation error:', stderr);
    }

    const result = JSON.parse(stdout);

    if (result.error) {
      return NextResponse.json({ error: result.error }, { status: 400 });
    }

    return NextResponse.json(result);
  } catch (error: any) {
    console.error('Analysis generation error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to generate analysis' },
      { status: 500 }
    );
  }
}
