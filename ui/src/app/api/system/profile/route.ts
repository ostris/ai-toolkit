import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

export async function GET(request: NextRequest) {
  try {
    // Run Python script to get system profile
    const scriptPath = path.join(process.cwd(), '..', 'toolkit', 'system_profiler.py');

    try {
      const { stdout, stderr } = await execAsync(`python3 "${scriptPath}"`, {
        timeout: 10000,
      });

      if (stderr && !stdout) {
        console.error('System profiler error:', stderr);
        return NextResponse.json({ error: 'Failed to detect system profile' }, { status: 500 });
      }

      const result = JSON.parse(stdout.trim());

      if (result.error) {
        return NextResponse.json({ error: result.error }, { status: 400 });
      }

      return NextResponse.json(result);
    } catch (execError: any) {
      console.error('System profiler execution error:', execError);

      // Return a fallback profile if detection fails
      const fallbackProfile = {
        gpu: {
          type: 'cpu_only',
          name: 'Detection failed - please configure manually',
          vramGB: 0,
          driverVersion: null,
          isUnifiedMemory: false
        },
        memory: {
          totalRAM: 16,
          availableRAM: 8,
          unifiedMemory: null
        },
        storage: {
          type: 'ssd',
          availableSpaceGB: 100
        },
        cpu: {
          cores: 4,
          name: 'Unknown'
        },
        detectionError: execError.message || 'Failed to execute system profiler'
      };

      return NextResponse.json(fallbackProfile);
    }
  } catch (error: any) {
    console.error('System profile API error:', error);
    return NextResponse.json(
      { error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
