import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    // Parse request body to get GPU selection
    const body = await request.json().catch(() => ({}));
    const selectedGpu = body.gpu !== undefined ? body.gpu : 0; // Default to GPU 0

    // Get the correct path - we're in ui/ directory, need to go up one level
    const projectRoot = path.resolve(process.cwd(), '..');
    const scriptPath = path.join(projectRoot, 'captioning_service', 'start_service.sh');

    console.log('Project root:', projectRoot);
    console.log('Script path:', scriptPath);

    // Check if script exists
    const fs = require('fs');
    if (!fs.existsSync(scriptPath)) {
      return NextResponse.json({
        success: false,
        error: `Script not found at: ${scriptPath}`
      }, { status: 404 });
    }

    // Start the captioning service in the background
    const child = spawn('bash', [scriptPath], {
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'], // Capture stdout/stderr for debugging
      cwd: projectRoot,
      env: {
        ...process.env,
        CAPTION_HOST: '127.0.0.1',
        CAPTION_PORT: '5000',
        CUDA_VISIBLE_DEVICES: selectedGpu.toString(),
      }
    });

    // Log output for debugging
    child.stdout?.on('data', (data) => {
      console.log('Caption service stdout:', data.toString());
    });

    child.stderr?.on('data', (data) => {
      console.error('Caption service stderr:', data.toString());
    });

    // Detach the process so it runs independently
    child.unref();

    return NextResponse.json({
      success: true,
      message: 'Captioning service is starting in the background',
      pid: child.pid,
      scriptPath,
      projectRoot,
    });

  } catch (error: any) {
    console.error('Failed to start captioning service:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Failed to start captioning service'
    }, { status: 500 });
  }
}
