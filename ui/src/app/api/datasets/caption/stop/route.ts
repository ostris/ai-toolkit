import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    console.log('Stopping captioning service...');
    
    // Try to stop the service gracefully first by calling the shutdown endpoint
    const CAPTION_SERVICE_URL = process.env.CAPTION_SERVICE_URL || 'http://127.0.0.1:5000';
    
    try {
      const response = await fetch(`${CAPTION_SERVICE_URL}/shutdown`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        timeout: 5000 // 5 second timeout
      });
      
      if (response.ok) {
        console.log('Caption service stopped gracefully');
        return NextResponse.json({ 
          success: true, 
          message: 'Caption service stopped successfully' 
        });
      }
    } catch (error) {
      console.log('Graceful shutdown failed, trying force stop...');
    }
    
    // If graceful shutdown fails, try to kill the process
    return new Promise((resolve) => {
      const killProcess = spawn('pkill', ['-f', 'caption_server.py'], {
        stdio: 'pipe'
      });
      
      let output = '';
      let errorOutput = '';
      
      killProcess.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      killProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      killProcess.on('close', (code) => {
        console.log(`Kill process exited with code ${code}`);
        console.log('Output:', output);
        console.log('Error output:', errorOutput);
        
        if (code === 0 || code === 1) { // 0 = success, 1 = no process found (also success)
          resolve(NextResponse.json({ 
            success: true, 
            message: 'Caption service stopped successfully' 
          }));
        } else {
          resolve(NextResponse.json({ 
            success: false, 
            error: `Failed to stop service (exit code: ${code})` 
          }, { status: 500 }));
        }
      });
      
      killProcess.on('error', (error) => {
        console.error('Error stopping caption service:', error);
        resolve(NextResponse.json({ 
          success: false, 
          error: 'Failed to stop caption service: ' + error.message 
        }, { status: 500 }));
      });
      
      // Timeout after 10 seconds
      setTimeout(() => {
        killProcess.kill();
        resolve(NextResponse.json({ 
          success: false, 
          error: 'Timeout stopping caption service' 
        }, { status: 500 }));
      }, 10000);
    });
    
  } catch (error: any) {
    console.error('Error stopping caption service:', error);
    return NextResponse.json({ 
      success: false, 
      error: 'Failed to stop caption service: ' + error.message 
    }, { status: 500 });
  }
}
