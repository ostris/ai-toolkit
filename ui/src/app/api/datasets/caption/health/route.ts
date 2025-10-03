import { NextRequest, NextResponse } from 'next/server';

const CAPTION_SERVICE_URL = process.env.CAPTION_SERVICE_URL || 'http://127.0.0.1:5000';

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${CAPTION_SERVICE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (!response.ok) {
      return NextResponse.json({ 
        available: false,
        error: `Caption service returned ${response.status}`,
        serviceUrl: CAPTION_SERVICE_URL
      }, { status: 503 });
    }
    
    const data = await response.json();
    
    return NextResponse.json({
      available: data.status === 'healthy' && data.model_loaded === true,
      modelLoaded: data.model_loaded,
      gpuAvailable: data.gpu_available,
      gpuCount: data.gpu_count,
      serviceUrl: CAPTION_SERVICE_URL,
      serviceStatus: data
    });

  } catch (error: any) {
    return NextResponse.json({ 
      available: false,
      error: error.message || 'Failed to connect to caption service',
      serviceUrl: CAPTION_SERVICE_URL
    }, { status: 503 });
  }
}
