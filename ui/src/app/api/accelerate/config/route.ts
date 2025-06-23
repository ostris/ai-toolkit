import fs from 'fs';
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';

const TOOLKIT_ROOT = process.env.TOOLKIT_ROOT || process.cwd();

export async function GET() {
  try {
    const configPath = path.join(TOOLKIT_ROOT, 'accelerate_config.yaml');
    
    if (!fs.existsSync(configPath)) {
      return NextResponse.json({ error: 'Accelerate config not found' }, { status: 404 });
    }

    const configContent = fs.readFileSync(configPath, 'utf8');
    return NextResponse.json({ config: configContent });
  } catch (error: any) {
    console.error('Error reading accelerate config:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { config } = body;

    if (!config) {
      return NextResponse.json({ error: 'Config content is required' }, { status: 400 });
    }

    const configPath = path.join(TOOLKIT_ROOT, 'accelerate_config.yaml');
    fs.writeFileSync(configPath, config);

    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('Error writing accelerate config:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
} 