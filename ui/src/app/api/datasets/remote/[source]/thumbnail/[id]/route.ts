/**
 * GET /api/datasets/remote/[source]/thumbnail/[id]?type=<thumbnail_type>
 *
 * Delegates entirely to the Python plugin via the CLI `thumbnail` command.
 * The core app has no knowledge of how individual plugins authenticate or
 * construct thumbnail URLs.
 */
import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';
import { existsSync } from 'fs';
import { TOOLKIT_ROOT } from '@/paths';

function getPythonPath(): string {
  for (const dir of ['.venv', 'venv']) {
    const p = path.join(TOOLKIT_ROOT, dir, 'bin', 'python');
    if (existsSync(p)) return p;
    const pw = path.join(TOOLKIT_ROOT, dir, 'Scripts', 'python.exe');
    if (existsSync(pw)) return pw;
  }
  return 'python';
}

export async function GET(
  request: NextRequest,
  { params }: { params: { source: string; id: string } },
) {
  const { source, id } = params;
  const type = request.nextUrl.searchParams.get('type') || '';

  return new Promise<NextResponse>(resolve => {
    const child = spawn(
      getPythonPath(),
      [
        '-m', 'toolkit.dataset_sources.cli',
        'thumbnail', source, decodeURIComponent(id), type,
      ],
      { cwd: TOOLKIT_ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
    );

    let out = '';
    child.stdout.on('data', (d: Buffer) => (out += d.toString()));

    child.on('close', code => {
      try {
        const data = JSON.parse(out);
        if (data.error || code !== 0) {
          resolve(new NextResponse(data.error || 'Thumbnail fetch failed', { status: 502 }));
          return;
        }
        const imageBuffer = Buffer.from(data.data as string, 'base64');
        resolve(
          new NextResponse(imageBuffer, {
            status: 200,
            headers: {
              'Content-Type': data.content_type as string,
              'Cache-Control': 'public, max-age=86400',
            },
          }),
        );
      } catch {
        resolve(new NextResponse('Invalid response from plugin', { status: 500 }));
      }
    });
  });
}

