import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const { id } = await request.json();

    if (!id) {
      return NextResponse.json({ error: 'id is required' }, { status: 400 });
    }

    const download = await prisma.videoDownload.findUnique({ where: { id } });
    if (!download) {
      return NextResponse.json({ error: 'Download not found' }, { status: 404 });
    }

    // Mark cancelled before sending the signal so the close handler doesn't overwrite it
    await prisma.videoDownload.update({
      where: { id },
      data: { status: 'cancelled', speed: '' },
    });

    // Terminate the yt-dlp process if it is still running
    if (download.pid) {
      try {
        process.kill(download.pid, 'SIGTERM');
      } catch {
        // Process may have already exited — ignore
      }
    }

    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to cancel download' }, { status: 500 });
  }
}
