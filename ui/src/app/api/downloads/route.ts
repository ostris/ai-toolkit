import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const downloads = await prisma.videoDownload.findMany({
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ downloads });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch downloads' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { url, dataset, format, title, thumbnail, cookies_file } = body;

    if (!url || !url.trim()) {
      return NextResponse.json({ error: 'URL is required' }, { status: 400 });
    }
    if (!dataset || !dataset.trim()) {
      return NextResponse.json({ error: 'Dataset is required' }, { status: 400 });
    }

    const download = await prisma.videoDownload.create({
      data: {
        url: url.trim(),
        dataset: dataset.trim(),
        format: format?.trim() ?? '',
        title: title?.trim() ?? '',
        thumbnail: thumbnail?.trim() ?? '',
        cookies_file: cookies_file?.trim() ?? '',
      },
    });
    return NextResponse.json(download);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to create download' }, { status: 500 });
  }
}
