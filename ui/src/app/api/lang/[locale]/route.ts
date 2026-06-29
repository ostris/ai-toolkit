import fs from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';

const localePattern = /^[a-z]{2}_[A-Z]{2}$/;

export async function GET(_request: Request, { params }: { params: Promise<{ locale: string }> }) {
  const { locale } = await params;

  if (!localePattern.test(locale)) {
    return NextResponse.json({ error: 'Invalid locale' }, { status: 400 });
  }

  const filePath = path.join(process.cwd(), 'lang', `${locale}.json`);

  try {
    const contents = await fs.readFile(filePath, 'utf8');
    return NextResponse.json(JSON.parse(contents));
  } catch (error) {
    return NextResponse.json({ error: 'Language file not found' }, { status: 404 });
  }
}
