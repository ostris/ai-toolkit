import fs from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';

interface LanguageInfo {
  locale: string;
  name: string;
}

export async function GET() {
  const langDir = path.join(process.cwd(), 'lang');

  try {
    const files = await fs.readdir(langDir);
    const languages = await Promise.all(
      files
        .filter(file => file.endsWith('.json'))
        .map(async file => {
          const locale = file.replace(/\.json$/, '');
          const contents = await fs.readFile(path.join(langDir, file), 'utf8');
          const messages = JSON.parse(contents);
          return {
            locale,
            name: messages['common.languageName'] || locale,
          } as LanguageInfo;
        }),
    );

    return NextResponse.json(
      [{ locale: 'en_US', name: 'English' }, ...languages].sort((a, b) => a.name.localeCompare(b.name)),
    );
  } catch (error) {
    return NextResponse.json([{ locale: 'en_US', name: 'English' }]);
  }
}
