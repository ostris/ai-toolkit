import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { folderPath, recursive } = body;

    if (!folderPath || typeof folderPath !== 'string') {
      return NextResponse.json({ error: 'folderPath is required' }, { status: 400 });
    }

    // Prevent path traversal
    const normalizedPath = path.normalize(folderPath);
    if (normalizedPath.includes('..')) {
      return NextResponse.json({ error: 'Invalid folder path' }, { status: 400 });
    }

    // Check if folder exists
    if (!fs.existsSync(normalizedPath) || !fs.statSync(normalizedPath).isDirectory()) {
      return NextResponse.json({ error: 'Folder not found' }, { status: 404 });
    }

    const foldersToAdd: string[] = [normalizedPath];

    // If recursive, also collect all subfolders
    if (recursive) {
      const subfolders = getSubfoldersRecursively(normalizedPath);
      foldersToAdd.push(...subfolders);
    }

    // Upsert all folders (ignore duplicates)
    const added: string[] = [];
    for (const fp of foldersToAdd) {
      await prisma.galleryFolder.upsert({
        where: { path: fp },
        update: {},
        create: { path: fp },
      });
      added.push(fp);
    }

    return NextResponse.json({ added });
  } catch (error) {
    console.error('Error adding gallery folder:', error);
    return NextResponse.json({ error: 'Failed to add gallery folder' }, { status: 500 });
  }
}

function getSubfoldersRecursively(dir: string): string[] {
  const results: string[] = [];
  try {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    for (const item of items) {
      if (item.isDirectory() && !item.name.startsWith('.')) {
        const subPath = path.join(dir, item.name);
        results.push(subPath);
        results.push(...getSubfoldersRecursively(subPath));
      }
    }
  } catch {
    // ignore errors reading subdirs
  }
  return results;
}
