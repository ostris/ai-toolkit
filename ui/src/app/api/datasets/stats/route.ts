import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

const IMAGE_EXT = new Set([
  '.png', '.jpg', '.jpeg', '.webp',
  '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv',
  '.mp3', '.wav',
]);

const THUMB_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp']);
const THUMB_LIMIT = 8;

interface FolderStats {
  image_count: number;
  total_size: number;
  modified_at: number;
  thumbs: string[];
}

function walk(dir: string, acc: FolderStats) {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === '_controls') continue;
      walk(full, acc);
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase();
      if (!IMAGE_EXT.has(ext)) continue;
      try {
        const stat = fs.statSync(full);
        acc.image_count += 1;
        acc.total_size += stat.size;
        if (stat.mtimeMs > acc.modified_at) acc.modified_at = stat.mtimeMs;
        if (acc.thumbs.length < THUMB_LIMIT && THUMB_EXT.has(ext)) {
          acc.thumbs.push(full);
        }
      } catch {
        // best-effort
      }
    }
  }
}

export async function GET() {
  try {
    const root = await getDatasetsRoot();
    if (!fs.existsSync(root)) return NextResponse.json({ datasets: [] });

    const folders = fs
      .readdirSync(root, { withFileTypes: true })
      .filter(d => d.isDirectory() && !d.name.startsWith('.'));

    const result = folders.map(d => {
      const acc: FolderStats = { image_count: 0, total_size: 0, modified_at: 0, thumbs: [] };
      walk(path.join(root, d.name), acc);
      let folderMtime = 0;
      try {
        folderMtime = fs.statSync(path.join(root, d.name)).mtimeMs;
      } catch {}
      return {
        name: d.name,
        image_count: acc.image_count,
        total_size: acc.total_size,
        modified_at: Math.max(acc.modified_at, folderMtime),
        thumbs: acc.thumbs,
      };
    });

    return NextResponse.json({ datasets: result });
  } catch (error) {
    console.error('Error computing dataset stats:', error);
    return NextResponse.json({ error: 'Failed to compute dataset stats' }, { status: 500 });
  }
}
