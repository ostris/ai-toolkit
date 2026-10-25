import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { getModelsPath } from '@/server/settings';

export const dynamic = 'force-dynamic';

// Chunked upload of a .safetensors file into <MODELS_PATH>/loras.
//   POST ?action=start&fileName=x.safetensors&size=N          -> { uploadId }
//   POST ?action=chunk&uploadId=..&offset=N   (raw body bytes) -> { received }
//   POST ?action=finish&uploadId=..&fileName=..&size=N         -> { path, name }
//   POST ?action=cancel&uploadId=..                            -> { ok }
// Chunks are written at explicit offsets so a retried chunk is idempotent.

const PART_DIR = '.uploads';
const STALE_MS = 24 * 3600 * 1000;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

const err = (msg: string, status = 400) => NextResponse.json({ error: msg }, { status });

const getRoots = async () => {
  const lorasRoot = path.resolve(await getModelsPath(), 'loras');
  const partDir = path.join(lorasRoot, PART_DIR);
  return { lorasRoot, partDir };
};

// Basename only; must be a .safetensors file.
const cleanFileName = (raw: string | null): string | null => {
  if (!raw) return null;
  const name = raw
    .trim()
    .replace(/[\\/]/g, '_')
    .replace(/[^\w.\- ()\[\]]/g, '_');
  if (!name.toLowerCase().endsWith('.safetensors') || name.startsWith('.') || name.length < 13) return null;
  return name;
};

const partPath = (partDir: string, uploadId: string | null): string | null => {
  if (!uploadId || !UUID_RE.test(uploadId)) return null;
  return path.join(partDir, `${uploadId}.part`);
};

const parseSize = (raw: string | null): number | null => {
  const n = Number(raw);
  return Number.isSafeInteger(n) && n > 0 ? n : null;
};

const removeStaleParts = async (partDir: string) => {
  let entries: string[];
  try {
    entries = await fs.promises.readdir(partDir);
  } catch {
    return;
  }
  const now = Date.now();
  for (const e of entries) {
    if (!e.endsWith('.part')) continue;
    const p = path.join(partDir, e);
    try {
      const st = await fs.promises.stat(p);
      if (now - st.mtimeMs > STALE_MS) await fs.promises.unlink(p);
    } catch { }
  }
};

export async function POST(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const action = sp.get('action');
  const { lorasRoot, partDir } = await getRoots();

  try {
    if (action === 'start') {
      const fileName = cleanFileName(sp.get('fileName'));
      if (!fileName) return err('File must be a .safetensors file');
      if (!parseSize(sp.get('size'))) return err('Invalid file size');
      const dest = path.join(lorasRoot, fileName);
      if (sp.get('overwrite') !== '1' && fs.existsSync(dest)) {
        return NextResponse.json({ error: 'A file with that name already exists', exists: true }, { status: 409 });
      }
      await fs.promises.mkdir(partDir, { recursive: true });
      await removeStaleParts(partDir);
      const uploadId = randomUUID();
      // Create empty part file so chunk writes can open it r+.
      await fs.promises.writeFile(path.join(partDir, `${uploadId}.part`), Buffer.alloc(0));
      return NextResponse.json({ uploadId, fileName });
    }

    const part = partPath(partDir, sp.get('uploadId'));
    if (!part) return err('Invalid uploadId');

    if (action === 'chunk') {
      const offset = Number(sp.get('offset'));
      if (!Number.isSafeInteger(offset) || offset < 0) return err('Invalid offset');
      if (!fs.existsSync(part)) return err('Upload not found or expired', 404);
      const buf = Buffer.from(await request.arrayBuffer());
      if (buf.length === 0) return err('Empty chunk');
      const fh = await fs.promises.open(part, 'r+');
      try {
        await fh.write(buf, 0, buf.length, offset);
      } finally {
        await fh.close();
      }
      return NextResponse.json({ received: buf.length });
    }

    if (action === 'finish') {
      const fileName = cleanFileName(sp.get('fileName'));
      const size = parseSize(sp.get('size'));
      if (!fileName || !size) return err('Invalid fileName or size');
      if (!fs.existsSync(part)) return err('Upload not found or expired', 404);
      const st = await fs.promises.stat(part);
      if (st.size !== size) {
        await fs.promises.unlink(part).catch(() => { });
        return err(`Upload incomplete: got ${st.size} of ${size} bytes`);
      }
      const dest = path.join(lorasRoot, fileName);
      if (sp.get('overwrite') !== '1' && fs.existsSync(dest)) {
        await fs.promises.unlink(part).catch(() => { });
        return NextResponse.json({ error: 'A file with that name already exists', exists: true }, { status: 409 });
      }
      await fs.promises.rename(part, dest);
      return NextResponse.json({ path: dest, name: fileName, size });
    }

    if (action === 'cancel') {
      await fs.promises.unlink(part).catch(() => { });
      return NextResponse.json({ ok: true });
    }

    return err('Unknown action');
  } catch (e: any) {
    console.error('LoRA upload error:', e);
    return err(e?.message || 'Upload failed', 500);
  }
}
