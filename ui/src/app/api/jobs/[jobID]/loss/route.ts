import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';
import zlib from 'zlib';
import { promisify } from 'util';

export const runtime = 'nodejs';

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

// Serialize a metric series to JSON arrays inside SQLite. Building the arrays
// with json_group_array avoids marshaling one JS object per row, which is the
// dominant cost on multi-million-point logs. Values are truncated to 7
// significant digits (float32-level precision) to shrink the wire payload;
// value_text rows are never numeric (see _coerce_value in logging_aitk.py)
// so they serialize as null.
const SERIES_SQL = `
  SELECT json_group_array(step) AS steps, json_group_array(v) AS vals
  FROM (
    SELECT step,
           CASE WHEN value_real IS NULL THEN NULL
                ELSE CAST(printf('%.7g', value_real) AS REAL) END AS v
    FROM metrics
    WHERE key = ? AND step >= ?
    ORDER BY step ASC
  )
`;

const gzip = promisify(zlib.gzip);

// GET /api/jobs/:id/loss?since_step=N
// Returns every metric key in one response: {keys, since_step, series: {key:
// {steps: number[], values: (number|null)[]}}}. Points with step >= since_step
// are included (default 0 = everything). The body is gzipped when the client
// accepts it; a 150k-step run with a dozen keys is tens of MB uncompressed.
export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  try {
    await fs.promises.access(logPath);
  } catch {
    return NextResponse.json({ keys: [], since_step: 0, series: {} });
  }

  const url = new URL(request.url);
  const sinceStepParam = Number(url.searchParams.get('since_step') ?? 0);
  const sinceStep = Number.isFinite(sinceStepParam) ? sinceStepParam : 0;

  const db = openDb(logPath);

  let body: string;
  try {
    const keysRows = await all<{ key: string }>(db, `SELECT key FROM metric_keys ORDER BY key ASC`);
    const keys = keysRows.map((r) => r.key);

    const parts: string[] = [];
    for (const key of keys) {
      const rows = await all<{ steps: string; vals: string }>(db, SERIES_SQL, [key, sinceStep]);
      const steps = rows[0]?.steps ?? '[]';
      const vals = rows[0]?.vals ?? '[]';
      parts.push(`${JSON.stringify(key)}:{"steps":${steps},"values":${vals}}`);
    }
    body = `{"keys":${JSON.stringify(keys)},"since_step":${sinceStep},"series":{${parts.join(',')}}}`;
  } finally {
    await closeDb(db);
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-store',
    Vary: 'Accept-Encoding',
  };
  const acceptEncoding = request.headers.get('accept-encoding') ?? '';
  if (/\bgzip\b/.test(acceptEncoding)) {
    // Level 1: the payload is mostly digits, so higher levels buy ~5% for 3x
    // the CPU time.
    const gz = await gzip(Buffer.from(body), { level: 1 });
    headers['Content-Encoding'] = 'gzip';
    return new NextResponse(gz, { headers });
  }
  return new NextResponse(body, { headers });
}

function run(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<void>((resolve, reject) => {
    db.run(sql, params, (err) => (err ? reject(err) : resolve()));
  });
}

// Delete every logged step in [min_step, max_step] (inclusive) across all
// metric keys. Used by the loss graph's "Delete Selected Range" action.
export async function DELETE(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  let body: { min_step?: unknown; max_step?: unknown } = {};
  try {
    body = await request.json();
  } catch {
    // fall through to validation below
  }
  const minStep = Number(body.min_step);
  const maxStep = Number(body.max_step);
  if (!Number.isFinite(minStep) || !Number.isFinite(maxStep) || minStep > maxStep) {
    return NextResponse.json({ error: 'min_step and max_step must be numbers with min_step <= max_step' }, { status: 400 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  try {
    await fs.promises.access(logPath);
  } catch {
    return NextResponse.json({ error: 'No loss log for this job' }, { status: 404 });
  }

  const db = openDb(logPath);

  try {
    await run(db, 'BEGIN;');
    try {
      // The FK cascade on metrics only fires with PRAGMA foreign_keys=ON, so
      // delete metrics explicitly rather than relying on it.
      await run(db, `DELETE FROM metrics WHERE step >= ? AND step <= ?;`, [minStep, maxStep]);
      await run(db, `DELETE FROM steps WHERE step >= ? AND step <= ?;`, [minStep, maxStep]);
      await run(
        db,
        `DELETE FROM metric_keys WHERE NOT EXISTS (SELECT 1 FROM metrics WHERE metrics.key = metric_keys.key);`
      );
      await run(
        db,
        `UPDATE metric_keys SET
          first_seen_step = (SELECT MIN(step) FROM metrics WHERE metrics.key = metric_keys.key),
          last_seen_step = (SELECT MAX(step) FROM metrics WHERE metrics.key = metric_keys.key);`
      );
      await run(db, 'COMMIT;');
    } catch (e) {
      await run(db, 'ROLLBACK;').catch(() => { });
      throw e;
    }
    return NextResponse.json({ ok: true, min_step: minStep, max_step: maxStep });
  } catch (e: any) {
    console.error('Error deleting loss range:', e);
    return NextResponse.json({ error: e?.message ?? 'Failed to delete range' }, { status: 500 });
  } finally {
    await closeDb(db);
  }
}
