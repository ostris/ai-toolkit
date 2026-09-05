import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

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
    return NextResponse.json({ keys: [], key: 'loss', points: [] });
  }

  const url = new URL(request.url);
  const key = url.searchParams.get('key') ?? 'loss';
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 2000), 20000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const stride = Math.max(1, Number(url.searchParams.get('stride') ?? 1));

  const db = openDb(logPath);

  try {
    const keysRows = await all<{ key: string }>(db, `SELECT key FROM metric_keys ORDER BY key ASC`);
    const keys = keysRows.map((r) => r.key);

    const points = await all<{
      step: number;
      wall_time: number;
      value: number | null;
      value_text: string | null;
    }>(
      db,
      `
      SELECT
        m.step AS step,
        s.wall_time AS wall_time,
        m.value_real AS value,
        m.value_text AS value_text
      FROM metrics m
      JOIN steps s ON s.step = m.step
      WHERE m.key = ?
        AND (? IS NULL OR m.step > ?)
        AND (m.step % ?) = 0
      ORDER BY m.step ASC
      LIMIT ?
      `,
      [key, sinceStep, sinceStep, stride, limit]
    );

    return NextResponse.json({
      key,
      keys,
      points: points.map((p) => ({
        step: p.step,
        wall_time: p.wall_time,
        value: p.value ?? (p.value_text ? Number(p.value_text) : null),
      })),
    });
  } finally {
    await closeDb(db);
  }
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
