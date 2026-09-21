import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { listEngines, resolveEngine } from '@/server/inferenceEngine';

// Reverse proxy to the resident inference engine (a python job that binds an
// ephemeral loopback port and publishes it in its job folder). Everything
// under /api/inference/* is forwarded verbatim; a streaming upstream body is
// passed through untouched, so /generate frames reach the browser as they
// are produced. `?job=<id>` picks an engine when several run.
//
// /api/inference/status is answered here: it never touches the engine.
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const HOP_BY_HOP = new Set(['connection', 'keep-alive', 'transfer-encoding', 'host', 'content-length', 'authorization']);

async function proxy(request: NextRequest, segments: string[]) {
  const upstreamPath = segments.join('/');
  const jobId = request.nextUrl.searchParams.get('job');

  if (upstreamPath === 'status') {
    const engines = await listEngines(true);
    const active = jobId ? engines.find(e => e.jobId === jobId) || null : engines.find(e => e.endpoint) || engines[0] || null;
    return NextResponse.json({
      running: !!active?.endpoint,
      engine: active
        ? { jobId: active.jobId, jobName: active.jobName, gpuIds: active.gpuIds, status: active.status, ready: !!active.endpoint }
        : null,
      engines: engines.map(e => ({ jobId: e.jobId, jobName: e.jobName, gpuIds: e.gpuIds, status: e.status, ready: !!e.endpoint })),
    });
  }

  if (upstreamPath === 'outputs/delete' && request.method === 'POST') {
    // delete generated files from disk. Handled here (not by the engine) so
    // history can be cleaned up while no engine is running. Only files under
    // an inference job's outputs folder are touched.
    const body = await request.json().catch(() => ({}));
    const paths: string[] = Array.isArray(body?.paths) ? body.paths : [];
    const trainingFolder = path.resolve(await getTrainingFolder());
    const jobs = await prisma.job.findMany({ where: { job_type: 'inference' }, select: { name: true } });
    const roots = jobs.map(j => path.join(trainingFolder, j.name, 'outputs'));
    const deleted: string[] = [];
    const refused: string[] = [];
    for (const p of paths) {
      const resolved = path.resolve(String(p));
      const root = roots.find(r => resolved.startsWith(r + path.sep));
      if (!root) {
        refused.push(p);
        continue;
      }
      try {
        await fs.promises.rm(resolved, { force: true });
        deleted.push(p);
        // request folders hold one generation each; drop the folder once empty
        const dir = path.dirname(resolved);
        if (dir !== root && (await fs.promises.readdir(dir)).length === 0) {
          await fs.promises.rmdir(dir);
        }
      } catch (e: any) {
        refused.push(`${p}: ${e?.message || e}`);
      }
    }
    return NextResponse.json({ deleted, refused });
  }

  const engine = await resolveEngine(jobId);
  if (!engine || !engine.endpoint) {
    return NextResponse.json({ error: 'no inference engine is running', engine: engine ? { status: engine.status } : null }, { status: 503 });
  }

  const search = new URLSearchParams(request.nextUrl.searchParams);
  search.delete('job');
  const qs = search.toString();
  const url = `http://${engine.endpoint.host}:${engine.endpoint.port}/${upstreamPath}${qs ? `?${qs}` : ''}`;

  const headers = new Headers();
  request.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers.set(key, value);
  });
  headers.set('x-engine-token', engine.endpoint.token);

  const method = request.method.toUpperCase();
  const init: RequestInit & { duplex?: 'half' } = { method, headers, redirect: 'manual' };
  if (method !== 'GET' && method !== 'HEAD') {
    // buffer the request body (JSON or a control-image upload); the response
    // side is what needs to stream
    init.body = Buffer.from(await request.arrayBuffer());
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (e: any) {
    return NextResponse.json({ error: `engine unreachable: ${e?.message || e}` }, { status: 502 });
  }

  const responseHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) responseHeaders.set(key, value);
  });
  responseHeaders.set('Cache-Control', 'no-store');
  // keep intermediaries from buffering the frame stream
  responseHeaders.set('X-Accel-Buffering', 'no');
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

type Ctx = { params: Promise<{ path: string[] }> | { path: string[] } };

export async function GET(request: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function POST(request: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}

export async function DELETE(request: NextRequest, ctx: Ctx) {
  const { path } = await ctx.params;
  return proxy(request, path);
}
