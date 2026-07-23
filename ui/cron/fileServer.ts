/**
 * Front server for the UI: fast file downloads + reverse proxy + Next.js supervisor.
 *
 * Owns the single public port. GET/HEAD /api/files/* are served directly with
 * raw Node streams (full HTTP Range support, no Next.js per-request overhead);
 * everything else — including the dev-mode HMR WebSocket — is piped unmodified
 * to a Next.js server this process spawns on an ephemeral, loopback-only port.
 * The internal port is never fixed (no collisions) and is not reachable from
 * the network, so the externally visible surface is identical to running
 * `next start` alone: one port.
 *
 * With AI_TOOLKIT_FILE_SERVER_WORKERS > 1 (default: min(4, cores) in start
 * mode, 1 in dev mode), the front server runs as that many cluster workers
 * sharing the public socket, so file serving is parallel across CPU cores and
 * a burst of thumbnail requests never queues behind a multi-GB download.
 *
 * This process is optional. The equivalent (slower) route at
 * src/app/api/files/[...filePath]/route.ts still exists, so deployments that
 * run the worker and `next start` directly keep working exactly as before.
 *
 * Usage:
 *   node dist/cron/fileServer.js start --port 8675
 *   ts-node-dev cron/fileServer.ts dev --port 3000
 */
import cluster from 'cluster';
import { spawn } from 'child_process';
import http from 'http';
import net from 'net';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { pipeline } from 'stream';
import prisma from './prisma';
import { defaultDatasetsFolder, defaultTrainFolder, defaultDataRoot } from './paths';

const isDev = process.argv.includes('dev');

function argValue(name: string, fallback: number): number {
  const i = process.argv.indexOf(name);
  if (i !== -1 && process.argv[i + 1]) {
    const v = parseInt(process.argv[i + 1], 10);
    if (!isNaN(v)) return v;
  }
  return fallback;
}

const PUBLIC_PORT = argValue('--port', isDev ? 3000 : 8675);
const UPSTREAM_HOST = '127.0.0.1';

const numWorkers = (() => {
  const env = parseInt(process.env.AI_TOOLKIT_FILE_SERVER_WORKERS || '', 10);
  if (!isNaN(env) && env > 0) return env;
  return isDev ? 1 : Math.min(4, os.cpus().length);
})();

// ---------------------------------------------------------------------------
// File serving (same security model as the Next.js route). Settings can be
// changed at runtime from the UI in the Next.js process, so cache briefly
// instead of forever.
// ---------------------------------------------------------------------------
type Roots = { datasets: string; training: string; data: string };
let rootsCache: { roots: Roots; ts: number } | null = null;

async function getRoots(): Promise<Roots> {
  if (rootsCache && Date.now() - rootsCache.ts < 10_000) {
    return rootsCache.roots;
  }
  const rows = await prisma.settings.findMany({
    where: { key: { in: ['DATASETS_FOLDER', 'TRAINING_FOLDER', 'DATA_ROOT'] } },
  });
  const fromRow = (key: string, fallback: string) => {
    const row = rows.find(r => r.key === key);
    return row?.value && row.value !== '' ? row.value : fallback;
  };
  const roots: Roots = {
    datasets: fromRow('DATASETS_FOLDER', defaultDatasetsFolder),
    training: fromRow('TRAINING_FOLDER', defaultTrainFolder),
    data: fromRow('DATA_ROOT', defaultDataRoot),
  };
  rootsCache = { roots, ts: Date.now() };
  return roots;
}

const contentTypeMap: { [key: string]: string } = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.bmp': 'image/bmp',
  '.safetensors': 'application/octet-stream',
  '.zip': 'application/zip',
  // Videos
  '.mp4': 'video/mp4',
  '.avi': 'video/x-msvideo',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.wmv': 'video/x-ms-wmv',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  // Audio
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

async function serveFile(req: http.IncomingMessage, res: http.ServerResponse, prefix: string): Promise<void> {
  try {
    // /api/img/ serves media inline with ETag revalidation; /api/files/ is a
    // forced download. Both mirror their Next.js route's exact semantics.
    const isImg = prefix === '/api/img/';
    const urlPath = (req.url || '').split('?')[0];
    const rest = urlPath.slice(prefix.length);
    const decodedFilePath = rest
      .split('/')
      .map(decodeURIComponent)
      .join('/');

    const resolvedFilePath = path.resolve(decodedFilePath);
    const roots = await getRoots();
    const allowedDirs = isImg ? [roots.datasets, roots.training, roots.data] : [roots.datasets, roots.training];
    const isAllowed = allowedDirs.some(
      allowedDir => resolvedFilePath === allowedDir || resolvedFilePath.startsWith(allowedDir + path.sep),
    );
    if (!isAllowed) {
      console.warn(`Access denied: ${resolvedFilePath} not in ${allowedDirs.join(', ')}`);
      res.writeHead(403);
      res.end('Access denied');
      return;
    }

    let stat: fs.Stats;
    try {
      stat = await fs.promises.stat(resolvedFilePath);
    } catch {
      res.writeHead(404);
      res.end('File not found');
      return;
    }
    if (!stat.isFile()) {
      res.writeHead(400);
      res.end('Not a file');
      return;
    }

    const filename = path.basename(resolvedFilePath);
    const ext = path.extname(resolvedFilePath).toLowerCase();
    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    const headers: http.OutgoingHttpHeaders = {
      'Content-Type': contentType,
      'Accept-Ranges': 'bytes',
    };
    if (isImg) {
      // Weak ETag from inode/size/mtime — cheap and stable enough for
      // revalidation of thumbnails/videos the UI requests over and over.
      const etag = `W/"${stat.ino.toString(36)}-${stat.size.toString(36)}-${stat.mtimeMs.toString(36)}"`;
      headers['Cache-Control'] = 'public, max-age=86400, immutable';
      headers['ETag'] = etag;
      if (req.headers['if-none-match'] === etag) {
        res.writeHead(304, { ETag: etag, 'Cache-Control': headers['Cache-Control'] });
        res.end();
        return;
      }
    } else {
      headers['Cache-Control'] = 'public, max-age=86400';
      headers['Content-Disposition'] = `attachment; filename="${encodeURIComponent(filename)}"`;
      headers['X-Content-Type-Options'] = 'nosniff';
    }

    const range = req.headers.range;
    let start = 0;
    let end = stat.size - 1;
    let status = 200;

    if (range) {
      // Open-ended ranges (`bytes=0-`) serve to EOF — capping them forces
      // clients into serial re-requests and RTT-bound throughput.
      const parts = range.replace(/bytes=/, '').split('-');
      start = parseInt(parts[0], 10);
      end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      if (isNaN(start) || isNaN(end) || start > end || start >= stat.size) {
        res.writeHead(416, { 'Content-Range': `bytes */${stat.size}` });
        res.end();
        return;
      }
      status = 206;
      headers['Content-Range'] = `bytes ${start}-${end}/${stat.size}`;
    }

    headers['Content-Length'] = String(end - start + 1);
    res.writeHead(status, headers);
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    if (res.destroyed) {
      // Client disconnected while we were stat-ing; piping into a destroyed
      // stream throws.
      return;
    }

    const fileStream = fs.createReadStream(resolvedFilePath, {
      start,
      end,
      highWaterMark: 4 * 1024 * 1024, // large buffer to minimize per-chunk overhead
    });
    pipeline(fileStream, res, err => {
      // Client disconnects mid-download land here; just make sure the socket
      // is gone.
      if (err) res.destroy();
    });
  } catch (error) {
    console.error('Error serving file:', error);
    if (!res.headersSent) res.writeHead(500);
    res.end();
  }
}

// ---------------------------------------------------------------------------
// Passthrough proxy to the Next.js server
// ---------------------------------------------------------------------------
const upstreamAgent = new http.Agent({ keepAlive: true });

// Hop-by-hop headers must not be forwarded: Node de-chunks bodies on read and
// re-frames them on write, so forwarding e.g. `transfer-encoding` would
// corrupt the stream.
const HOP_BY_HOP = [
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
];

function stripHopByHop(headers: http.IncomingHttpHeaders): http.IncomingHttpHeaders {
  const out = { ...headers };
  for (const h of HOP_BY_HOP) delete out[h];
  return out;
}

function proxy(req: http.IncomingMessage, res: http.ServerResponse, upstreamPort: number, attempt = 0): void {
  const bodyless = req.method === 'GET' || req.method === 'HEAD';
  const upstreamReq = http.request(
    {
      host: UPSTREAM_HOST,
      port: upstreamPort,
      path: req.url,
      method: req.method,
      headers: stripHopByHop(req.headers),
      agent: upstreamAgent,
    },
    upstreamRes => {
      if (res.destroyed) {
        // Client disconnected while Next.js was working; drain and drop.
        upstreamRes.resume();
        return;
      }
      res.writeHead(upstreamRes.statusCode || 502, stripHopByHop(upstreamRes.headers));
      pipeline(upstreamRes, res, () => { });
    },
  );
  // If the client goes away, stop the upstream work too. No-op after normal
  // completion.
  const abortUpstream = () => upstreamReq.destroy();
  res.once('close', abortUpstream);
  upstreamReq.on('error', (err: NodeJS.ErrnoException) => {
    res.removeListener('close', abortUpstream);
    // Next.js is still booting. Nothing was sent on a refused connection, so
    // bodyless requests are safe to retry (request bodies can't be replayed).
    if (err.code === 'ECONNREFUSED' && bodyless && attempt < 120 && !res.destroyed) {
      setTimeout(() => proxy(req, res, upstreamPort, attempt + 1), 250);
      return;
    }
    if (res.destroyed) return;
    if (!res.headersSent) res.writeHead(502);
    res.end('UI server unavailable');
  });
  if (bodyless) {
    upstreamReq.end();
  } else {
    pipeline(req, upstreamReq, () => { });
  }
}

function startServer(publicPort: number, upstreamPort: number): void {
  const server = http.createServer((req, res) => {
    const urlPath = (req.url || '').split('?')[0];
    if (req.method === 'GET' || req.method === 'HEAD') {
      const prefix = ['/api/files/', '/api/img/'].find(p => urlPath.startsWith(p));
      if (prefix) {
        serveFile(req, res, prefix);
        return;
      }
    }
    proxy(req, res, upstreamPort);
  });

  // WebSocket upgrades (dev HMR) are forwarded as raw TCP with the original
  // request head replayed, so the hop-by-hop upgrade headers pass through
  // intact.
  server.on('upgrade', (req, socket, head) => {
    const upstream = net.connect(upstreamPort, UPSTREAM_HOST, () => {
      if (socket.destroyed) {
        upstream.destroy();
        return;
      }
      let raw = `${req.method} ${req.url} HTTP/1.1\r\n`;
      for (let i = 0; i < req.rawHeaders.length; i += 2) {
        raw += `${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}\r\n`;
      }
      raw += '\r\n';
      upstream.write(raw);
      if (head && head.length) upstream.write(head);
      upstream.pipe(socket);
      socket.pipe(upstream);
    });
    upstream.on('error', () => socket.destroy());
    socket.on('error', () => upstream.destroy());
  });

  // Uploads can be huge (100gb body limit upstream); the default 5-minute
  // request timeout would kill them mid-transfer.
  server.requestTimeout = 0;

  server.listen(publicPort);
}

// ---------------------------------------------------------------------------
// Supervisor: spawn Next.js on an ephemeral loopback port, then serve
// ---------------------------------------------------------------------------
function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.on('error', reject);
    probe.listen(0, UPSTREAM_HOST, () => {
      const port = (probe.address() as net.AddressInfo).port;
      probe.close(err => (err ? reject(err) : resolve(port)));
    });
  });
}

async function primaryMain(): Promise<void> {
  const upstreamPort = await getFreePort();

  const nextBin = require.resolve('next/dist/bin/next');
  const nextArgs = isDev ? ['dev', '--turbopack'] : ['start'];
  nextArgs.push('--port', String(upstreamPort), '--hostname', UPSTREAM_HOST);

  const nextChild = spawn(process.execPath, [nextBin, ...nextArgs], {
    stdio: ['inherit', 'pipe', 'pipe'],
    cwd: process.cwd(),
  });

  // Next.js prints its banner with the internal loopback port ("- Local:
  // http://127.0.0.1:39865"). Users who follow that link get the
  // non-accelerated server (or nothing, remotely). Rewrite every mention of
  // the internal address to the real public port before it reaches the
  // console.
  const internalAddr = new RegExp(`(https?://)?(127\\.0\\.0\\.1|localhost):${upstreamPort}`, 'g');
  const rewritePipe = (from: NodeJS.ReadableStream, to: NodeJS.WritableStream) => {
    let buffered = '';
    from.on('data', (chunk: Buffer) => {
      buffered += chunk.toString();
      const lines = buffered.split('\n');
      buffered = lines.pop() || '';
      for (const line of lines) {
        to.write(line.replace(internalAddr, `http://localhost:${PUBLIC_PORT}`) + '\n');
      }
    });
    from.on('end', () => {
      if (buffered) to.write(buffered.replace(internalAddr, `http://localhost:${PUBLIC_PORT}`) + '\n');
    });
  };
  rewritePipe(nextChild.stdout!, process.stdout);
  rewritePipe(nextChild.stderr!, process.stderr);

  let shuttingDown = false;
  const shutdown = (code: number) => {
    if (shuttingDown) return;
    shuttingDown = true;
    nextChild.kill('SIGTERM');
    for (const worker of Object.values(cluster.workers || {})) {
      worker?.kill();
    }
    process.exit(code);
  };

  nextChild.on('exit', code => {
    if (!shuttingDown) {
      console.error(`Next.js exited with code ${code}, shutting down file server`);
      shutdown(code ?? 1);
    }
  });
  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));
  // Never leave an orphaned Next.js child behind if we die unexpectedly.
  process.on('uncaughtException', err => {
    console.error('File server uncaught exception:', err);
    shutdown(1);
  });

  console.log(`AI Toolkit UI: http://localhost:${PUBLIC_PORT} (${numWorkers} file server worker${numWorkers === 1 ? '' : 's'})`);

  if (numWorkers <= 1) {
    startServer(PUBLIC_PORT, upstreamPort);
    return;
  }

  // AI_TOOLKIT_QUIET_PATHS keeps each forked worker from re-printing the
  // TOOLKIT_ROOT line from paths.ts.
  const workerEnv = {
    UPSTREAM_PORT: String(upstreamPort),
    PUBLIC_PORT: String(PUBLIC_PORT),
    AI_TOOLKIT_QUIET_PATHS: '1',
  };
  for (let i = 0; i < numWorkers; i++) {
    cluster.fork(workerEnv);
  }
  cluster.on('exit', worker => {
    if (!shuttingDown) {
      console.warn(`File server worker ${worker.id} died, restarting`);
      cluster.fork(workerEnv);
    }
  });
}

if (cluster.isPrimary) {
  primaryMain().catch(err => {
    console.error('File server failed to start:', err);
    process.exit(1);
  });
} else {
  process.on('disconnect', () => process.exit(0));
  startServer(parseInt(process.env.PUBLIC_PORT!, 10), parseInt(process.env.UPSTREAM_PORT!, 10));
}
