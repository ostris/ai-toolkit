#!/usr/bin/env node
/**
 * Free the AI Toolkit UI port before `npm start`.
 *
 * Stability Matrix's Linux stop path often leaves orphaned concurrently /
 * cluster workers holding :8675, which then causes EADDRINUSE crash loops on
 * the next launch. This clears listeners on the target port that belong to
 * this package (or any node process bound to it).
 */
const { execSync } = require('child_process');
const path = require('path');

const port = parseInt(process.argv[2] || '8675', 10);
if (!port || Number.isNaN(port)) {
  console.error('usage: ensure-port-free.js <port>');
  process.exit(1);
}

// Windows Process Job Objects / taskkill are handled by the launcher; this
// helper targets Linux orphan trees (ss + /proc) left after a hard stop.
if (process.platform === 'win32') {
  process.exit(0);
}

const toolkitRoot = path.resolve(__dirname, '..', '..');

function pidsOnPort(p) {
  try {
    const out = execSync(`ss -ltnp '( sport = :${p} )'`, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    const pids = new Set();
    for (const match of out.matchAll(/pid=(\d+)/g)) {
      pids.add(match[1]);
    }
    return [...pids];
  } catch {
    return [];
  }
}

function cmdline(pid) {
  try {
    return execSync(`ps -p ${pid} -o args=`, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return '';
  }
}

function cwdOf(pid) {
  try {
    return execSync(`readlink -f /proc/${pid}/cwd`, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return '';
  }
}

function belongsToToolkit(pid, args) {
  if (args.includes(toolkitRoot)) return true;
  const cwd = cwdOf(pid);
  if (cwd === toolkitRoot || cwd.startsWith(toolkitRoot + path.sep)) return true;
  // Workers often have a short argv (`node dist/cron/worker.js`) with cwd in ui/.
  if (
    /dist\/cron\/(worker|fileServer)\.js|next-server|concurrently/.test(args) &&
    (cwd.endsWith('/ui') || cwd.includes('/Packages/ai-toolkit') || cwd.includes('/ai-toolkit'))
  ) {
    return true;
  }
  // Orphans reparented to systemd may lose a readable cwd; still match the
  // canonical short argv this package uses.
  if (/(?:^|[\/\s])dist\/cron\/(worker|fileServer)\.js(?:\s|$)/.test(args)) {
    return true;
  }
  return false;
}

function shouldKill(pid) {
  const args = cmdline(pid);
  if (!args) return false;
  if (belongsToToolkit(pid, args)) return true;
  // Last resort: any node still bound to our UI port after a failed SM stop.
  return /node/.test(args);
}

const pids = pidsOnPort(port);
if (!pids.length) {
  process.exit(0);
}

const victims = pids.filter(shouldKill);
if (!victims.length) {
  console.error(
    `Port ${port} is in use by non-AI-Toolkit process(es): ${pids.join(', ')}. ` +
      'Stop that process or change the UI port.',
  );
  process.exit(1);
}

console.warn(`Freeing port ${port} (stopping leftover PIDs: ${victims.join(', ')})`);
try {
  execSync(`kill -TERM ${victims.join(' ')}`, { stdio: 'ignore' });
} catch {
  /* already gone */
}
const deadline = Date.now() + 3000;
while (Date.now() < deadline && pidsOnPort(port).some(p => victims.includes(p))) {
  try {
    execSync('sleep 0.1', { stdio: 'ignore' });
  } catch {
    break;
  }
}
const still = pidsOnPort(port).filter(p => victims.includes(p) || shouldKill(p));
if (still.length) {
  try {
    execSync(`kill -KILL ${still.join(' ')}`, { stdio: 'ignore' });
  } catch {
    /* already gone */
  }
}

// Also sweep orphaned workers that no longer hold the port but still thrash CPU.
try {
  const all = execSync('ps -eo pid=,args=', {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  const extra = [];
  for (const line of all.split('\n')) {
    const m = line.trim().match(/^(\d+)\s+(.*)$/);
    if (!m) continue;
    const [, pid, args] = m;
    if (String(process.pid) === pid || String(process.ppid) === pid) continue;
    if (!/dist\/cron\/(worker|fileServer)\.js|concurrently|next-server/.test(args)) {
      continue;
    }
    if (!belongsToToolkit(pid, args)) continue;
    extra.push(pid);
  }
  if (extra.length) {
    console.warn(`Stopping leftover AI Toolkit workers: ${extra.join(', ')}`);
    try {
      execSync(`kill -KILL ${extra.join(' ')}`, { stdio: 'ignore' });
    } catch {
      /* already gone */
    }
  }
} catch {
  /* ignore */
}

if (pidsOnPort(port).length) {
  console.error(`Port ${port} is still busy after cleanup.`);
  process.exit(1);
}
process.exit(0);
