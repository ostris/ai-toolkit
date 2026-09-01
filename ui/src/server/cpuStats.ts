import fs from 'fs';
import os from 'os';
import path from 'path';
import si from 'systeminformation';

/**
 * Spawn-free CPU load / memory / temperature readers for the monitor tick.
 *
 * Why this exists: `systeminformation` shells out for several of these
 * (`si.currentLoad()` runs `execSync('cat /proc/stat')` on Linux, `si.mem()`
 * runs PowerShell on Windows / vm_stat on macOS, `si.cpuTemperature()` runs
 * `sensors`). Every child process forks the calling process first, and on
 * Linux fork copies the parent's page tables — from a multi-GB Node heap that
 * is ~300ms with the event loop frozen. At the 500ms tick cadence that made
 * *every* request through the UI server wait behind the monitor.
 *
 * Everything on the tick path here reads from Node built-ins or plain files,
 * on every platform. `systeminformation` is kept only as a throttled fallback
 * for temperature where the OS offers no file-based source (Windows), and for
 * memory on macOS when macstats isn't available.
 */

const isLinux = os.platform() === 'linux';
const isMac = os.platform() === 'darwin';

// ---------------------------------------------------------------------------
// Load: os.cpus() cumulative times, differenced between calls. Same numbers
// systeminformation derives on every platform, minus its /proc/stat shell-out.
// ---------------------------------------------------------------------------
type CpuTimes = { busy: number; total: number };

function readCpuTimes(): CpuTimes {
  let busy = 0;
  let total = 0;
  for (const cpu of os.cpus()) {
    const t = cpu.times;
    const all = t.user + t.nice + t.sys + t.idle + t.irq;
    busy += all - t.idle;
    total += all;
  }
  return { busy, total };
}

/**
 * Returns a function that yields the CPU load percentage (0-100) accumulated
 * since the previous call. The first call reports the since-boot average
 * (there is nothing else to difference against), matching systeminformation.
 */
export function createLoadSampler(): () => number {
  let last: CpuTimes | null = null;
  let lastLoad = 0;
  return () => {
    const now = readCpuTimes();
    if (last) {
      const dTotal = now.total - last.total;
      const dBusy = now.busy - last.busy;
      // dTotal can be 0 if called twice within a jiffy; keep the last value.
      if (dTotal > 0) lastLoad = Math.min(100, Math.max(0, (dBusy / dTotal) * 100));
    } else if (now.total > 0) {
      lastLoad = (now.busy / now.total) * 100;
    }
    last = now;
    return lastLoad;
  };
}

// ---------------------------------------------------------------------------
// Memory (bytes). `available` is what the OS would hand to a new allocation
// without swapping (MemAvailable / ullAvailPhys); `free` is untouched pages.
// ---------------------------------------------------------------------------
export interface MemoryStats {
  total: number;
  free: number;
  available: number;
}

const memValue = (lines: string[], key: string): number => {
  const line = lines.find(l => l.startsWith(key + ':'));
  if (!line) return NaN;
  const value = parseInt(line.slice(key.length + 1).trim(), 10);
  return isNaN(value) ? NaN : value * 1024; // /proc/meminfo is in kB
};

export async function readMemory(): Promise<MemoryStats> {
  if (isLinux) {
    // Node's os.freemem() on Linux is MemAvailable on current libuv but MemFree
    // on older ones; reading /proc/meminfo gives both unambiguously.
    try {
      const lines = (await fs.promises.readFile('/proc/meminfo', 'utf8')).split('\n');
      const total = memValue(lines, 'MemTotal');
      const free = memValue(lines, 'MemFree');
      let available = memValue(lines, 'MemAvailable');
      if (isNaN(available)) {
        // Pre-3.14 kernels: same estimate systeminformation uses
        available = free + (memValue(lines, 'Buffers') || 0) + (memValue(lines, 'Cached') || 0);
      }
      if (!isNaN(total) && !isNaN(free)) {
        return { total, free, available: isNaN(available) ? free : available };
      }
    } catch {
      // fall through to os.*
    }
  } else if (isMac) {
    // os.freemem() on macOS is only the truly-free page count and badly
    // undercounts what's usable (inactive/purgeable pages). systeminformation
    // gets the real number from vm_stat; process creation on macOS is
    // posix_spawn (no page-table copy), so this is cheap there.
    try {
      const mem = await si.mem();
      return { total: mem.total, free: mem.free, available: mem.available };
    } catch {
      // fall through to os.*
    }
  }
  // Windows (and anything else): os.freemem() is GlobalMemoryStatusEx
  // ullAvailPhys, i.e. exactly "available", which is also what
  // systeminformation reports for both free and available there.
  const total = os.totalmem();
  const free = os.freemem();
  return { total, free, available: free };
}

// ---------------------------------------------------------------------------
// Temperature (°C, or null if unknown)
// ---------------------------------------------------------------------------

// Resolved once: which sysfs file(s) to read, and how to combine them.
type LinuxTempSource = { files: string[]; mode: 'single' | 'mean' };
let linuxTempSource: LinuxTempSource | null | undefined; // undefined = not scanned yet

const millideg = (raw: string): number | null => {
  const v = parseInt(raw, 10);
  return isNaN(v) ? null : Math.round(v / 100) / 10;
};

/**
 * Walks /sys/class/hwmon looking for the same labels systeminformation picks
 * out of `sensors` (Tctl / Tdie on AMD, "Package id N" / "Physical id N" on
 * Intel, else the mean of the Core N readings), then falls back to a CPU-ish
 * thermal zone. Returns null when nothing looks like a CPU sensor.
 */
async function scanLinuxTempSource(): Promise<LinuxTempSource | null> {
  const hwmonRoot = '/sys/class/hwmon';
  const monitors = await fs.promises.readdir(hwmonRoot).catch(() => [] as string[]);
  const labeled: { label: string; file: string }[] = [];
  for (const mon of monitors) {
    const dir = path.join(hwmonRoot, mon);
    const files = await fs.promises.readdir(dir).catch(() => [] as string[]);
    for (const file of files) {
      if (!/^temp\d+_label$/.test(file)) continue;
      const label = (await fs.promises.readFile(path.join(dir, file), 'utf8').catch(() => '')).trim();
      if (!label) continue;
      labeled.push({ label, file: path.join(dir, file.replace(/_label$/, '_input')) });
    }
  }
  const byLabel = (pred: (l: string) => boolean) => labeled.filter(e => pred(e.label.toLowerCase()));
  const tctl = byLabel(l => l === 'tctl');
  if (tctl.length) return { files: [tctl[0].file], mode: 'single' };
  const tdie = byLabel(l => l === 'tdie');
  if (tdie.length) return { files: [tdie[0].file], mode: 'single' };
  const pkg = byLabel(l => l.includes('package') || l.includes('physical') || l === 'tccd1');
  if (pkg.length) return { files: [pkg[0].file], mode: 'single' };
  const cores = byLabel(l => l.startsWith('core'));
  if (cores.length) return { files: cores.map(c => c.file), mode: 'mean' };

  // No labeled hwmon sensor — try the generic thermal zones (ARM SBCs, some
  // laptops expose only these).
  const thermalRoot = '/sys/class/thermal';
  const zones = (await fs.promises.readdir(thermalRoot).catch(() => [] as string[])).filter(z =>
    z.startsWith('thermal_zone'),
  );
  for (const zone of zones) {
    const type = (await fs.promises.readFile(path.join(thermalRoot, zone, 'type'), 'utf8').catch(() => ''))
      .trim()
      .toLowerCase();
    if (type === 'x86_pkg_temp' || type.includes('cpu') || type.includes('soc')) {
      return { files: [path.join(thermalRoot, zone, 'temp')], mode: 'single' };
    }
  }
  return null;
}

async function readLinuxTemp(): Promise<number | null> {
  if (linuxTempSource === undefined) {
    linuxTempSource = await scanLinuxTempSource();
  }
  const source = linuxTempSource;
  if (!source) return null;
  const values: number[] = [];
  for (const file of source.files) {
    const raw = await fs.promises.readFile(file, 'utf8').catch(() => null);
    if (raw === null) {
      // Sensor went away (module unloaded / renumbered) — rescan next time.
      linuxTempSource = undefined;
      return null;
    }
    const v = millideg(raw);
    if (v !== null) values.push(v);
  }
  if (values.length === 0) return null;
  if (source.mode === 'mean') return Math.round(values.reduce((a, b) => a + b, 0) / values.length);
  return values[0];
}

/**
 * On Linux this is a couple of sysfs reads. Elsewhere it defers to
 * systeminformation (osx-temperature-sensor on macOS — an in-process native
 * module; PowerShell/WMI on Windows, where process creation doesn't fork and
 * so doesn't scale with our heap). Callers throttle this; it is not meant to
 * run every tick.
 */
export async function readCpuTemperature(): Promise<number | null> {
  if (isLinux) {
    const t = await readLinuxTemp();
    if (t !== null) return t;
    // A known sensor that failed to read: rescan on the next call, don't
    // shell out. Only when sysfs has nothing at all is `sensors` (via
    // systeminformation) worth a try.
    if (linuxTempSource !== null) return null;
  }
  try {
    const t = await si.cpuTemperature();
    return typeof t.main === 'number' && !isNaN(t.main) ? t.main : null;
  } catch {
    return null;
  }
}
