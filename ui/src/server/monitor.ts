import { spawn, execFile, ChildProcess } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import si from 'systeminformation';
import { loadMacstats } from '@/server/macstats';
import path from 'path';
import fs from 'fs';
import { createLoadSampler, readCpuTemperature, readMemory } from '@/server/cpuStats';
import { CpuInfo, GpuInfo, GPUApiResponse, MonitorHistoryPoint, MonitorInit, MonitorSample } from '@/types';
import { historyPointFromSample, MONITOR_HISTORY_LENGTH, MONITOR_TICK_MS } from '@/utils/monitorSample';

const execFileAsync = promisify(execFile);
const execAsync = promisify(require('child_process').exec);

/**
 * Always-on system monitor. Samples CPU + GPU every MONITOR_TICK_MS, keeps a
 * 2-minute rolling history (load + memory only), and pushes every full sample
 * to subscribers (the SSE route at /api/monitor).
 *
 * GPU stats come from a single resident `nvidia-smi ... -lms` child process —
 * NVML stays initialized between samples, which is what made the old
 * spawn-per-request /api/gpu route slow. If loop mode never produces output
 * (or nvidia-smi keeps hanging), we fall back to a one-shot spawn per tick,
 * which matches the old route's behavior exactly.
 */

const NV_QUERY_ARGS = [
  '--query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed',
  '--format=csv,noheader,nounits',
];
const NV_ENV = { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' };
// No stdout line for this long means the loop child is buffering or hung.
const NV_WATCHDOG_MS = 15_000;
// A line-less gap this long after the last line closes out a batch (all
// lines of one iteration arrive together; iterations are MONITOR_TICK_MS
// apart, so this can never bleed into the next batch).
const NV_BATCH_FLUSH_MS = 100;
// Temperature refresh is decoupled from the tick (see refreshCpuTemp)
const CPU_TEMP_REFRESH_MS = 5000;


// Helper function to get venv PATH
function getVenvPath(): NodeJS.ProcessEnv {
  const projectRoot = path.resolve(process.cwd(), '..');
  let env: NodeJS.ProcessEnv = { ...process.env };
  const venvPaths = [
    path.join(projectRoot, '.venv', 'bin'),
    path.join(projectRoot, 'venv', 'bin'),
  ];
  for (const venvBin of venvPaths) {
    if (fs.existsSync(venvBin)) {
      const currentPath = process.env.PATH || '';
      const pathSeparator = process.platform === 'win32' ? ';' : ':';
      env.PATH = `${venvBin}${pathSeparator}${currentPath}`;
      break;
    }
  }
  return env;
}

function parseValue(value: string | undefined, defaultValue: number = 0): number {
  if (!value || value === 'N/A' || value.trim() === '') {
    return defaultValue;
  }
  const parsed = parseFloat(value);
  return isNaN(parsed) ? defaultValue : parsed;
}

async function getAmdSmiGpuStats(): Promise<GpuInfo[]> {
  const env = getVenvPath();
  const { stdout: listStdout } = await execAsync('amd-smi list --json', { env } as any);
  let gpuList: Array<{ gpu: number }> = [];
  try {
    const listData = JSON.parse(listStdout);
    if (Array.isArray(listData)) {
      gpuList = listData;
    } else if (listData && Array.isArray((listData as any).gpu_data)) {
      gpuList = (listData as any).gpu_data;
    }
  } catch {
    return [];
  }
  if (gpuList.length === 0) return [];
  const gpus = await Promise.all(
    gpuList.map(async (gpuInfo) => {
      const gpuId = gpuInfo.gpu;
      const metricCommand = `amd-smi metric --gpu ${gpuId} --csv`;
      try {
        const { stdout: metricStdout } = await execAsync(metricCommand, { env } as any);
        const lines = metricStdout.trim().split('\n').filter(line => line.trim().length > 0);
        if (lines.length < 2) return null;
        // Use robust CSV parsing for amd-smi (fields may contain quoted arrays with commas)
        function parseCSVLine(line: string): string[] {
          const fields: string[] = [];
          let current = '';
          let inQuotes = false;
          for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') inQuotes = !inQuotes;
            else if (char === ',' && !inQuotes) { fields.push(current.trim()); current = ''; }
            else current += char;
          }
          fields.push(current.trim());
          return fields;
        }
        const header = parseCSVLine(lines[0]);
        const dataLine = parseCSVLine(lines[1]);
        const getFieldIndex = (fieldNames: string | string[]): number => {
          const names = Array.isArray(fieldNames) ? fieldNames : [fieldNames];
          for (const name of names) {
            const idx = header.findIndex(h => h.toLowerCase().includes(name.toLowerCase()));
            if (idx >= 0) return idx;
          }
          return -1;
        };
        const gpuIndex = getFieldIndex('gpu');
        const usageIndex = getFieldIndex(['usage', 'gpu_use', 'utilization', 'gfx_activity']);
        const edgeIndex = getFieldIndex(['edge', 'temperature', 'temp', 'junction']);
        const totalVramIndex = getFieldIndex(['total_vram', 'vram_total', 'memory_total']);
        const usedVramIndex = getFieldIndex(['used_vram', 'vram_used', 'memory_used']);
        const freeVramIndex = getFieldIndex(['free_vram', 'vram_free', 'memory_free']);
        const socketPowerIndex = getFieldIndex(['socket_power', 'power', 'power_draw', 'tdp']);
        const fanMaxIndex = getFieldIndex(['fan_max', 'fan_speed', 'max', 'fan_percent']);
        let gfxClkIndex = -1;
        for (let i = 0; i < header.length; i++) {
          if (header[i].toLowerCase().startsWith('gfx_') && header[i].toLowerCase().endsWith('_clk')) { gfxClkIndex = i; break; }
        }
        const memClkIndex = getFieldIndex('mem_0_clk');
        const index = gpuIndex >= 0 ? parseInt(dataLine[gpuIndex] || '0') || gpuId : gpuId;
        const usage = usageIndex >= 0 ? parseValue(dataLine[usageIndex]) : 0;
        const temperature = edgeIndex >= 0 ? parseValue(dataLine[edgeIndex]) : 0;
        const memoryTotalMB = totalVramIndex >= 0 ? parseValue(dataLine[totalVramIndex]) : 0;
        const memoryUsedMB = usedVramIndex >= 0 ? parseValue(dataLine[usedVramIndex]) : 0;
        const memoryFreeMB = freeVramIndex >= 0 ? parseValue(dataLine[freeVramIndex]) : (memoryTotalMB - memoryUsedMB);
        const powerDraw = socketPowerIndex >= 0 ? parseValue(dataLine[socketPowerIndex]) : 0;
        const clockGraphics = gfxClkIndex >= 0 ? parseValue(dataLine[gfxClkIndex]) : 0;
        const clockMemory = memClkIndex >= 0 ? parseValue(dataLine[memClkIndex]) : 0;
        let fanSpeed = 0;
        if (fanMaxIndex >= 0 && dataLine[fanMaxIndex] && dataLine[fanMaxIndex] !== 'N/A') fanSpeed = parseValue(dataLine[fanMaxIndex]);
        let name = `AMD GPU ${index}`;
        try {
          const staticCommand = `amd-smi static --gpu ${gpuId} --json`;
          const { stdout: staticStdout } = await execAsync(staticCommand, { env } as any);
          const staticData = JSON.parse(staticStdout);
          if (staticData && (staticData as any).gpu_data && (staticData as any).gpu_data[0]) {
            const gpuData = (staticData as any).gpu_data[0];
            if (gpuData.asic && gpuData.asic.market_name) name = gpuData.asic.market_name;
            else if (gpuData.asic && gpuData.asic.name) name = gpuData.asic.name;
            else if (gpuData.name) name = gpuData.name;
          }
        } catch {}
        const memoryUtilPercent = memoryTotalMB > 0 ? Math.max(0, Math.min(100, Math.round((memoryUsedMB / memoryTotalMB) * 100))) : 0;
        const validTemperature = temperature >= 0 && temperature <= 200 ? temperature : 0;
        const validUsage = Math.max(0, Math.min(100, usage));
        const hasBasicData = validTemperature > 0 || memoryTotalMB > 0;
        const hasPerformanceData = validUsage > 0 || powerDraw > 0 || clockGraphics > 0 || clockMemory > 0;
        const hasSufficientData = hasBasicData && hasPerformanceData;
        return {
          index, name, driverVersion: 'ROCm',
          temperature: validTemperature > 0 ? Math.round(validTemperature) : 0,
          utilization: { gpu: Math.round(validUsage), memory: memoryUtilPercent },
          memory: { total: Math.max(0, Math.round(memoryTotalMB)), free: Math.max(0, Math.round(memoryFreeMB)), used: Math.max(0, Math.round(memoryUsedMB)) },
          power: { draw: Math.max(0, powerDraw), limit: 0 },
          clocks: { graphics: Math.max(0, Math.round(clockGraphics)), memory: Math.max(0, Math.round(clockMemory)) },
          fan: { speed: Math.max(0, Math.min(100, fanSpeed)) },
          _hasSufficientData: hasSufficientData,
        } as any;
      } catch { return null; }
    })
  );
  const validGpus = gpus.filter((gpu): gpu is NonNullable<typeof gpu> => gpu !== null);
  const hasAnyData = validGpus.some((gpu: any) => gpu._hasSufficientData);
  if (!hasAnyData && validGpus.length > 0) return [];
  return validGpus.map(({ _hasSufficientData, ...gpu }: any) => gpu);
}

async function getRocmGpuStats(): Promise<GpuInfo[]> {
  const env = getVenvPath();
  const command = 'rocm-smi --showid --showproductname --showtemp --showuse --showmemuse --showmeminfo vram --showpower --showclocks --csv';
  const { stdout } = await execAsync(command, { env } as any);
  const lines = stdout.split('\n').map(line => line.trim()).filter(line => line.length > 0 && !line.startsWith('Exception') && !line.startsWith('Error'));
  const headerIndex = lines.findIndex(line => line.startsWith('device,'));
  if (headerIndex === -1 || lines.length < headerIndex + 2) return [];
  function parseCSVLine(line: string): string[] {
    const fields: string[] = []; let current = ''; let inQuotes = false;
    for (let i = 0; i < line.length; i++) { const char = line[i]; if (char === '"') inQuotes = !inQuotes; else if (char === ',' && !inQuotes) { fields.push(current.trim()); current = ''; } else current += char; }
    fields.push(current.trim()); return fields;
  }
  const headerFields = parseCSVLine(lines[headerIndex]);
  function findHeaderIndex(namePatterns: string[]): number {
    for (let i = 0; i < headerFields.length; i++) { const h = headerFields[i].toLowerCase().trim(); for (const p of namePatterns) if (h.includes(p.toLowerCase())) return i; }
    return -1;
  }
  const tempFieldIdx = findHeaderIndex(['temperature', '(c)', 'temp']);
  const mclkFieldIdx = findHeaderIndex(['mclk clock speed', 'mclk']);
  const sclkFieldIdx = findHeaderIndex(['sclk clock speed', 'sclk']);
  const powerFieldIdx = findHeaderIndex(['power (w)', 'power']);
  const usageFieldIdx = findHeaderIndex(['gpu use', 'gpu_use', 'gpu use (%)']);
  const memTotalFieldIdx = findHeaderIndex(['vram total memory', 'vram total']);
  const memUsedFieldIdx = findHeaderIndex(['vram total used memory', 'vram total used', 'vram used']);
  const cardSkuFieldIdx = findHeaderIndex(['card sku', 'sku']);
  const cardModelFieldIdx = findHeaderIndex(['card model', 'model']);
  const deviceIdFieldIdx = findHeaderIndex(['device id', 'gpu id']);
  const cardVendorFieldIdx = findHeaderIndex(['card vendor', 'vendor']);
  const gpus = lines.slice(headerIndex + 1).map((line, idx) => {
    const fields = parseCSVLine(line);
    const deviceName = fields[0]?.trim() || '';
    const deviceMatch = deviceName.match(/\d+/);
    const index = deviceMatch ? parseInt(deviceMatch[0]) : idx;
    const tempStr = fields[tempFieldIdx]?.trim() || '';
    let temperature = 0;
    if (tempStr && tempStr !== 'N/A' && !isNaN(parseFloat(tempStr))) { const v = parseFloat(tempStr); if (v >= 0 && v <= 200) temperature = v; }
    const gpuUtilStr = usageFieldIdx >= 0 ? (fields[usageFieldIdx]?.trim() || '0') : '0';
    let gpuUtil = 0; if (gpuUtilStr && gpuUtilStr !== 'N/A' && !isNaN(parseFloat(gpuUtilStr))) { const p = parseFloat(gpuUtilStr); if (p >= 0 && p <= 100) gpuUtil = p; }
    gpuUtil = Math.max(0, Math.min(100, gpuUtil));
    let memoryTotal = parseFloat(fields[memTotalFieldIdx]?.trim() || '0') || 0;
    let memoryUsed = parseFloat(fields[memUsedFieldIdx]?.trim() || '0') || 0;
    if (memoryTotal < 0 || isNaN(memoryTotal)) memoryTotal = 0;
    if (memoryUsed < 0 || isNaN(memoryUsed)) memoryUsed = 0;
    if (memoryUsed > memoryTotal) memoryUsed = memoryTotal;
    const memoryFree = Math.max(0, memoryTotal - memoryUsed);
    const powerDrawStr = fields[powerFieldIdx]?.trim() || '';
    let powerDraw = 0;
    if (powerDrawStr && !powerDrawStr.toLowerCase().includes('mhz') && powerDrawStr !== 'N/A') { const m = powerDrawStr.match(/(\d+\.?\d*)/); if (m) { const p = parseFloat(m[1]); if (p >= 0 && p <= 1000) powerDraw = p; } }
    let clockGraphics = 0, clockMemory = 0;
    const mclkStr = fields[mclkFieldIdx]?.trim() || ''; const sclkStr = fields[sclkFieldIdx]?.trim() || '';
    const mclkMatch = mclkStr.match(/(\d+)/); const sclkMatch = sclkStr.match(/(\d+)/);
    if (mclkMatch) clockMemory = parseInt(mclkMatch[1]); if (sclkMatch) clockGraphics = parseInt(sclkMatch[1]);
    if (clockGraphics > 10000) clockGraphics = Math.round(clockGraphics / 1_000_000);
    if (clockMemory > 10000) clockMemory = Math.round(clockMemory / 1_000_000);
    if (clockGraphics > 5000 || clockGraphics < 0) clockGraphics = 0;
    if (clockMemory > 3000 || clockMemory < 0) clockMemory = 0;
    const cardSku = fields[cardSkuFieldIdx]?.trim() || ''; const cardModel = fields[cardModelFieldIdx]?.trim() || '';
    const cardVendor = fields[cardVendorFieldIdx]?.trim() || ''; const gpuId = fields[deviceIdFieldIdx]?.trim() || '';
    let name = '';
    if (cardSku && !cardSku.startsWith('0x') && !/^\d+$/.test(cardSku) && cardSku !== gpuId && cardSku !== memoryTotal.toString() && cardSku !== memoryUsed.toString()) name = cardSku;
    else if (cardModel && cardModel !== gpuId && !cardModel.startsWith('0x') && cardModel !== memoryTotal.toString()) name = cardModel;
    else if (cardVendor && (cardVendor.includes('AMD') || cardVendor.includes('Advanced Micro Devices'))) name = `AMD GPU ${index}`;
    else name = `GPU ${index}`;
    if (name === memoryTotal.toString() || name === memoryUsed.toString() || /^\d+$/.test(name)) name = `AMD GPU ${index}`;
    let memoryTotalMB = 0, memoryUsedMB = 0, memoryFreeMB = 0;
    if (memoryTotal > 0) {
      if (memoryTotal > 1000000) { memoryTotalMB = Math.round(memoryTotal / (1024*1024)); memoryUsedMB = Math.round(memoryUsed / (1024*1024)); memoryFreeMB = Math.round(memoryFree / (1024*1024)); }
      else if (memoryTotal >= 10) { memoryTotalMB = Math.round(memoryTotal); memoryUsedMB = Math.round(memoryUsed); memoryFreeMB = Math.round(memoryFree); }
      else { memoryTotalMB = Math.round(memoryTotal*1024); memoryUsedMB = Math.round(memoryUsed*1024); memoryFreeMB = Math.round(memoryFree*1024); }
    }
    const memoryUtilPercent = memoryTotalMB > 0 ? Math.max(0, Math.min(100, Math.round((memoryUsedMB / memoryTotalMB)*100))) : 0;
    return { index: isNaN(index) ? idx : index, name, driverVersion: 'ROCm', temperature: temperature>0?Math.round(temperature):0, utilization: { gpu: Math.round(gpuUtil), memory: memoryUtilPercent }, memory: { total: memoryTotalMB, free: memoryFreeMB, used: memoryUsedMB }, power: { draw: Math.max(0,powerDraw), limit: 0 }, clocks: { graphics: Math.max(0,clockGraphics), memory: Math.max(0,clockMemory) }, fan: { speed: 0 } };
  });
  return gpus;
}

function parseGpuLine(line: string): GpuInfo | null {
  const [
    index,
    name,
    driverVersion,
    temperature,
    gpuUtil,
    memoryUtil,
    memoryTotal,
    memoryFree,
    memoryUsed,
    powerDraw,
    powerLimit,
    clockGraphics,
    clockMemory,
    fanSpeed,
  ] = line.split(', ').map(item => item.trim());
  if (isNaN(parseInt(index))) return null;
  return {
    index: parseInt(index),
    name,
    driverVersion,
    temperature: parseInt(temperature),
    utilization: {
      gpu: parseInt(gpuUtil),
      memory: parseInt(memoryUtil),
    },
    memory: {
      total: parseInt(memoryTotal),
      free: parseInt(memoryFree),
      used: parseInt(memoryUsed),
    },
    power: {
      draw: parseFloat(powerDraw),
      limit: parseFloat(powerLimit),
    },
    clocks: {
      graphics: parseInt(clockGraphics),
      memory: parseInt(clockMemory),
    },
    fan: {
      speed: parseInt(fanSpeed) || 0, // Some GPUs might not report fan speed, default to 0
    },
  };
}

// Serialized once per tick and shared, so N subscribers don't stringify N times
type Subscriber = (sample: MonitorSample, serialized: string) => void;

class SystemMonitor {
  private isMac = os.platform() === 'darwin';
  private history: MonitorHistoryPoint[] = [];
  private subscribers = new Set<Subscriber>();
  private latestCpu: CpuInfo | null = null;
  private latestGpu: GPUApiResponse = { hasNvidiaSmi: false, isMac: this.isMac, gpus: [] };
  private macGpuName = 'Apple GPU';
  private backend: 'nvidia' | 'amd' | 'rocm' | 'none' | null = null;
  private amdUnavailable = false;
  private rocmUnavailable = false;
  private amdInFlight = false;
  private rocmInFlight = false;
  private nvChild: ChildProcess | null = null;
  private nvBatch: GpuInfo[] = [];
  private nvStdoutBuffer = '';
  private nvFlushTimer: NodeJS.Timeout | null = null;
  private nvUnavailable = false;
  private nvReaped = false;
  private nvEverGotLine = false;
  private nvOneShotMode = false;
  private nvOneShotInFlight = false;
  private lastNvLineAt = 0;
  private tickInFlight = false;
  private lastCpuTemp = 0;
  private cpuTempInFlight = false;
  private lastCpuTempAt = 0;
  private cpuStatic: { name: string; cores: number } | null = null;
  // Differences os.cpus() times between ticks; no child process involved.
  private sampleLoad = createLoadSampler();

  start(): void {
    if (this.isMac) {
      this.initMacGpuName();
    } else {
      this.detectBackend().then(backend => {
        this.backend = backend;
        if (backend === 'nvidia') {
          this.startNvLoop();
        }
        // amd/rocm use one-shot per tick, no loop needed
      });
    }
    // Never leave a resident nvidia-smi behind.
    process.once('exit', () => {
      try {
        this.nvChild?.kill('SIGKILL');
      } catch {
        // already gone
      }
    });
    // Fixed cadence; a tick that overruns the interval (slow temperature
    // read, hung nvidia-smi one-shot) just skips beats instead of stacking.
    this.tick();
    setInterval(() => this.tick(), MONITOR_TICK_MS);
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  getInit(): MonitorInit {
    return {
      t: Date.now(),
      cpu: this.latestCpu,
      gpu: this.latestGpu,
      history: [...this.history],
    };
  }

  // -------------------------------------------------------------------------
  // Tick loop
  // -------------------------------------------------------------------------
  private async tick(): Promise<void> {
    if (this.tickInFlight) return;
    this.tickInFlight = true;
    try {
      await this.doTick();
    } finally {
      this.tickInFlight = false;
    }
  }

  private async doTick(): Promise<void> {
    const t = Date.now();
    try {
      this.latestCpu = await this.sampleCpu();
    } catch (error) {
      console.error('Monitor: CPU sample failed:', error);
    }
    try {
      if (this.isMac) {
        this.latestGpu = this.sampleMacGpu();
      } else if (this.backend === 'amd') {
        await this.sampleAmdOneShot();
      } else if (this.backend === 'rocm') {
        await this.sampleRocmOneShot();
      } else if (this.backend === 'nvidia') {
        if (this.nvOneShotMode) {
          await this.sampleNvOneShot();
        } else {
          this.nvWatchdog();
          // If loop hasn't produced data yet, also try one-shot as fallback until loop is ready
          if (this.latestGpu.gpus.length === 0 && !this.nvEverGotLine) {
            await this.sampleNvOneShot();
            if (this.latestGpu.gpus.length === 0 && this.nvUnavailable) {
              // nvidia failed, re-detect as amd/rocm
              this.backend = await this.detectBackend(true);
            }
          }
        }
      } else if (this.backend === 'none' || this.backend === null) {
        // First tick detection if start() hasn't completed
        this.backend = await this.detectBackend();
        if (this.backend === 'amd') await this.sampleAmdOneShot();
        else if (this.backend === 'rocm') await this.sampleRocmOneShot();
        else if (this.backend === 'nvidia') {
          await this.sampleNvOneShot();
        } else {
          this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: false, hasRocmSmi: false, isMac: false, gpus: [], error: 'No GPU tools found (nvidia-smi, amd-smi, rocm-smi)' };
        }
      }
    } catch (error) {
      console.error('Monitor: GPU sample failed:', error);
    }

    const sample: MonitorSample = { t, cpu: this.latestCpu, gpu: this.latestGpu };
    this.history.push(historyPointFromSample(sample));
    if (this.history.length > MONITOR_HISTORY_LENGTH) {
      this.history.splice(0, this.history.length - MONITOR_HISTORY_LENGTH);
    }
    const serialized = JSON.stringify(sample);
    for (const subscriber of this.subscribers) {
      try {
        subscriber(sample, serialized);
      } catch (error) {
        console.error('Monitor: subscriber failed:', error);
      }
    }
  }

  // -------------------------------------------------------------------------
  // CPU (mirrors /api/cpu exactly)
  // -------------------------------------------------------------------------
  private async sampleCpu(): Promise<CpuInfo> {
    // si.cpu() shells out to lscpu/dmidecode on every call — at tick cadence
    // that alone was eating a measurable slice of a core. Name and core count
    // never change, so resolve them exactly once.
    if (!this.cpuStatic) {
      const cpuInfoRaw = await si.cpu();
      this.cpuStatic = { name: `${cpuInfoRaw.manufacturer} ${cpuInfoRaw.brand}`, cores: cpuInfoRaw.cores };
    }
    const cpuStatic = this.cpuStatic;

    if (this.isMac) {
      try {
        const ms = loadMacstats();
        if (!ms) throw new Error('macstats unavailable');
        const ramData = ms.getRAMUsageSync();
        const cpuData = ms.getCpuDataSync();
        return {
          name: cpuStatic.name,
          cores: cpuStatic.cores,
          temperature: cpuData.temperature || 0,
          totalMemory: ramData.total / (1024 * 1024),
          availableMemory: ramData.free / (1024 * 1024),
          freeMemory: ramData.free / (1024 * 1024),
          currentLoad: this.sampleLoad(),
        };
      } catch {
        // Fallback to the generic path if macstats fails
      }
    }

    // The temperature read can take >1s on some machines and would make the
    // tick skip beats, so it refreshes concurrently and we use the cached
    // value (at most a tick or two old).
    this.refreshCpuTemp();
    // Nothing below spawns a process. Every child process forks this server
    // first, and forking a multi-GB Node heap freezes the event loop for
    // hundreds of ms — at tick cadence that stalled every request in the UI.
    const memoryData = await readMemory();
    return {
      name: cpuStatic.name,
      cores: cpuStatic.cores,
      temperature: this.lastCpuTemp,
      totalMemory: memoryData.total / (1024 * 1024),
      availableMemory: memoryData.available / (1024 * 1024),
      freeMemory: memoryData.free / (1024 * 1024),
      currentLoad: this.sampleLoad(),
    };
  }

  private refreshCpuTemp(): void {
    // On Linux this is a sysfs read; elsewhere it may still shell out (via
    // systeminformation), and e.g. `sensors` can take ~1s of slow SMBus
    // polling — refreshed back-to-back it becomes a permanently resident
    // child. Every few seconds is plenty for a temperature.
    if (this.cpuTempInFlight || Date.now() - this.lastCpuTempAt < CPU_TEMP_REFRESH_MS) return;
    this.lastCpuTempAt = Date.now();
    this.cpuTempInFlight = true;
    readCpuTemperature()
      .then(t => {
        this.lastCpuTemp = t ?? 0;
      })
      .catch(() => {
        // keep the previous value
      })
      .finally(() => {
        this.cpuTempInFlight = false;
      });
  }

  // -------------------------------------------------------------------------
  // Mac GPU (mirrors /api/gpu's mac path)
  // -------------------------------------------------------------------------
  private initMacGpuName(): void {
    execFileAsync('sh', ['-c', 'system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset Model|Total Number of Cores"'], {
      timeout: 5000,
    })
      .then(({ stdout }) => {
        const nameMatch = stdout.match(/Chipset Model:\s*(.+)/);
        const coresMatch = stdout.match(/Total Number of Cores:\s*(\d+)/);
        if (nameMatch) {
          this.macGpuName = nameMatch[1].trim();
          if (coresMatch) {
            this.macGpuName += ` GPU (${coresMatch[1]} cores)`;
          }
        }
      })
      .catch(() => {
        // fallback to generic name
      });
  }

  private sampleMacGpu(): GPUApiResponse {
    let temperature = 0;
    let gpuLoad = 0;
    let fanSpeed = 0;
    let powerDraw = 0;
    let memUsed = 0;
    let memTotal = os.totalmem() / (1024 * 1024);

    const ms = loadMacstats();
    if (ms) {
      try {
        const gpuData = ms.getGpuDataSync();
        temperature = gpuData.temperature || 0;
        gpuLoad = gpuData.usage || 0;
      } catch {
        // ignore
      }
      try {
        const fanData = ms.getFanDataSync();
        const fanKeys = Object.keys(fanData);
        if (fanKeys.length > 0) {
          fanSpeed = fanData[fanKeys[0]].rpm || 0;
        }
      } catch {
        // ignore
      }
      try {
        const powerData = ms.getPowerDataSync();
        powerDraw = powerData.gpu || 0;
      } catch {
        // ignore
      }
      try {
        const ramData = ms.getRAMUsageSync();
        memUsed = ramData.used / (1024 * 1024);
        memTotal = ramData.total / (1024 * 1024);
      } catch {
        // ignore
      }
    }

    return {
      hasNvidiaSmi: false,
      isMac: true,
      gpus: [
        {
          index: 0,
          name: this.macGpuName,
          driverVersion: 'macOS',
          temperature: Math.round(temperature),
          utilization: {
            gpu: gpuLoad,
            memory: memTotal > 0 ? Math.round((memUsed / memTotal) * 100) : 0,
          },
          memory: {
            total: Math.round(memTotal),
            free: Math.round(memTotal - memUsed),
            used: Math.round(memUsed),
          },
          power: { draw: powerDraw, limit: 0 },
          clocks: { graphics: 0, memory: 0 },
          fan: { speed: fanSpeed },
        },
      ],
    };
  }

  // -------------------------------------------------------------------------
  // NVIDIA loop-mode child
  // -------------------------------------------------------------------------
  private startNvLoop(): void {
    if (this.nvUnavailable || this.nvOneShotMode || this.nvChild) return;

    // A hard-killed server (SIGKILL never runs the exit hook) can orphan the
    // resident loop child. Our exact query string only ever appears in
    // children we spawned, so reap any stray once at boot — after it
    // completes, to not race the kill against our own fresh child.
    if (!this.nvReaped && process.platform !== 'win32') {
      this.nvReaped = true;
      // No loop flag in the pattern so strays from any past cadence match
      execFile('pkill', ['-9', '-f', `nvidia-smi ${NV_QUERY_ARGS.join(' ')}`], () => this.startNvLoop());
      return;
    }
    this.nvReaped = true;

    let child: ChildProcess;
    try {
      child = spawn('nvidia-smi', [...NV_QUERY_ARGS, '-lms', String(MONITOR_TICK_MS)], {
        env: NV_ENV,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch {
      this.markNvUnavailable();
      return;
    }
    this.nvChild = child;
    this.nvStdoutBuffer = '';
    this.nvBatch = [];
    this.lastNvLineAt = Date.now();

    child.stdout!.on('data', (chunk: Buffer) => this.onNvData(chunk.toString()));
    child.stderr!.on('data', () => {
      // nvidia-smi warnings are not actionable here
    });
    child.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'ENOENT') {
        this.markNvUnavailable();
      }
    });
    child.on('exit', () => {
      if (this.nvChild !== child) return;
      this.nvChild = null;
      if (this.nvFlushTimer) {
        clearTimeout(this.nvFlushTimer);
        this.nvFlushTimer = null;
      }
      if (!this.nvUnavailable && !this.nvOneShotMode) {
        setTimeout(() => this.startNvLoop(), 5000);
      }
    });
  }

  private onNvData(text: string): void {
    this.nvStdoutBuffer += text;
    const lines = this.nvStdoutBuffer.split('\n');
    this.nvStdoutBuffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      const gpu = parseGpuLine(line);
      if (!gpu) continue;
      this.nvEverGotLine = true;
      this.lastNvLineAt = Date.now();
      this.nvBatch.push(gpu);
    }
    if (this.nvFlushTimer) clearTimeout(this.nvFlushTimer);
    this.nvFlushTimer = setTimeout(() => this.flushNvBatch(), NV_BATCH_FLUSH_MS);
  }

  private flushNvBatch(): void {
    this.nvFlushTimer = null;
    if (this.nvBatch.length === 0) return;
    this.latestGpu = {
      hasNvidiaSmi: true,
      isMac: false,
      gpus: this.nvBatch.sort((a, b) => a.index - b.index),
    };
    this.nvBatch = [];
  }

  private nvWatchdog(): void {
    if (this.nvUnavailable || !this.nvChild) return;
    if (Date.now() - this.lastNvLineAt <= NV_WATCHDOG_MS) return;
    console.warn('Monitor: no output from nvidia-smi loop, restarting it');
    // Loop mode that never produced a single line isn't going to start —
    // switch to a one-shot spawn per tick instead of kill/respawn forever.
    if (!this.nvEverGotLine) {
      this.nvOneShotMode = true;
    }
    this.nvChild.kill('SIGKILL');
  }

  private async sampleNvOneShot(): Promise<void> {
    if (this.nvUnavailable || this.nvOneShotInFlight) return;
    this.nvOneShotInFlight = true;
    try {
      const { stdout } = await execFileAsync('nvidia-smi', NV_QUERY_ARGS, { env: NV_ENV });
      const gpus = stdout
        .trim()
        .split('\n')
        .map(parseGpuLine)
        .filter((gpu): gpu is GpuInfo => gpu !== null)
        .sort((a, b) => a.index - b.index);
      this.latestGpu = { hasNvidiaSmi: true, isMac: false, gpus };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        this.markNvUnavailable();
      } else {
        console.error('Monitor: one-shot nvidia-smi failed:', err);
      }
    } finally {
      this.nvOneShotInFlight = false;
    }
  }

  private async detectBackend(force = false): Promise<'nvidia' | 'amd' | 'rocm' | 'none'> {
    if (this.backend && !force) return this.backend;
    // Check nvidia first
    try {
      await execFileAsync('which', ['nvidia-smi']);
      return 'nvidia';
    } catch {}
    try {
      await execFileAsync('nvidia-smi', ['-L']);
      return 'nvidia';
    } catch {}
    // Check amd-smi
    try {
      await execAsync('which amd-smi' as any);
      // Verify it actually works by listing
      await execAsync('amd-smi list --json' as any, { env: getVenvPath() } as any);
      return 'amd';
    } catch {}
    // Check rocm-smi
    try {
      await execAsync('which rocm-smi' as any);
      return 'rocm';
    } catch {}
    try {
      await execAsync('rocm-smi --version' as any);
      return 'rocm';
    } catch {}
    return 'none';
  }

  private async sampleAmdOneShot(): Promise<void> {
    if (this.amdInFlight) return;
    this.amdInFlight = true;
    try {
      const gpus = await getAmdSmiGpuStats();
      if (gpus.length > 0) {
        this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: true, hasRocmSmi: false, isMac: false, gpus };
      } else {
        // amd-smi returned empty (hasSufficientData false), fallback to rocm
        const rocmGpus = await getRocmGpuStats();
        if (rocmGpus.length > 0) {
          this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: true, hasRocmSmi: true, isMac: false, gpus: rocmGpus };
          this.backend = 'rocm';
        } else {
          this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: true, hasRocmSmi: false, isMac: false, gpus: [] };
        }
      }
    } catch (err) {
      try {
        const rocmGpus = await getRocmGpuStats();
        if (rocmGpus.length > 0) {
          this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: true, hasRocmSmi: true, isMac: false, gpus: rocmGpus };
          this.backend = 'rocm';
        } else {
          throw err;
        }
      } catch (e) {
        console.error('Monitor: amd-smi failed, trying rocm:', e);
        await this.sampleRocmOneShot();
      }
    } finally {
      this.amdInFlight = false;
    }
  }

  private async sampleRocmOneShot(): Promise<void> {
    if (this.rocmInFlight) return;
    this.rocmInFlight = true;
    try {
      const gpus = await getRocmGpuStats();
      this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: false, hasRocmSmi: true, isMac: false, gpus };
      if (gpus.length === 0) {
        this.latestGpu.error = 'rocm-smi returned no GPUs';
      }
    } catch (err) {
      if ((err as any).code === 'ENOENT') {
        this.rocmUnavailable = true;
        this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: false, hasRocmSmi: false, isMac: false, gpus: [], error: 'rocm-smi not found' };
      } else {
        console.error('Monitor: rocm-smi failed:', err);
        this.latestGpu = { hasNvidiaSmi: false, hasAmdSmi: false, hasRocmSmi: true, isMac: false, gpus: [], error: String(err) };
      }
    } finally {
      this.rocmInFlight = false;
    }
  }

  private markNvUnavailable(): void {
    this.nvUnavailable = true;
    this.latestGpu = {
      hasNvidiaSmi: false,
      isMac: false,
      gpus: [],
      error: 'nvidia-smi not found or not accessible',
    };
  }
}

/**
 * Idempotent starter. Guarded on globalThis so dev-mode module reloads never
 * stack a second sampler (and a second resident nvidia-smi) in the same
 * process.
 */
export function startMonitor(): SystemMonitor {
  const g = globalThis as unknown as { __aiToolkitSystemMonitor?: SystemMonitor };
  if (!g.__aiToolkitSystemMonitor) {
    g.__aiToolkitSystemMonitor = new SystemMonitor();
    // Escape hatch: with AI_TOOLKIT_DISABLE_MONITOR=1 the monitor object
    // exists (the SSE route stays functional) but never samples anything.
    if (process.env.AI_TOOLKIT_DISABLE_MONITOR !== '1') {
      g.__aiToolkitSystemMonitor.start();
    }
  }
  return g.__aiToolkitSystemMonitor;
}
