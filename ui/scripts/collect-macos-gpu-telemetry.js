#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const { execFile } = require('child_process');

const execFileAsync = promisify(execFile);

const OUTPUT_PATH = process.env.AI_TOOLKIT_MAC_GPU_TELEMETRY_PATH || '/tmp/ai-toolkit-mac-gpu-telemetry.json';
const SAMPLE_MS = Number.parseInt(process.env.AI_TOOLKIT_MAC_GPU_SAMPLE_MS || '2000', 10);
const SAFE_SAMPLE_MS = Number.isFinite(SAMPLE_MS) && SAMPLE_MS >= 250 ? SAMPLE_MS : 2000;

function usageAndExit(message, code = 1) {
  console.error(message);
  process.exit(code);
}

if (process.platform !== 'darwin') {
  usageAndExit('This collector is macOS-only.');
}

if (typeof process.getuid === 'function' && process.getuid() !== 0) {
  usageAndExit('powermetrics requires root. Run with sudo.');
}

function asNumber(value) {
  return Number.isFinite(value) ? value : undefined;
}

function parseWithUnit(pattern, text, unitConversion = x => x) {
  const match = text.match(pattern);
  if (!match) return undefined;

  const rawValue = Number.parseFloat(match[1]);
  if (!Number.isFinite(rawValue)) return undefined;

  const maybeUnit = match[2] ? String(match[2]).toLowerCase() : '';
  return unitConversion(rawValue, maybeUnit);
}

function parsePowermetricsOutput(text) {
  const powerDrawW = parseWithUnit(/GPU Power:\s*([0-9.]+)\s*(mW|W)\b/i, text, (value, unit) => {
    if (unit === 'mw') return value / 1000;
    return value;
  });

  const powerLimitW = parseWithUnit(/GPU Power(?:\s+Limit)?:\s*[0-9.]+\s*(?:mW|W)\s*\/\s*([0-9.]+)\s*(mW|W)\b/i, text, (value, unit) => {
    if (unit === 'mw') return value / 1000;
    return value;
  });

  const temperatureC = parseWithUnit(/GPU die temperature:\s*([0-9.]+)\s*C\b/i, text);
  const clockMHz = parseWithUnit(/GPU(?:\s+HW)?\s+active frequency:\s*([0-9.]+)\s*MHz\b/i, text);
  const utilizationGpuPercent = parseWithUnit(/GPU HW active residency:\s*([0-9.]+)\s*%\b/i, text);

  const fanMatches = [...text.matchAll(/Fan(?:\s+\d+)?:\s*([0-9.]+)\s*rpm\b/gi)];
  const fanRpmValues = fanMatches
    .map(match => Number.parseFloat(match[1]))
    .filter(value => Number.isFinite(value));
  const fanRpm = fanRpmValues.length > 0 ? Math.max(...fanRpmValues) : undefined;

  return {
    source: 'powermetrics',
    timestampMs: Date.now(),
    temperatureC: asNumber(temperatureC),
    fanRpm: asNumber(fanRpm),
    powerDrawW: asNumber(powerDrawW),
    powerLimitW: asNumber(powerLimitW),
    clockMHz: asNumber(clockMHz),
    utilizationGpuPercent: asNumber(utilizationGpuPercent),
  };
}

function writeTelemetrySnapshot(snapshot) {
  const directory = path.dirname(OUTPUT_PATH);
  fs.mkdirSync(directory, { recursive: true });

  const tempPath = `${OUTPUT_PATH}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(snapshot), 'utf8');
  fs.renameSync(tempPath, OUTPUT_PATH);
}

async function collectOnce() {
  const args = ['--samplers', 'cpu_power,gpu_power,thermal', '-n', '1', '-i', String(SAFE_SAMPLE_MS)];
  const { stdout = '', stderr = '' } = await execFileAsync('powermetrics', args, {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 4,
  });
  const sample = parsePowermetricsOutput(`${stdout}\n${stderr}`);
  writeTelemetrySnapshot(sample);
}

async function runLoop() {
  console.log(`Starting macOS GPU telemetry collector. Writing to ${OUTPUT_PATH}`);
  console.log(`Sampling with powermetrics every ${SAFE_SAMPLE_MS}ms`);

  while (true) {
    try {
      await collectOnce();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const fallbackSnapshot = {
        source: 'powermetrics',
        timestampMs: Date.now(),
        error: message,
      };
      try {
        writeTelemetrySnapshot(fallbackSnapshot);
      } catch {
        // If file write fails we still keep the loop alive.
      }
      console.error(`[collector] ${message}`);
      await new Promise(resolve => setTimeout(resolve, SAFE_SAMPLE_MS));
    }
  }
}

runLoop().catch(error => {
  usageAndExit(error instanceof Error ? error.message : String(error));
});
