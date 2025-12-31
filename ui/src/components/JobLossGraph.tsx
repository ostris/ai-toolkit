'use client';

import { Job } from '@prisma/client';
import useJobLossLog, { LossPoint } from '@/hooks/useJobLossLog';
import { useMemo, useState, useEffect } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

interface Props {
  job: Job;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// EMA smoothing that works on a per-series list.
// alpha=1 -> no smoothing, alpha closer to 0 -> more smoothing.
function emaSmoothPoints(points: { step: number; value: number }[], alpha: number) {
  if (points.length === 0) return [];
  const a = clamp01(alpha);
  const out: { step: number; value: number }[] = new Array(points.length);

  let prev = points[0].value;
  out[0] = { step: points[0].step, value: prev };

  for (let i = 1; i < points.length; i++) {
    const x = points[i].value;
    prev = a * x + (1 - a) * prev;
    out[i] = { step: points[i].step, value: prev };
  }

  return out;
}

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

const PALETTE = [
  'rgba(96,165,250,1)', // blue-400
  'rgba(52,211,153,1)', // emerald-400
  'rgba(167,139,250,1)', // purple-400
  'rgba(251,191,36,1)', // amber-400
  'rgba(244,114,182,1)', // pink-400
  'rgba(248,113,113,1)', // red-400
  'rgba(34,211,238,1)', // cyan-400
  'rgba(129,140,248,1)', // indigo-400
];

function strokeForKey(key: string) {
  return PALETTE[hashToIndex(key, PALETTE.length)];
}

export default function JobLossGraph({ job }: Props) {
  const { series, lossKeys, status, refreshLoss } = useJobLossLog(job.id, 2000);

  // Controls
  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);

  // 0..100 slider. 100 = no smoothing, 0 = heavy smoothing.
  const [smoothing, setSmoothing] = useState(90);

  // UI-only downsample for rendering speed
  const [plotStride, setPlotStride] = useState(1);

  // show only last N points in the chart (0 = all)
  const [windowSize, setWindowSize] = useState<number>(4000);

  // quick y clipping for readability
  const [clipOutliers, setClipOutliers] = useState(false);

  // which loss series are enabled (default: all enabled)
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  // keep enabled map in sync with discovered keys (enable new ones automatically)
  useEffect(() => {
    setEnabled(prev => {
      const next = { ...prev };
      for (const k of lossKeys) {
        if (next[k] === undefined) next[k] = true;
      }
      // drop removed keys
      for (const k of Object.keys(next)) {
        if (!lossKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [lossKeys]);

  const activeKeys = useMemo(() => lossKeys.filter(k => enabled[k] !== false), [lossKeys, enabled]);

  const perSeries = useMemo(() => {
    // Build per-series processed point arrays (raw + smoothed), then merge by step for charting.
    const stride = Math.max(1, plotStride | 0);

    // smoothing%: 0 => no smoothing (alpha=1.0), 100 => heavy smoothing (alpha=0.02)
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98; // 1.0 -> 0.02

    const out: Record<string, { raw: { step: number; value: number }[]; smooth: { step: number; value: number }[] }> =
      {};

    for (const key of activeKeys) {
      const pts: LossPoint[] = series[key] ?? [];

      let raw = pts
        .filter(p => p.value !== null && Number.isFinite(p.value as number))
        .map(p => ({ step: p.step, value: p.value as number }))
        .filter(p => (useLogScale ? p.value > 0 : true))
        .filter((_, idx) => idx % stride === 0);

      // windowing (applies after stride)
      if (windowSize > 0 && raw.length > windowSize) {
        raw = raw.slice(raw.length - windowSize);
      }

      const smooth = emaSmoothPoints(raw, alpha);

      out[key] = { raw, smooth };
    }

    return out;
  }, [series, activeKeys, smoothing, plotStride, windowSize, useLogScale]);

  const chartData = useMemo(() => {
    // Merge series into one array of objects keyed by step.
    // Fields: `${key}__raw` and `${key}__smooth`
    const map = new Map<number, any>();

    for (const key of activeKeys) {
      const s = perSeries[key];
      if (!s) continue;

      for (const p of s.raw) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__raw`] = p.value;
        map.set(p.step, row);
      }
      for (const p of s.smooth) {
        const row = map.get(p.step) ?? { step: p.step };
        row[`${key}__smooth`] = p.value;
        map.set(p.step, row);
      }
    }

    const arr = Array.from(map.values());
    arr.sort((a, b) => a.step - b.step);
    return arr;
  }, [activeKeys, perSeries]);

  const hasData = chartData.length > 1;

  const yDomain = useMemo((): [number | 'auto', number | 'auto'] => {
    if (!clipOutliers || chartData.length < 10) return ['auto', 'auto'];

    // Collect visible values (prefer smoothed if shown, else raw)
    const vals: number[] = [];
    for (const row of chartData) {
      for (const key of activeKeys) {
        const k = showSmoothed ? `${key}__smooth` : `${key}__raw`;
        const v = row[k];
        if (typeof v === 'number' && Number.isFinite(v)) vals.push(v);
      }
    }
    if (vals.length < 10) return ['auto', 'auto'];

    vals.sort((a, b) => a - b);
    const lo = vals[Math.floor(vals.length * 0.02)];
    const hi = vals[Math.ceil(vals.length * 0.98) - 1];

    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return ['auto', 'auto'];
    return [lo, hi];
  }, [clipOutliers, chartData, activeKeys, showSmoothed]);

  const latestSummary = useMemo(() => {
    // Provide a simple “latest” readout for the first active series
    const firstKey = activeKeys[0];
    if (!firstKey) return null;

    const s = perSeries[firstKey];
    if (!s) return null;

    const lastRaw = s.raw.length ? s.raw[s.raw.length - 1] : null;
    const lastSmooth = s.smooth.length ? s.smooth[s.smooth.length - 1] : null;

    return {
      key: firstKey,
      step: lastRaw?.step ?? lastSmooth?.step ?? null,
      raw: lastRaw?.value ?? null,
      smooth: lastSmooth?.value ?? null,
    };
  }, [activeKeys, perSeries]);

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-blue-400" />
          <h2 className="text-gray-100 text-sm font-medium">Loss graph</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${chartData.length.toLocaleString()} steps`}
            {status === 'success' && !hasData && 'No data yet'}
          </span>
        </div>

        <button
          type="button"
          onClick={refreshLoss}
          className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
        >
          Refresh
        </button>
      </div>

      {/* Chart */}
      <div className="px-4  pt-4 pb-4">
        <div className="bg-gray-950 rounded-lg border border-gray-800 h-96 relative">
          {!hasData ? (
            <div className="h-full w-full flex items-center justify-center text-sm text-gray-400">
              {status === 'error' ? 'Failed to load loss logs.' : 'Waiting for loss points...'}
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 16, bottom: 10, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  minTickGap={40}
                />
                <YAxis
                  scale={useLogScale ? 'log' : 'linear'}
                  tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 12 }}
                  tickLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.15)' }}
                  width={72}
                  tickFormatter={formatNum}
                  domain={yDomain}
                  allowDataOverflow={clipOutliers}
                />
                <Tooltip
                  cursor={{ stroke: 'rgba(59,130,246,0.25)', strokeWidth: 1 }}
                  contentStyle={{
                    background: 'rgba(17,24,39,0.96)',
                    border: '1px solid rgba(31,41,55,1)',
                    borderRadius: 10,
                    color: 'rgba(255,255,255,0.9)',
                    fontSize: 12,
                  }}
                  labelStyle={{ color: 'rgba(255,255,255,0.75)' }}
                  labelFormatter={(label: any) => `step ${label}`}
                  formatter={(value: any, name: any) => [formatNum(Number(value)), name]}
                />

                <Legend
                  wrapperStyle={{
                    paddingTop: 8,
                    color: 'rgba(255,255,255,0.7)',
                    fontSize: 12,
                  }}
                />

                {activeKeys.map(k => {
                  const color = strokeForKey(k);

                  return (
                    <g key={k}>
                      {showRaw && (
                        <Line
                          type="monotone"
                          dataKey={`${k}__raw`}
                          name={`${k} (raw)`}
                          stroke={color.replace('1)', '0.40)')}
                          strokeWidth={1.25}
                          dot={false}
                          isAnimationActive={false}
                        />
                      )}
                      {showSmoothed && (
                        <Line
                          type="monotone"
                          dataKey={`${k}__smooth`}
                          name={`${k}`}
                          stroke={color}
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      )}
                    </g>
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 pb-2">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Display</label>
            <div className="flex flex-wrap gap-2">
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
              <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers(v => !v)} label="Clip outliers" />
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <label className="block text-xs text-gray-400 mb-2">Series</label>
            {lossKeys.length === 0 ? (
              <div className="text-sm text-gray-400">No loss keys found yet.</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {lossKeys.map(k => (
                  <button
                    key={k}
                    type="button"
                    onClick={() => setEnabled(prev => ({ ...prev, [k]: !(prev[k] ?? true) }))}
                    className={[
                      'px-3 py-1 rounded-md text-xs border transition-colors',
                      enabled[k] === false
                        ? 'bg-gray-900 text-gray-400 border-gray-800 hover:bg-gray-800/60'
                        : 'bg-gray-900 text-gray-200 border-gray-800 hover:bg-gray-800/60',
                    ].join(' ')}
                    aria-pressed={enabled[k] !== false}
                    title={k}
                  >
                    <span className="inline-block h-2 w-2 rounded-full mr-2" style={{ background: strokeForKey(k) }} />
                    {k}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Smoothing</label>
              <span className="text-xs text-gray-300">{smoothing}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={smoothing}
              onChange={e => setSmoothing(Number(e.target.value))}
              className="w-full accent-blue-500"
              disabled={!showSmoothed}
            />
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Plot stride</label>
              <span className="text-xs text-gray-300">every {plotStride} pt</span>
            </div>
            <input
              type="range"
              min={1}
              max={20}
              value={plotStride}
              onChange={e => setPlotStride(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">UI downsample for huge runs.</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 md:col-span-2">
            <div className="flex items-center justify-between mb-1">
              <label className="block text-xs text-gray-400">Window (last N points)</label>
              <span className="text-xs text-gray-300">{windowSize === 0 ? 'all' : windowSize.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min={0}
              max={20000}
              step={250}
              value={windowSize}
              onChange={e => setWindowSize(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <div className="mt-2 text-[11px] text-gray-500">
              Set to 0 to show all (not recommended for very long runs).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}
