'use client';

import { useEffect, useState, useMemo } from 'react';
import { Job } from '@prisma/client';
import { Activity, TrendingDown, TrendingUp, Gauge, Layers, BarChart3, HelpCircle } from 'lucide-react';

// Tooltip component for explaining metrics
const Tooltip = ({ children, text }: { children: React.ReactNode; text: string }) => (
  <div className="group relative inline-flex items-center">
    {children}
    <HelpCircle className="w-3 h-3 ml-1 text-gray-500 hover:text-gray-300 cursor-help" />
    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 w-64 shadow-xl border border-gray-700">
      {text}
      <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1 border-4 border-transparent border-t-gray-800"></div>
    </div>
  </div>
);

interface MetricsData {
  step: number;
  timestamp?: string;
  loss?: number;
  loss_slope?: number;
  loss_r2?: number;
  gradient_stability?: number;
  gradient_stability_avg?: number;
  expert?: string;
  alpha_enabled?: boolean;
  phase?: string;
  phase_idx?: number;
  steps_in_phase?: number;
  conv_alpha?: number;
  linear_alpha?: number;
  learning_rate?: number;
  lr_0?: number;  // MoE: learning rate for expert 0
  lr_1?: number;  // MoE: learning rate for expert 1
}

interface JobMetricsProps {
  job: Job;
}

export default function JobMetrics({ job }: JobMetricsProps) {
  const [metrics, setMetrics] = useState<MetricsData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [windowSize, setWindowSize] = useState<10 | 50 | 100>(100);
  const [switchBoundaryEvery, setSwitchBoundaryEvery] = useState<number>(100);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await fetch(`/api/jobs/${job.id}/metrics`);
        const data = await res.json();

        if (data.error) {
          setError(data.error);
        } else {
          setMetrics(data.metrics || []);
          if (data.switchBoundaryEvery) {
            setSwitchBoundaryEvery(data.switchBoundaryEvery);
          }
        }
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch metrics');
        setLoading(false);
      }
    };

    fetchMetrics();

    // Poll every 5 seconds if job is running
    if (job.status === 'running') {
      const interval = setInterval(fetchMetrics, 5000);
      return () => clearInterval(interval);
    }
  }, [job.id, job.status]);

  // Calculate aggregate statistics with configurable window
  const stats = useMemo(() => {
    if (metrics.length === 0) return null;

    const currentMetric = metrics[metrics.length - 1];

    // Helper function to infer expert from step number
    const inferExpert = (m: MetricsData): string => {
      if (m.expert) return m.expert;
      // MoE switches experts every switchBoundaryEvery steps
      const blockIndex = Math.floor(m.step / switchBoundaryEvery);
      return blockIndex % 2 === 0 ? 'high_noise' : 'low_noise';
    };

    // CRITICAL FIX: Separate by expert FIRST, then apply windowing
    // This prevents mixing high-noise and low-noise data in the same window
    const allHighNoiseMetrics = metrics.filter(m => inferExpert(m) === 'high_noise');
    const allLowNoiseMetrics = metrics.filter(m => inferExpert(m) === 'low_noise');

    // Apply windowing to each expert separately
    const recentHighNoise = allHighNoiseMetrics.slice(-windowSize);
    const recentLowNoise = allLowNoiseMetrics.slice(-windowSize);

    // For backward compatibility, also create a mixed recent window
    const recent = metrics.slice(-windowSize);
    const losses = recent.filter(m => m.loss != null).map(m => m.loss!);
    const gradStabilities = recent.filter(m => m.gradient_stability != null).map(m => m.gradient_stability!);

    // Calculate loss statistics from mixed window (for overall metrics)
    const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / losses.length : null;
    const minLoss = losses.length > 0 ? Math.min(...losses) : null;
    const maxLoss = losses.length > 0 ? Math.max(...losses) : null;

    // Calculate Exponential Moving Average (EMA) for loss with spike filtering
    // EMA gives more weight to recent values: EMA_t = α * value_t + (1-α) * EMA_{t-1}
    // α (smoothing factor) = 2 / (N + 1), where N is the window size
    // SPIKE_THRESHOLD filters out expert-switch spikes (e.g., 0.554 at boundary)
    const SPIKE_THRESHOLD = 0.3; // Filter losses > 0.3 from EMA calculation
    const calculateEMA = (values: number[], windowSize: number, filterSpikes: boolean = false) => {
      if (values.length === 0) return null;
      const alpha = 2 / (windowSize + 1);

      // Optionally filter extreme spikes (from expert switches)
      const filtered = filterSpikes ? values.filter(v => v < SPIKE_THRESHOLD) : values;
      if (filtered.length === 0) return null;

      let ema = filtered[0]; // Initialize with first value
      for (let i = 1; i < filtered.length; i++) {
        ema = alpha * filtered[i] + (1 - alpha) * ema;
      }
      return ema;
    };

    const emaLoss = calculateEMA(losses, windowSize);

    // Calculate gradient stability statistics from mixed window
    const avgGradStability = gradStabilities.length > 0
      ? gradStabilities.reduce((a, b) => a + b, 0) / gradStabilities.length
      : null;
    const emaGradStability = calculateEMA(gradStabilities, windowSize);

    // Extract per-expert data from properly windowed metrics
    const highNoiseLosses = recentHighNoise.filter(m => m.loss != null).map(m => m.loss!);
    const lowNoiseLosses = recentLowNoise.filter(m => m.loss != null).map(m => m.loss!);

    const highNoiseLoss = highNoiseLosses.length > 0
      ? highNoiseLosses.reduce((a, b) => a + b, 0) / highNoiseLosses.length
      : null;

    const lowNoiseLoss = lowNoiseLosses.length > 0
      ? lowNoiseLosses.reduce((a, b) => a + b, 0) / lowNoiseLosses.length
      : null;

    // Calculate per-expert EMAs with spike filtering enabled
    const highNoiseLossEMA = calculateEMA(highNoiseLosses, windowSize, true);
    const lowNoiseLossEMA = calculateEMA(lowNoiseLosses, windowSize, true);

    const highNoiseGradStabilities = recentHighNoise.filter(m => m.gradient_stability != null).map(m => m.gradient_stability!);
    const lowNoiseGradStabilities = recentLowNoise.filter(m => m.gradient_stability != null).map(m => m.gradient_stability!);

    const highNoiseGradStabilityEMA = calculateEMA(highNoiseGradStabilities, windowSize);
    const lowNoiseGradStabilityEMA = calculateEMA(lowNoiseGradStabilities, windowSize);

    return {
      current: currentMetric,
      avgLoss,
      emaLoss,
      minLoss,
      maxLoss,
      avgGradStability,
      emaGradStability,
      highNoiseLoss,
      lowNoiseLoss,
      highNoiseLossEMA,
      lowNoiseLossEMA,
      highNoiseGradStabilityEMA,
      lowNoiseGradStabilityEMA,
      totalSteps: metrics.length,
      recentMetrics: recent,
      recentHighNoise,  // NEW: properly windowed high-noise data
      recentLowNoise,   // NEW: properly windowed low-noise data
    };
  }, [metrics, windowSize, switchBoundaryEvery]);

  if (loading) {
    return (
      <div className="p-6 text-center text-gray-400">
        <Activity className="w-8 h-8 mx-auto mb-2 animate-pulse" />
        <p>Loading metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-amber-500">
        <p>{error}</p>
      </div>
    );
  }

  if (!stats || metrics.length === 0) {
    return (
      <div className="p-6 text-center text-gray-400">
        <Activity className="w-8 h-8 mx-auto mb-2" />
        <p>No metrics data available yet.</p>
        <p className="text-sm mt-2">Metrics will appear once training starts.</p>
      </div>
    );
  }

  const { current } = stats;

  // Determine which expert is currently active based on step
  const currentBlockIndex = Math.floor(current.step / switchBoundaryEvery);
  const currentActiveExpert = currentBlockIndex % 2 === 0 ? 'high_noise' : 'low_noise';
  const stepsInCurrentBlock = current.step % switchBoundaryEvery;

  // Separate ALL metrics by expert for full history visualization
  const allWithExpert = metrics.map((m) => {
    if (m.expert) return { ...m, inferredExpert: m.expert };
    // Calculate which block this step is in based on switchBoundaryEvery
    const blockIndex = Math.floor(m.step / switchBoundaryEvery);
    const inferredExpert = blockIndex % 2 === 0 ? 'high_noise' : 'low_noise';
    return { ...m, inferredExpert };
  });

  const allHighNoiseData = allWithExpert.filter(m => m.inferredExpert === 'high_noise');
  const allLowNoiseData = allWithExpert.filter(m => m.inferredExpert === 'low_noise');

  // Use properly windowed per-expert data from stats
  // CRITICAL: These are already separated by expert BEFORE windowing
  const highNoiseData = stats.recentHighNoise;
  const lowNoiseData = stats.recentLowNoise;

  // Helper function to calculate regression line for a dataset
  const calculateRegression = (data: MetricsData[]) => {
    const lossDataPoints = data
      .map((m, idx) => ({ x: idx, y: m.loss }))
      .filter(p => p.y != null) as { x: number; y: number }[];

    let regressionLine: { x: number; y: number }[] = [];
    let slope = 0;

    if (lossDataPoints.length > 2) {
      const n = lossDataPoints.length;
      const sumX = lossDataPoints.reduce((sum, p) => sum + p.x, 0);
      const sumY = lossDataPoints.reduce((sum, p) => sum + p.y, 0);
      const sumXY = lossDataPoints.reduce((sum, p) => sum + p.x * p.y, 0);
      const sumX2 = lossDataPoints.reduce((sum, p) => sum + p.x * p.x, 0);

      slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
      const intercept = (sumY - slope * sumX) / n;

      regressionLine = [
        { x: 0, y: intercept },
        { x: data.length - 1, y: slope * (data.length - 1) + intercept }
      ];
    }

    return { regressionLine, slope };
  };

  // Recent window regressions
  const highNoiseRegression = calculateRegression(highNoiseData);
  const lowNoiseRegression = calculateRegression(lowNoiseData);

  // Full history regressions
  const allHighNoiseRegression = calculateRegression(allHighNoiseData);
  const allLowNoiseRegression = calculateRegression(allLowNoiseData);

  // Calculate chart bounds from windowed data
  const allLosses = stats.recentMetrics.filter(m => m.loss != null).map(m => m.loss!);
  const maxChartLoss = allLosses.length > 0 ? Math.max(...allLosses) : 1;
  const minChartLoss = allLosses.length > 0 ? Math.min(...allLosses) : 0;
  const lossRange = maxChartLoss - minChartLoss || 0.1;

  // Calculate chart bounds from ALL data for full history charts
  const allHistoryLosses = metrics.filter(m => m.loss != null).map(m => m.loss!);
  const maxAllLoss = allHistoryLosses.length > 0 ? Math.max(...allHistoryLosses) : 1;
  const minAllLoss = allHistoryLosses.length > 0 ? Math.min(...allHistoryLosses) : 0;
  const allLossRange = maxAllLoss - minAllLoss || 0.1;

  // Helper function to render a loss chart for a specific expert
  const renderLossChart = (
    data: MetricsData[],
    regression: { regressionLine: { x: number; y: number }[]; slope: number },
    expertName: string,
    color: string,
    minLoss: number,
    maxLoss: number,
    lossRangeParam: number
  ) => {
    if (data.length === 0) {
      return <div className="text-gray-500 text-sm text-center py-8">No data for {expertName}</div>;
    }

    return (
      <div className="relative h-48 bg-gray-950 rounded-lg p-4">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500 pr-2 text-right">
          <span>{maxLoss.toFixed(3)}</span>
          <span>{((maxLoss + minLoss) / 2).toFixed(3)}</span>
          <span>{minLoss.toFixed(3)}</span>
        </div>

        {/* Chart area */}
        <div className="absolute left-12 right-4 top-4 bottom-8 flex items-end justify-between gap-px">
          {data.map((m, idx) => {
            if (m.loss == null) return <div key={idx} className="flex-1" />;

            const heightPercent = ((m.loss - minLoss) / lossRangeParam) * 100;
            return (
              <div
                key={idx}
                className={`flex-1 ${color} hover:brightness-110 transition-all relative group`}
                style={{ height: `${heightPercent}%` }}
                title={`Step ${m.step}: ${m.loss.toFixed(4)}`}
              >
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                  {m.loss.toFixed(4)}
                </div>
              </div>
            );
          })}
        </div>

        {/* Line of best fit overlay */}
        {regression.regressionLine.length === 2 && (
          <svg className="absolute left-12 right-4 top-4 bottom-8 pointer-events-none" style={{ width: 'calc(100% - 4rem)', height: 'calc(100% - 3rem)' }}>
            <line
              x1={`${(regression.regressionLine[0].x / (data.length - 1)) * 100}%`}
              y1={`${100 - ((regression.regressionLine[0].y - minLoss) / lossRangeParam) * 100}%`}
              x2={`${(regression.regressionLine[1].x / (data.length - 1)) * 100}%`}
              y2={`${100 - ((regression.regressionLine[1].y - minLoss) / lossRangeParam) * 100}%`}
              stroke="#ef4444"
              strokeWidth="2"
              strokeDasharray="4 4"
            />
            {/* Slope indicator label */}
            <text
              x="95%"
              y="10%"
              fill="#ef4444"
              fontSize="10"
              textAnchor="end"
              className="font-mono"
            >
              slope: {regression.slope.toFixed(4)}
            </text>
          </svg>
        )}

        {/* X-axis label */}
        <div className="absolute bottom-0 left-12 right-4 text-xs text-gray-500 text-center">
          Steps (most recent →)
        </div>
      </div>
    );
  };

  // Helper function to render gradient stability chart for a specific expert
  const renderGradientChart = (
    data: MetricsData[],
    expertName: string,
    color: string
  ) => {
    if (data.length === 0) {
      return <div className="text-gray-500 text-sm text-center py-8">No data for {expertName}</div>;
    }

    return (
      <div className="relative h-32 bg-gray-950 rounded-lg p-4">
        {/* Target zone indicator */}
        <div className="absolute left-12 right-4 bg-green-500/10 border border-green-500/30"
             style={{ top: '20%', bottom: '50%' }}>
          <span className="absolute right-2 top-0 text-xs text-green-400">Target Zone</span>
        </div>

        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500 pr-2 text-right">
          <span>100%</span>
          <span>50%</span>
          <span>0%</span>
        </div>

        {/* Chart bars */}
        <div className="absolute left-12 right-4 top-4 bottom-8 flex items-end justify-between gap-px">
          {data.map((m, idx) => {
            if (m.gradient_stability == null) return <div key={idx} className="flex-1" />;

            const heightPercent = m.gradient_stability * 100;
            const isInTarget = m.gradient_stability >= 0.55 && m.gradient_stability <= 0.70;

            return (
              <div
                key={idx}
                className={`flex-1 ${isInTarget ? 'bg-green-500' : 'bg-amber-500'} hover:brightness-110 transition-all relative group`}
                style={{ height: `${heightPercent}%` }}
                title={`Step ${m.step}: ${(m.gradient_stability * 100).toFixed(1)}%`}
              >
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
                  {(m.gradient_stability * 100).toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>

        {/* X-axis label */}
        <div className="absolute bottom-0 left-12 right-4 text-xs text-gray-500 text-center">
          Steps (most recent →)
        </div>
      </div>
    );
  };

  // Helper function to render learning rate chart for MoE (both experts on same chart)
  const renderLearningRateChart = () => {
    const dataWithLR = stats.recentMetrics.filter(m => m.lr_0 != null || m.lr_1 != null);

    if (dataWithLR.length === 0) {
      return <div className="text-gray-500 text-sm text-center py-8">No learning rate data available</div>;
    }

    // Calculate Y-axis range
    const allLRs = dataWithLR.flatMap(m => [m.lr_0, m.lr_1].filter(lr => lr != null)) as number[];
    const maxLR = Math.max(...allLRs);
    const minLR = Math.min(...allLRs);
    const lrRange = maxLR - minLR || 0.0001;

    return (
      <div className="relative h-48 bg-gray-950 rounded-lg p-4">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 w-16 flex flex-col justify-between text-xs text-gray-500 pr-2 text-right">
          <span>{maxLR.toExponential(2)}</span>
          <span>{((maxLR + minLR) / 2).toExponential(2)}</span>
          <span>{minLR.toExponential(2)}</span>
        </div>

        {/* Chart area with lines */}
        <svg className="absolute left-16 right-4 top-4 bottom-8" viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: 'calc(100% - 5rem)', height: 'calc(100% - 3rem)' }}>
          {/* High Noise (lr_0) line */}
          <polyline
            points={dataWithLR.map((m, idx) => {
              const x = (idx / (dataWithLR.length - 1)) * 100;
              const y = m.lr_0 != null ? (1 - ((m.lr_0 - minLR) / lrRange)) * 100 : null;
              return y != null ? `${x},${y}` : null;
            }).filter(p => p).join(' ')}
            fill="none"
            stroke="#fb923c"
            strokeWidth="0.5"
            vectorEffect="non-scaling-stroke"
          />

          {/* Low Noise (lr_1) line */}
          <polyline
            points={dataWithLR.map((m, idx) => {
              const x = (idx / (dataWithLR.length - 1)) * 100;
              const y = m.lr_1 != null ? (1 - ((m.lr_1 - minLR) / lrRange)) * 100 : null;
              return y != null ? `${x},${y}` : null;
            }).filter(p => p).join(' ')}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="0.5"
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        {/* Legend */}
        <div className="absolute top-2 right-4 flex gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-orange-400"></div>
            <span className="text-gray-400">High Noise</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-blue-500"></div>
            <span className="text-gray-400">Low Noise</span>
          </div>
        </div>

        {/* X-axis label */}
        <div className="absolute bottom-0 left-16 right-4 text-xs text-gray-500 text-center">
          Steps (most recent →)
        </div>
      </div>
    );
  };

  // Helper function to render alpha scheduling chart (conv and linear alphas)
  const renderAlphaChart = () => {
    const dataWithAlpha = stats.recentMetrics.filter(m => m.conv_alpha != null || m.linear_alpha != null);

    if (dataWithAlpha.length === 0) {
      return <div className="text-gray-500 text-sm text-center py-8">Alpha scheduling not enabled</div>;
    }

    // Calculate Y-axis range
    const allAlphas = dataWithAlpha.flatMap(m => [m.conv_alpha, m.linear_alpha].filter(a => a != null)) as number[];
    const maxAlpha = Math.max(...allAlphas);
    const minAlpha = Math.min(...allAlphas);
    const alphaRange = maxAlpha - minAlpha || 0.1;

    return (
      <div className="relative h-48 bg-gray-950 rounded-lg p-4">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500 pr-2 text-right">
          <span>{maxAlpha.toFixed(1)}</span>
          <span>{((maxAlpha + minAlpha) / 2).toFixed(1)}</span>
          <span>{minAlpha.toFixed(1)}</span>
        </div>

        {/* Chart area with lines and phase backgrounds */}
        <svg className="absolute left-12 right-4 top-4 bottom-8" viewBox="0 0 100 100" preserveAspectRatio="none" style={{ width: 'calc(100% - 4rem)', height: 'calc(100% - 3rem)' }}>
          {/* Conv Alpha line */}
          <polyline
            points={dataWithAlpha.map((m, idx) => {
              const x = (idx / (dataWithAlpha.length - 1)) * 100;
              const y = m.conv_alpha != null ? (1 - ((m.conv_alpha - minAlpha) / alphaRange)) * 100 : null;
              return y != null ? `${x},${y}` : null;
            }).filter(p => p).join(' ')}
            fill="none"
            stroke="#10b981"
            strokeWidth="0.5"
            vectorEffect="non-scaling-stroke"
          />

          {/* Linear Alpha line */}
          <polyline
            points={dataWithAlpha.map((m, idx) => {
              const x = (idx / (dataWithAlpha.length - 1)) * 100;
              const y = m.linear_alpha != null ? (1 - ((m.linear_alpha - minAlpha) / alphaRange)) * 100 : null;
              return y != null ? `${x},${y}` : null;
            }).filter(p => p).join(' ')}
            fill="none"
            stroke="#8b5cf6"
            strokeWidth="0.5"
            strokeDasharray="2 2"
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        {/* Legend */}
        <div className="absolute top-2 right-4 flex gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-green-500"></div>
            <span className="text-gray-400">Conv Alpha</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-purple-500" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #8b5cf6 0, #8b5cf6 4px, transparent 4px, transparent 8px)' }}></div>
            <span className="text-gray-400">Linear Alpha</span>
          </div>
        </div>

        {/* X-axis label */}
        <div className="absolute bottom-0 left-12 right-4 text-xs text-gray-500 text-center">
          Steps (most recent →)
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6 p-6">
      {/* Window Size Selector */}
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold text-gray-200">Training Metrics</h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-400">Window:</span>
          <div className="flex gap-1">
            {[10, 50, 100].map((size) => (
              <button
                key={size}
                onClick={() => setWindowSize(size as 10 | 50 | 100)}
                className={`px-3 py-1 rounded text-sm ${
                  windowSize === size
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {size}
              </button>
            ))}
          </div>
          <span className="text-sm text-gray-400">steps</span>
        </div>
      </div>

      {/* Alpha Schedule Status (if enabled) */}
      {current.alpha_enabled && (
        <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-xl p-6 border border-blue-800/30">
          <h3 className="text-xl font-semibold mb-4 flex items-center text-blue-300">
            <Layers className="w-5 h-5 mr-2" />
            Alpha Schedule Progress
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Tooltip text="Which training phase you're in. Foundation (α=8) → Balance (α=14) → Emphasis (α=20). Transitions automatically when loss plateaus and training stabilizes.">
                <p className="text-sm text-gray-400">Current Phase</p>
              </Tooltip>
              <p className="text-2xl font-bold text-blue-400 uppercase">{current.phase || 'N/A'}</p>
              <p className="text-xs text-gray-500">Step {current.steps_in_phase} in phase</p>
            </div>
            <div>
              <Tooltip text="How strong your LoRA effect is for convolutional layers. Starts low (8) for stability, increases as training progresses. LoRA strength = alpha / rank.">
                <p className="text-sm text-gray-400">Conv Alpha</p>
              </Tooltip>
              <p className="text-2xl font-bold text-green-400">{current.conv_alpha?.toFixed(2) || 'N/A'}</p>
            </div>
            <div>
              <Tooltip text="LoRA strength for linear layers. Usually kept constant while conv alpha increases. Affects how much the LoRA influences the model.">
                <p className="text-sm text-gray-400">Linear Alpha</p>
              </Tooltip>
              <p className="text-2xl font-bold text-green-400">{current.linear_alpha?.toFixed(2) || 'N/A'}</p>
            </div>
          </div>
        </div>
      )}

      {/* Full History Loss Charts - Per Expert */}
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-300 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-purple-400" />
          Full Training History (Step 0 → {metrics.length > 0 ? metrics[metrics.length - 1].step : 0})
        </h3>
        <p className="text-xs text-gray-500">Complete training progression showing all {metrics.length} logged steps</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* High Noise Expert - Full History */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-orange-400" />
              High Noise Expert Loss
            </h3>
            <div className="text-sm text-gray-400">
              {allHighNoiseData.length} steps
            </div>
          </div>
          {renderLossChart(allHighNoiseData, allHighNoiseRegression, 'High Noise', 'bg-orange-500', minAllLoss, maxAllLoss, allLossRange)}
        </div>

        {/* Low Noise Expert - Full History */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
              Low Noise Expert Loss
            </h3>
            <div className="text-sm text-gray-400">
              {allLowNoiseData.length} steps
            </div>
          </div>
          {renderLossChart(allLowNoiseData, allLowNoiseRegression, 'Low Noise', 'bg-blue-500', minAllLoss, maxAllLoss, allLossRange)}
        </div>
      </div>

      {/* Recent Window Loss Charts - Per Expert */}
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-gray-300 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          Recent Window (Last {windowSize} steps)
        </h3>
        <p className="text-xs text-gray-500">Detailed view of recent training behavior</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* High Noise Expert - Recent */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-orange-400" />
              High Noise Expert Loss
            </h3>
            <div className="text-sm text-gray-400">
              Avg: {stats.highNoiseLoss != null ? stats.highNoiseLoss.toFixed(4) : 'N/A'}
            </div>
          </div>
          {renderLossChart(highNoiseData, highNoiseRegression, 'High Noise', 'bg-orange-500', minChartLoss, maxChartLoss, lossRange)}
        </div>

        {/* Low Noise Expert - Recent */}
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200 flex items-center">
              <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
              Low Noise Expert Loss
            </h3>
            <div className="text-sm text-gray-400">
              Avg: {stats.lowNoiseLoss != null ? stats.lowNoiseLoss.toFixed(4) : 'N/A'}
            </div>
          </div>
          {renderLossChart(lowNoiseData, lowNoiseRegression, 'Low Noise', 'bg-blue-500', minChartLoss, maxChartLoss, lossRange)}
        </div>
      </div>

      {/* Gradient Stability Charts - Per Expert */}
      {stats.avgGradStability != null && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* High Noise Expert */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-200 flex items-center">
                <Gauge className="w-5 h-5 mr-2 text-orange-400" />
                High Noise Gradient Stability
              </h3>
              <div className="text-sm text-gray-400">
                Target: <span className="text-green-400">0.55-0.70</span>
              </div>
            </div>
            {renderGradientChart(highNoiseData, 'High Noise', 'bg-orange-500')}
          </div>

          {/* Low Noise Expert */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-200 flex items-center">
                <Gauge className="w-5 h-5 mr-2 text-blue-400" />
                Low Noise Gradient Stability
              </h3>
              <div className="text-sm text-gray-400">
                Target: <span className="text-green-400">0.55-0.70</span>
              </div>
            </div>
            {renderGradientChart(lowNoiseData, 'Low Noise', 'bg-blue-500')}
          </div>
        </div>
      )}

      {/* Learning Rate Chart - Per Expert */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-200 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-yellow-400" />
            Learning Rate per Expert
          </h3>
        </div>
        {renderLearningRateChart()}
      </div>

      {/* Alpha Scheduling Chart (if enabled) */}
      {stats.recentMetrics.some(m => m.conv_alpha != null || m.linear_alpha != null) && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200 flex items-center">
              <Layers className="w-5 h-5 mr-2 text-purple-400" />
              Alpha Scheduler Progress
            </h3>
          </div>
          {renderAlphaChart()}
        </div>
      )}

      {/* Training Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Current Loss */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="How wrong your model's predictions are. Lower is better. Should decrease over time. Video training is noisier than image training.">
              <p className="text-sm text-gray-400">Current Loss</p>
            </Tooltip>
            <Activity className="w-4 h-4 text-blue-400" />
          </div>
          <p className="text-2xl font-bold text-blue-400">
            {current.loss != null ? current.loss.toFixed(4) : 'N/A'}
          </p>
          {current.loss_slope != null && (
            <p className="text-xs text-gray-500 mt-1 flex items-center">
              {current.loss_slope > 0 ? (
                <><TrendingUp className="w-3 h-3 mr-1 text-red-400" />Increasing</>
              ) : (
                <><TrendingDown className="w-3 h-3 mr-1 text-green-400" />Decreasing</>
              )}
            </p>
          )}
        </div>

        {/* Average Loss */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-gray-400">Avg Loss ({windowSize})</p>
            <Activity className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-purple-400">
            {stats.avgLoss != null ? stats.avgLoss.toFixed(4) : 'N/A'}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Range: {stats.minLoss?.toFixed(4)} - {stats.maxLoss?.toFixed(4)}
          </p>
        </div>

        {/* EMA Loss */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="Exponential Moving Average of loss. Weights recent values more heavily than simple average. Best indicator of current training trend.">
              <p className="text-sm text-gray-400">EMA Loss ({windowSize})</p>
            </Tooltip>
            <TrendingDown className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-2xl font-bold text-cyan-400">
            {stats.emaLoss != null ? stats.emaLoss.toFixed(4) : 'N/A'}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Weighted toward recent steps
          </p>
        </div>

        {/* Gradient Stability */}
        {stats.avgGradStability != null && (
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="flex items-center justify-between mb-2">
              <Tooltip text="How consistent your training updates are (0-100%). This is MEASURED, not a setting. Video needs >50%, images need >55%. Below threshold means training is unstable.">
                <p className="text-sm text-gray-400">Grad Stability</p>
              </Tooltip>
              <Gauge className="w-4 h-4 text-green-400" />
            </div>
            <p className="text-2xl font-bold text-green-400">
              {(stats.avgGradStability * 100).toFixed(1)}%
            </p>
            <p className="text-xs mt-1">
              {stats.avgGradStability >= 0.55 && stats.avgGradStability <= 0.70 ? (
                <span className="text-green-400">✓ In target range</span>
              ) : stats.avgGradStability < 0.55 ? (
                <span className="text-amber-400">⚠ Below target (0.55)</span>
              ) : (
                <span className="text-amber-400">⚠ Above target (0.70)</span>
              )}
            </p>
          </div>
        )}

        {/* Total Steps Logged */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-gray-400">Steps Logged</p>
            <Activity className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-2xl font-bold text-amber-400">{stats.totalSteps}</p>
          <p className="text-xs text-gray-500 mt-1">Total metrics collected</p>
        </div>
      </div>

      {/* Current Training Status (MoE) */}
      {(stats.highNoiseLoss != null || stats.lowNoiseLoss != null) && (
        <div className="bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-xl p-6 border-2 border-purple-600/50">
          <h3 className="text-xl font-semibold mb-4 flex items-center text-purple-300">
            <Layers className="w-6 h-6 mr-2" />
            Currently Training: {currentActiveExpert === 'high_noise' ? 'High Noise Expert' : 'Low Noise Expert'}
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-400">Current Step</p>
              <p className="text-3xl font-bold text-white">{current.step}</p>
              <p className="text-xs text-gray-500 mt-1">Step {stepsInCurrentBlock + 1}/{switchBoundaryEvery} in expert block</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Current Loss</p>
              <p className="text-3xl font-bold text-yellow-400">
                {current.loss != null ? current.loss.toFixed(4) : 'N/A'}
              </p>
              <p className="text-xs text-gray-500 mt-1">This step only</p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Expert Learning Rate</p>
              <p className="text-2xl font-bold text-green-400">
                {currentActiveExpert === 'high_noise'
                  ? (current.lr_0 != null ? current.lr_0.toExponential(2) : 'N/A')
                  : (current.lr_1 != null ? current.lr_1.toExponential(2) : 'N/A')
                }
              </p>
              <p className="text-xs text-gray-500 mt-1">{currentActiveExpert === 'high_noise' ? 'lr_0' : 'lr_1'}</p>
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-950/50 rounded-lg">
            <p className="text-xs text-gray-400">
              💡 MoE switches experts every {switchBoundaryEvery} steps. {currentActiveExpert === 'high_noise' ? 'High Noise' : 'Low Noise'} expert handles
              {currentActiveExpert === 'high_noise' ? ' harder denoising (timesteps 1000-900)' : ' detail refinement (timesteps 900-0)'}.
              Next switch in {switchBoundaryEvery - stepsInCurrentBlock - 1} steps.
            </p>
          </div>
        </div>
      )}

      {/* MoE Expert Comparison (if applicable) */}
      {(stats.highNoiseLoss != null || stats.lowNoiseLoss != null) && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-gray-200">
            <Layers className="w-5 h-5 mr-2 text-purple-400" />
            Historical Averages (Last {windowSize} steps)
          </h3>
          <p className="text-xs text-gray-500 mb-4">These averages include historical data from both experts and update as the window slides. See "Currently Training" above for real-time info.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className={`bg-gradient-to-br from-red-900/20 to-orange-900/20 rounded-lg p-4 border ${currentActiveExpert === 'high_noise' ? 'border-red-500 border-2' : 'border-red-800/30'}`}>
              <p className="text-sm text-gray-400 mb-1 flex items-center">
                High Noise Expert
                {currentActiveExpert === 'high_noise' && <span className="ml-2 px-2 py-0.5 bg-red-500 text-white text-xs rounded">ACTIVE</span>}
              </p>
              <p className="text-xs text-gray-500 mb-2">Timesteps 1000-900 (harder denoising)</p>
              <div className="space-y-2">
                <div>
                  <p className="text-xs text-gray-400">Simple Average</p>
                  <p className="text-2xl font-bold text-red-400">
                    {stats.highNoiseLoss != null ? stats.highNoiseLoss.toFixed(4) : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">EMA (weighted recent)</p>
                  <p className="text-3xl font-bold text-red-300">
                    {stats.highNoiseLossEMA != null ? stats.highNoiseLossEMA.toFixed(4) : 'N/A'}
                  </p>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-2">Window: last {windowSize} steps</p>
            </div>
            <div className={`bg-gradient-to-br from-blue-900/20 to-cyan-900/20 rounded-lg p-4 border ${currentActiveExpert === 'low_noise' ? 'border-blue-500 border-2' : 'border-blue-800/30'}`}>
              <p className="text-sm text-gray-400 mb-1 flex items-center">
                Low Noise Expert
                {currentActiveExpert === 'low_noise' && <span className="ml-2 px-2 py-0.5 bg-blue-500 text-white text-xs rounded">ACTIVE</span>}
              </p>
              <p className="text-xs text-gray-500 mb-2">Timesteps 900-0 (detail refinement)</p>
              <div className="space-y-2">
                <div>
                  <p className="text-xs text-gray-400">Simple Average</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {stats.lowNoiseLoss != null ? stats.lowNoiseLoss.toFixed(4) : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-gray-400">EMA (weighted recent)</p>
                  <p className="text-3xl font-bold text-blue-300">
                    {stats.lowNoiseLossEMA != null ? stats.lowNoiseLossEMA.toFixed(4) : 'N/A'}
                  </p>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-2">Window: last {windowSize} steps</p>
            </div>
          </div>
          {stats.highNoiseLoss != null && stats.lowNoiseLoss != null && (
            <div className="mt-4 p-3 bg-gray-950 rounded-lg">
              <p className="text-sm text-gray-400">
                Loss Ratio: {(stats.highNoiseLoss / stats.lowNoiseLoss).toFixed(2)}x
                {stats.highNoiseLoss > stats.lowNoiseLoss * 1.1 ? (
                  <span className="ml-2 text-green-400">✓ High noise learning harder timesteps (expected)</span>
                ) : (
                  <span className="ml-2 text-amber-400">⚠ Ratio may be unusual (expect high > low)</span>
                )}
              </p>
            </div>
          )}
          <p className="text-xs text-gray-500 mt-4">
            * Note: If expert tracking shows "null", experts are inferred from step alternation pattern.
            This is normal for this training setup.
          </p>
        </div>
      )}

      {/* Loss Trend Indicator */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">Loss Trend Analysis</h3>
        {current.loss_slope != null && current.loss_r2 != null ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Tooltip text="How fast loss is changing. Negative = good (improving). Near zero = plateaued. Positive = getting worse! Based on linear regression of recent loss values.">
                <p className="text-sm text-gray-400">Slope</p>
              </Tooltip>
              <p className={`text-xl font-bold ${current.loss_slope < 0 ? 'text-green-400' : 'text-red-400'}`}>
                {current.loss_slope.toExponential(3)}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {current.loss_slope < 0 ? 'Decreasing ✓' : 'Increasing ⚠'}
              </p>
            </div>
            <div>
              <Tooltip text="How well we can predict your loss trend (0-1 scale). Video: >0.01 is good. Images: >0.1 is good. Lower means very noisy data. Needed to confirm loss has actually plateaued before phase transition.">
                <p className="text-sm text-gray-400">R² (Fit Quality)</p>
              </Tooltip>
              <p className="text-xl font-bold text-purple-400">
                {current.loss_r2.toFixed(6)}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {current.loss_r2 < 0.01 ? 'Very noisy (normal for video)' : 'Smooth convergence'}
              </p>
            </div>
            <div>
              <Tooltip text="Overall training status based on loss slope. Converging = actively improving, Plateaued = ready for phase transition, Training = in progress.">
                <p className="text-sm text-gray-400">Status</p>
              </Tooltip>
              <p className="text-xl font-bold text-blue-400">
                {current.loss_slope < -0.001 ? 'Converging' :
                 Math.abs(current.loss_slope) < 0.0001 ? 'Plateaued' :
                 'Training'}
              </p>
            </div>
          </div>
        ) : (
          <div className="text-gray-400 text-center py-8">
            <p className="text-lg mb-2">Collecting samples... ({current.loss_samples || 0}/20)</p>
            <p className="text-sm">Need 20 loss samples to calculate trend analysis</p>
            <p className="text-xs mt-2 text-gray-500">Loss trends will appear after {20 - (current.loss_samples || 0)} more steps</p>
          </div>
        )}
      </div>
    </div>
  );
}
