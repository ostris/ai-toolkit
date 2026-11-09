import type { ProcessConfig, SigmoidScheduleRow, TrainConfig } from '@/types';

export const SIGMOID_SCHEDULE_MESSAGE = {
  init: 'aitk-timestep-schedule:init',
  progress: 'aitk-timestep-schedule:progress',
  apply: 'aitk-timestep-schedule:apply',
  close: 'aitk-timestep-schedule:close',
} as const;

export interface TrainingBucket {
  width: number;
  height: number;
  frames?: number;
}

export function bucketLookupStep(buckets: TrainingBucket[] | undefined, step: number | undefined) {
  return buckets?.length ? undefined : step;
}

export function parseScheduleRows(value: unknown): SigmoidScheduleRow[] | null {
  if (!Array.isArray(value)) return null;
  const rows: SigmoidScheduleRow[] = [];
  let previous = -1;
  for (const row of value) {
    if (!row || typeof row !== 'object') return null;
    if (Object.keys(row).some(key => key !== 'start_step' && key !== 'shift' && key !== 'scale')) return null;
    const { start_step, shift, scale } = row;
    if (
      !Number.isSafeInteger(start_step) ||
      start_step <= previous ||
      !Number.isFinite(shift) ||
      !Number.isFinite(scale) ||
      scale <= 0
    )
      return null;
    rows.push({ start_step, shift, scale });
    previous = start_step;
  }
  return rows;
}

export function trainingBucketsFromLog(log: string): TrainingBucket[] {
  const buckets = new Map<string, TrainingBucket>();
  let inBucketBlock = false;
  for (const line of log.split(/[\r\n]+/)) {
    if (line.startsWith('Bucket sizes for ') && line.endsWith(':')) {
      inBucketBlock = true;
      continue;
    }
    if (!inBucketBlock) continue;
    const match = /^(\d+)x(\d+)(?:x(\d+)f)?(?:_[^:]+)?:\s+\d+\s+files?\b/.exec(line);
    if (!match) {
      inBucketBlock = false;
      continue;
    }
    const width = Number(match[1]);
    const height = Number(match[2]);
    const frames = match[3] ? Number(match[3]) : undefined;
    if (![width, height, frames ?? 1].every(value => Number.isSafeInteger(value) && value > 0)) continue;
    buckets.set(`${width}x${height}x${frames ?? ''}`, { width, height, ...(frames ? { frames } : {}) });
  }
  return [...buckets.values()];
}

type ScheduleMode = Partial<Pick<TrainConfig, 'timestep_type' | 'noise_scheduler'>> & {
  linear_timesteps?: boolean;
  linear_timesteps2?: boolean;
};

export function supportsSigmoidSchedule(train?: ScheduleMode, arch?: string) {
  if (train?.noise_scheduler !== 'flowmatch' || arch?.split(':')[0] === 'minimax_h3_image') return false;
  const { timestep_type = 'sigmoid', linear_timesteps, linear_timesteps2 } = train;
  return timestep_type === 'sigmoid' && !linear_timesteps && !linear_timesteps2;
}

export function sigmoidJobProcess(config: string): ProcessConfig | undefined {
  try {
    const process = JSON.parse(config)?.config?.process?.[0];
    return supportsSigmoidSchedule(process?.train, process?.model?.arch) ? process : undefined;
  } catch {
    return undefined;
  }
}

// Mirror the legacy flag precedence in Toolkit's ModelConfig.
const legacyArchitectures = [
  ['is_v2', 'sd2'],
  ['is_v3', 'sd3'],
  ['is_xl', 'sdxl'],
  ['is_pixart', 'pixart'],
  ['is_pixart_sigma', 'pixart_sigma'],
  ['is_auraflow', 'auraflow'],
  ['is_flux', 'flux'],
  ['is_lumina2', 'lumina2'],
  ['is_vega', 'vega'],
  ['is_ssd', 'ssd'],
] as const;

function modelArchitecture(model?: ProcessConfig['model']) {
  if (model?.arch) return model.arch === 'flex1' ? 'flux' : model.arch;
  if (!model) return undefined;
  return legacyArchitectures.find(([flag]) => Reflect.get(model, flag))?.[1] ?? 'sd1';
}

export function scheduleContext({ train, model }: Pick<ProcessConfig, 'train' | 'model'>) {
  const { steps, timestep_shift, timestep_scale, timestep_sigmoid_schedule, content_or_style } = train;
  return {
    train: { steps, timestep_shift, timestep_scale, timestep_sigmoid_schedule, content_or_style },
    model: { arch: modelArchitecture(model) },
  };
}

export function jobScheduleStep(job: { step: number; status: string }, steps: number) {
  return job.status === 'completed' && steps > 0 ? Math.min(job.step, steps - 1) : job.step;
}
