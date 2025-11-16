'use client';

import ComprehensiveWizard from './wizard/ComprehensiveWizard';
import { JobConfig } from '@/types';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  onExit: () => void;
};

/**
 * GuidedWizard - Comprehensive 12-step configuration wizard
 *
 * Features:
 * - Pre-flight hardware detection (GPU, RAM, storage)
 * - User intent questionnaire (training type, priority, experience level)
 * - Smart defaults engine (batch size, caching, prefetching, workers)
 * - Real-time advisor panel with educational content
 * - Performance predictions (VRAM, time, disk space)
 * - Support for unified memory (Apple Silicon, AMD ROCm)
 */
export default function GuidedWizard(props: Props) {
  return <ComprehensiveWizard {...props} />;
}
