'use client';

import { useEffect, useRef, useState } from 'react';
import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react';
import type { Job } from '@prisma/client';
import type { ProcessConfig, SigmoidScheduleRow } from '@/types';
import { apiClient } from '@/utils/api';
import {
  jobScheduleStep,
  bucketLookupStep,
  supportsSigmoidSchedule,
  parseScheduleRows,
  scheduleContext,
  sigmoidJobProcess,
  SIGMOID_SCHEDULE_MESSAGE as message,
} from '@/utils/sigmoidSchedule';
import type { TrainingBucket } from '@/utils/sigmoidSchedule';

interface Props {
  context: ReturnType<typeof scheduleContext>;
  jobId?: string;
  step?: number;
  onApply?: (rows: SigmoidScheduleRow[]) => void;
  onClose?: () => void;
}

export default function TimestepSchedulePlugin({ context, jobId, step, onApply, onClose }: Props) {
  const iframe = useRef<HTMLIFrameElement>(null);
  const [lookup, setLookup] = useState<{ jobId: string; buckets: TrainingBucket[] }>();
  const buckets = lookup?.jobId === jobId ? lookup?.buckets : undefined;
  const hasBuckets = Boolean(buckets?.length);
  // A resumed job can already have a positive step with an empty new log.
  const retryStep = bucketLookupStep(buckets, step);

  useEffect(() => {
    if (!jobId || hasBuckets) return;
    const controller = new AbortController();
    apiClient
      .get<{ buckets: TrainingBucket[] }>(`/api/jobs/${encodeURIComponent(jobId)}/log`, {
        params: { view: 'buckets' },
        signal: controller.signal,
      })
      .then(({ data }) => {
        if (!controller.signal.aborted) setLookup({ jobId, buckets: data.buckets ?? [] });
      })
      .catch(() => {
        if (!controller.signal.aborted) setLookup({ jobId, buckets: [] });
      });
    return () => controller.abort();
  }, [jobId, hasBuckets, retryStep]);

  useEffect(() => {
    function receive(event: MessageEvent) {
      const source = iframe.current?.contentWindow;
      if (!source || event.source !== source || event.origin !== 'null') return;
      if (event.data?.type === message.apply && onApply) {
        const rows = parseScheduleRows(event.data.schedule);
        if (rows) onApply(rows);
      } else if (event.data?.type === message.close && onClose) {
        onClose();
      }
    }
    window.addEventListener('message', receive);
    return () => window.removeEventListener('message', receive);
  }, [onApply, onClose]);

  useEffect(() => {
    if (step !== undefined) {
      iframe.current?.contentWindow?.postMessage({ type: message.progress, step, buckets }, '*');
    }
  }, [step, buckets]);

  if (jobId && !buckets) return <p className="p-4">Loading timestep visualizer…</p>;

  return (
    <iframe
      ref={iframe}
      src="/timestep-schedule/index.html"
      title="Timestep Schedule Visualizer"
      className="block h-full min-h-[300px] w-full border-0"
      sandbox="allow-scripts"
      onLoad={() =>
        iframe.current?.contentWindow?.postMessage(
          {
            type: message.init,
            context: { ...context, buckets: buckets ?? [], step, readOnly: !onApply },
          },
          '*',
        )
      }
    />
  );
}

interface EditProps {
  process: Pick<ProcessConfig, 'train' | 'model'>;
  jobId?: string;
  onApply: (rows: SigmoidScheduleRow[]) => void;
}

export function EditTimestepSchedule({ process, jobId, onApply }: EditProps) {
  const [open, setOpen] = useState(false);
  const supported = supportsSigmoidSchedule(process.train, process.model?.arch);
  const schedule = process.train.timestep_sigmoid_schedule;
  if (!supported || (schedule !== undefined && !parseScheduleRows(schedule))) {
    if (schedule === undefined) return null;
    return (
      <div className="pt-2">
        <p className="pb-1 text-xs text-amber-300">
          {supported
            ? 'Fix the invalid timestep schedule in Advanced YAML or clear it here.'
            : 'This job’s timestep mode bypasses the sigmoid schedule.'}
        </p>
        <button
          type="button"
          className="rounded border border-gray-600 px-3 py-1.5 text-sm hover:bg-gray-700"
          onClick={() => onApply([])}
        >
          Clear Timestep Schedule
        </button>
      </div>
    );
  }
  const close = () => setOpen(false);
  return (
    <div className="pt-2">
      <button
        type="button"
        className="rounded border border-gray-600 px-3 py-1.5 text-sm hover:bg-gray-700"
        onClick={() => setOpen(true)}
      >
        Edit Timestep Schedule
      </button>
      {open && (
        <Dialog open={open} onClose={close} className="relative z-50">
          <div aria-hidden="true" className="fixed inset-0 bg-black/70" />
          <div className="fixed inset-0 flex items-center justify-center p-2 sm:p-4">
            <DialogPanel className="flex h-[94dvh] w-full max-w-7xl flex-col overflow-hidden rounded-md bg-gray-900">
              <div className="flex items-center justify-between px-4 py-2">
                <DialogTitle className="text-base font-semibold">Timestep Schedule</DialogTitle>
                <button type="button" className="rounded px-3 py-1 hover:bg-gray-700" onClick={close}>
                  Close
                </button>
              </div>
              <div className="min-h-0 flex-1">
                <TimestepSchedulePlugin
                  context={scheduleContext(process)}
                  jobId={jobId}
                  onApply={rows => {
                    onApply(rows);
                    close();
                  }}
                  onClose={close}
                />
              </div>
            </DialogPanel>
          </div>
        </Dialog>
      )}
    </div>
  );
}

export function JobTimestepSchedule({ job }: { job: Job }) {
  const process = sigmoidJobProcess(job.job_config);
  if (!process) return null;
  const schedule = process.train.timestep_sigmoid_schedule;
  if (schedule !== undefined && !parseScheduleRows(schedule)) {
    return <p className="p-4">This job has an invalid timestep schedule. Check its Advanced YAML.</p>;
  }
  return (
    <div className="h-[calc(100dvh-80px)]">
      <TimestepSchedulePlugin
        context={scheduleContext(process)}
        jobId={job.id}
        step={jobScheduleStep(job, process.train.steps)}
      />
    </div>
  );
}
