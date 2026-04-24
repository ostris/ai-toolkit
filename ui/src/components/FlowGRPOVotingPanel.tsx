'use client';

import { type CSSProperties, useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { Button } from '@headlessui/react';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import useFlowGRPOVoteTasks from '@/hooks/useFlowGRPOVoteTasks';
import { FlowGRPOLiveTaskConfig, JobConfig } from '@/types';

type Props = {
  job: Job;
  compact?: boolean;
  limit?: number;
};

const getTaskDefaults = (job: Job): FlowGRPOLiveTaskConfig => {
  const jobConfig = JSON.parse(job.job_config) as JobConfig;
  const process = jobConfig.config.process[0];
  return {
    prompt: '',
    negative_prompt: process.sample?.neg || '',
    requested_candidates: Math.max(2, process.grpo?.candidates_per_task || 4),
    width: process.sample?.width || 1024,
    height: process.sample?.height || 1024,
    seed: process.sample?.seed ?? null,
    guidance_scale: process.sample?.guidance_scale || 4,
    num_inference_steps: process.sample?.sample_steps || 30,
    sampler: process.sample?.sampler || 'flowmatch',
    scheduler: process.train?.noise_scheduler || 'flowmatch',
  };
};

const statusLabels: Record<string, string> = {
  requested: 'Queued',
  generating: 'Generating',
  open: 'Ready For Vote',
  voted: 'Applying Vote',
  processed: 'Processed',
  skipped: 'Skipped',
  failed: 'Failed',
};

type FlowGRPOLiveTaskDraft = Omit<
  FlowGRPOLiveTaskConfig,
  'requested_candidates' | 'width' | 'height' | 'seed' | 'guidance_scale' | 'num_inference_steps'
> & {
  requested_candidates: string;
  width: string;
  height: string;
  seed: string;
  guidance_scale: string;
  num_inference_steps: string;
};

const getTaskDraftDefaults = (job: Job): FlowGRPOLiveTaskDraft => {
  const defaults = getTaskDefaults(job);
  return {
    ...defaults,
    requested_candidates: `${defaults.requested_candidates}`,
    width: `${defaults.width}`,
    height: `${defaults.height}`,
    seed: defaults.seed == null ? '' : `${defaults.seed}`,
    guidance_scale: `${defaults.guidance_scale}`,
    num_inference_steps: `${defaults.num_inference_steps}`,
  };
};

const parseIntegerField = (value: string, fallback: number, min: number) => {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(min, parsed) : fallback;
};

const parseFloatField = (value: string, fallback: number, min: number) => {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? Math.max(min, parsed) : fallback;
};

const getTaskStatusMessage = (task: { status: string; error?: string | null }) => {
  if (task.status === 'requested') return 'Waiting for the trainer to pick up this task.';
  if (task.status === 'generating') return 'The trainer is generating candidates for this task.';
  if (task.status === 'voted') return 'Vote received. The trainer is applying the Flow-GRPO update.';
  if (task.status === 'processed') return 'Vote processed.';
  if (task.status === 'skipped') return 'Task skipped.';
  if (task.status === 'failed') return task.error ? `Task failed: ${task.error}` : 'Task failed.';
  return statusLabels[task.status] || task.status;
};

export default function FlowGRPOVotingPanel({ job, compact = false, limit = 5 }: Props) {
  const { tasks, status, refreshTasks } = useFlowGRPOVoteTasks(job.id, 3000, limit);
  const [submittingTaskId, setSubmittingTaskId] = useState<string | null>(null);
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [taskDraft, setTaskDraft] = useState<FlowGRPOLiveTaskDraft>(() => getTaskDraftDefaults(job));
  const taskDraftJobId = useRef(job.id);

  useEffect(() => {
    if (taskDraftJobId.current !== job.id) {
      taskDraftJobId.current = job.id;
      setTaskDraft(getTaskDraftDefaults(job));
    }
  }, [job]);

  const updateDraft = <K extends keyof FlowGRPOLiveTaskDraft>(key: K, value: FlowGRPOLiveTaskDraft[K]) => {
    setTaskDraft(current => ({ ...current, [key]: value }));
  };

  const submitVote = async (taskID: string, payload: { action: 'select' | 'skip'; selectedCandidateId?: string }) => {
    if (submittingTaskId) return;
    setSubmittingTaskId(taskID);
    try {
      await apiClient.post(`/api/grpo/jobs/${job.id}/tasks/${taskID}/vote`, payload);
      refreshTasks();
    } catch (error) {
      console.error('Error submitting Flow-GRPO vote:', error);
    } finally {
      setSubmittingTaskId(null);
    }
  };

  const createTask = async () => {
    if (isCreatingTask || !taskDraft.prompt.trim()) return;
    const defaults = getTaskDefaults(job);
    const seedValue = taskDraft.seed.trim();
    const parsedSeed = seedValue === '' ? null : parseInt(seedValue, 10);
    const payload: FlowGRPOLiveTaskConfig = {
      prompt: taskDraft.prompt,
      negative_prompt: taskDraft.negative_prompt || '',
      requested_candidates: parseIntegerField(taskDraft.requested_candidates, defaults.requested_candidates, 2),
      width: parseIntegerField(taskDraft.width, defaults.width, 64),
      height: parseIntegerField(taskDraft.height, defaults.height, 64),
      seed: Number.isFinite(parsedSeed) ? parsedSeed : null,
      guidance_scale: parseFloatField(taskDraft.guidance_scale, defaults.guidance_scale, 0),
      num_inference_steps: parseIntegerField(taskDraft.num_inference_steps, defaults.num_inference_steps, 1),
      sampler: taskDraft.sampler,
      scheduler: taskDraft.scheduler,
    };
    setIsCreatingTask(true);
    try {
      await apiClient.post(`/api/grpo/jobs/${job.id}/tasks`, payload);
      setTaskDraft(current => ({ ...current, prompt: '' }));
      refreshTasks();
    } catch (error) {
      console.error('Error creating Flow-GRPO vote task:', error);
    } finally {
      setIsCreatingTask(false);
    }
  };

  const title = compact ? `Flow-GRPO: ${job.name}` : 'Live Voting';
  const candidateGridStyle: CSSProperties = {
    gridTemplateColumns: compact
      ? 'repeat(auto-fill, minmax(180px, 1fr))'
      : 'repeat(auto-fill, minmax(220px, 1fr))',
  };

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-gray-800 bg-gray-900 p-4">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-gray-100">{title}</h2>
            <p className="text-sm text-gray-400">
              {tasks.length > 0 ? `${tasks.length} active Flow-GRPO task${tasks.length === 1 ? '' : 's'}` : 'No active Flow-GRPO tasks'}
            </p>
          </div>
          <Button
            onClick={refreshTasks}
            className="rounded-md bg-gray-800 px-3 py-1 text-sm text-gray-200 hover:bg-gray-700"
          >
            Refresh
          </Button>
        </div>
      </div>

      <div className={classNames('grid gap-4', !compact && 'lg:grid-cols-12 lg:items-start')}>
        {!compact && (
          <div className="lg:col-span-4">
            <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-gray-100">Create Live Task</h3>
                  <p className="mt-1 text-xs text-gray-400">
                    Prompts and sampling parameters are entered at vote time, not in the saved job config.
                  </p>
                </div>

                <div className="space-y-3">
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Prompt</label>
                    <textarea
                      value={taskDraft.prompt}
                      onChange={event => updateDraft('prompt', event.target.value)}
                      rows={4}
                      className="min-h-28 w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                      placeholder="Describe the sample request for this Flow-GRPO round"
                    />
                  </div>

                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Negative Prompt</label>
                    <textarea
                      value={taskDraft.negative_prompt || ''}
                      onChange={event => updateDraft('negative_prompt', event.target.value)}
                      rows={3}
                      className="min-h-20 w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                      placeholder="Optional negative prompt"
                    />
                  </div>
                </div>

                <div className="grid gap-3 grid-cols-2">
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Candidates</label>
                    <input
                      type="number"
                      min={2}
                      value={taskDraft.requested_candidates}
                      onChange={event => updateDraft('requested_candidates', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Seed</label>
                    <input
                      type="number"
                      value={taskDraft.seed}
                      onChange={event => updateDraft('seed', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                      placeholder="Optional"
                    />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Width</label>
                    <input
                      type="number"
                      min={64}
                      value={taskDraft.width}
                      onChange={event => updateDraft('width', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Height</label>
                    <input
                      type="number"
                      min={64}
                      value={taskDraft.height}
                      onChange={event => updateDraft('height', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Guidance Scale</label>
                    <input
                      type="number"
                      min={0}
                      step="0.1"
                      value={taskDraft.guidance_scale}
                      onChange={event => updateDraft('guidance_scale', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Sample Steps</label>
                    <input
                      type="number"
                      min={1}
                      value={taskDraft.num_inference_steps}
                      onChange={event => updateDraft('num_inference_steps', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Sampler</label>
                    <select
                      value={taskDraft.sampler}
                      onChange={event => updateDraft('sampler', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    >
                      <option value="flowmatch">FlowMatch</option>
                    </select>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Scheduler</label>
                    <select
                      value={taskDraft.scheduler || ''}
                      onChange={event => updateDraft('scheduler', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    >
                      <option value="flowmatch">FlowMatch</option>
                    </select>
                  </div>
                </div>

                <Button
                  onClick={createTask}
                  disabled={isCreatingTask || !taskDraft.prompt.trim()}
                  className={classNames(
                    'w-full rounded-md px-4 py-2.5 text-sm font-medium',
                    isCreatingTask || !taskDraft.prompt.trim()
                      ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                      : 'bg-blue-600 text-white hover:bg-blue-500',
                  )}
                >
                  Queue Flow-GRPO Task
                </Button>
              </div>
            </div>
          </div>
        )}

        <div className={classNames('space-y-4', !compact && 'lg:col-span-8')}>
          {status === 'loading' && tasks.length === 0 && (
            <div className="rounded-xl border border-dashed border-gray-700 bg-gray-900/60 p-6 text-sm text-gray-400">
              Loading Flow-GRPO tasks...
            </div>
          )}

          {status === 'error' && tasks.length === 0 && (
            <div className="rounded-xl border border-dashed border-red-900 bg-red-950/20 p-6 text-sm text-red-300">
              Failed to load Flow-GRPO tasks.
            </div>
          )}

          {status !== 'error' && tasks.length === 0 && (
            <div className="rounded-xl border border-dashed border-gray-700 bg-gray-900/60 p-6 text-sm text-gray-400">
              No live tasks yet. Queue one from the job detail page to start a Flow-GRPO round.
            </div>
          )}

          <div className="space-y-4">
        {tasks.map(task => {
          const canVote = task.status === 'open' && task.candidates.length > 0;
          return (
            <div key={task.id} className="rounded-xl border border-gray-800 bg-gray-950 p-4">
              <div className={classNames('grid gap-4', !compact && 'sm:grid-cols-12')}>
                <div className={classNames('space-y-4', !compact && 'sm:col-span-4 lg:col-span-3')}>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="text-xs uppercase tracking-wide text-gray-500">Prompt</div>
                      <div className="mt-1 text-sm text-gray-100">{task.prompt}</div>
                      {task.negative_prompt && (
                        <div className="mt-2 text-xs text-gray-500">Negative: {task.negative_prompt}</div>
                      )}
                    </div>
                    <div
                      className={classNames(
                        'rounded-full px-3 py-1 text-xs font-medium',
                        task.status === 'open' && 'bg-blue-950 text-blue-200',
                        task.status === 'generating' && 'bg-amber-950 text-amber-200',
                        task.status === 'requested' && 'bg-gray-800 text-gray-200',
                        task.status === 'voted' && 'bg-emerald-950 text-emerald-200',
                        task.status === 'failed' && 'bg-red-950 text-red-200',
                      )}
                    >
                      {statusLabels[task.status] || task.status}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                    <div>Candidates: {task.requested_candidates}</div>
                    <div>Size: {task.width && task.height ? `${task.width}x${task.height}` : 'default'}</div>
                    <div>Seed: {task.seed ?? 'auto'}</div>
                    <div>CFG: {task.guidance_scale ?? 'default'}</div>
                    <div>Steps: {task.num_inference_steps ?? 'default'}</div>
                    <div>Sampler: {task.sampler || 'default'}</div>
                    <div>Scheduler: {task.scheduler || 'default'}</div>
                    <div>Status: {statusLabels[task.status] || task.status}</div>
                  </div>

                  {canVote && (
                    <Button
                      onClick={() => submitVote(task.id, { action: 'skip' })}
                      disabled={submittingTaskId === task.id}
                      className={classNames(
                        'w-full rounded-md px-3 py-2 text-sm',
                        submittingTaskId === task.id
                          ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                          : 'bg-gray-800 text-gray-200 hover:bg-gray-700',
                      )}
                    >
                      Skip Task
                    </Button>
                  )}
                </div>

                <div className={classNames(!compact && 'sm:col-span-8 lg:col-span-9')}>
                  {!canVote && (
                    <div className="rounded-lg border border-dashed border-gray-800 bg-gray-900/60 p-4 text-sm text-gray-400">
                      {getTaskStatusMessage(task)}
                    </div>
                  )}

                  {canVote && (
                    <div className="grid gap-3" style={candidateGridStyle}>
                      {task.candidates.map(candidate => (
                        <div key={candidate.id} className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900">
                          <img src={candidate.image_url} alt={candidate.prompt} className="aspect-square w-full bg-black object-cover" />
                          <div className="space-y-3 p-3">
                            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-gray-400">
                              <div>Seed: {candidate.seed}</div>
                              <div>CFG: {candidate.guidance_scale}</div>
                              <div>Steps: {candidate.num_inference_steps}</div>
                              <div>Sampler: {candidate.sampler}</div>
                            </div>
                            <Button
                              onClick={() => submitVote(task.id, { action: 'select', selectedCandidateId: candidate.id })}
                              disabled={submittingTaskId === task.id}
                              className={classNames(
                                'w-full rounded-md px-3 py-2 text-sm font-medium',
                                submittingTaskId === task.id
                                  ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                                  : 'bg-blue-600 text-white hover:bg-blue-500',
                              )}
                            >
                              Prefer This Candidate
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
          </div>
        </div>
      </div>
    </div>
  );
}
