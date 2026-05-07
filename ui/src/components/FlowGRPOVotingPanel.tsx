'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { Button } from '@headlessui/react';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import useFlowGRPOVoteTasks from '@/hooks/useFlowGRPOVoteTasks';
import { FlowGRPOLiveTaskConfig, JobConfig } from '@/types';

type Props = {
  job: Job;
  compact?: boolean;
};

const getTaskDefaults = (job: Job): FlowGRPOLiveTaskConfig => {
  const jobConfig = JSON.parse(job.job_config) as JobConfig;
  const process = jobConfig.config.process[0];
  return {
    prompt: '',
    negative_prompt: process.sample?.neg || '',
    width: process.sample?.width || 1024,
    height: process.sample?.height || 1024,
    seed: process.sample?.seed ?? null,
    guidance_scale: process.sample?.guidance_scale || 4,
    num_inference_steps: process.sample?.sample_steps || 30,
    sampler: process.sample?.sampler || 'flowmatch_step_with_logprob',
    scheduler: process.train?.noise_scheduler || 'flowmatch_step_with_logprob',
  };
};

const statusLabels: Record<string, string> = {
  requested: 'Queued',
  generating: 'Generating',
  open: 'Ready For Vote',
  voted: 'Applying Vote',
  processed: 'Processed',
  stale: 'Stale',
  failed: 'Failed',
};

const voteOptions = [
  { value: 'up', label: 'Up', reward: 1, className: 'bg-emerald-600 text-white hover:bg-emerald-500' },
  { value: 'down', label: 'Down', reward: -1, className: 'bg-red-600 text-white hover:bg-red-500' },
  { value: 'skip', label: 'Skip', reward: 0, className: 'bg-gray-800 text-gray-200 hover:bg-gray-700' },
];

type FlowGRPOLiveTaskDraft = Omit<
  FlowGRPOLiveTaskConfig,
  'width' | 'height' | 'seed' | 'guidance_scale' | 'num_inference_steps'
> & {
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
  if (task.status === 'generating') return 'The trainer is generating a rollout for this task.';
  if (task.status === 'open') return 'Waiting for a complete Flow-GRPO candidate group.';
  if (task.status === 'voted') return 'Vote received. The trainer will apply this Flow-GRPO group.';
  if (task.status === 'processed') return 'Vote processed.';
  if (task.status === 'stale') return 'Task skipped because rollout trajectories became stale.';
  if (task.status === 'failed') return task.error ? `Task failed: ${task.error}` : 'Task failed.';
  return statusLabels[task.status] || task.status;
};

export default function FlowGRPOVotingPanel({ job, compact = false }: Props) {
  const { tasks, status, refreshTasks } = useFlowGRPOVoteTasks(job.id, 3000);
  const [submittingTaskId, setSubmittingTaskId] = useState<string | null>(null);
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [candidateVotes, setCandidateVotes] = useState<Record<string, Record<string, string>>>({});
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

  const setCandidateVote = (taskID: string, candidateID: string, value: string) => {
    setCandidateVotes(current => ({
      ...current,
      [taskID]: {
        ...(current[taskID] || {}),
        [candidateID]: value,
      },
    }));
  };

  const submitVote = async (taskID: string, candidateIDs: string[]) => {
    if (submittingTaskId) return;
    const taskVotes = candidateVotes[taskID] || {};
    if (candidateIDs.some(candidateID => !taskVotes[candidateID])) return;
    setSubmittingTaskId(taskID);
    try {
      await apiClient.post(`/api/grpo/jobs/${job.id}/tasks/${taskID}/vote`, {
        rewards: candidateIDs.map(candidateID => ({
          candidate_id: candidateID,
          value: taskVotes[candidateID],
        })),
      });
      setCandidateVotes(current => {
        const next = { ...current };
        delete next[taskID];
        return next;
      });
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
                    Prompts and generation settings are entered live; group size remains a job-level training setting.
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
                      <option value="flowmatch_step_with_logprob">FlowMatch Step With LogProb</option>
                    </select>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Scheduler</label>
                    <select
                      value={taskDraft.scheduler || ''}
                      onChange={event => updateDraft('scheduler', event.target.value)}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    >
                      <option value="flowmatch_step_with_logprob">FlowMatch Step With LogProb</option>
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
              No live tasks yet. Queue one from the job detail page to start a Flow-GRPO rollout.
            </div>
          )}

          <div className="space-y-4">
            {tasks.map(task => {
              const canVote = task.status === 'open' && task.candidates.length > 1;
              const taskVotes = candidateVotes[task.id] || {};
              const candidateIDs = task.candidates.map(candidate => candidate.id);
              const allCandidatesVoted = canVote && candidateIDs.every(candidateID => taskVotes[candidateID]);
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
                            task.status === 'stale' && 'bg-orange-950 text-orange-200',
                            task.status === 'failed' && 'bg-red-950 text-red-200',
                          )}
                        >
                          {statusLabels[task.status] || task.status}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                        <div>Size: {task.width && task.height ? `${task.width}x${task.height}` : 'default'}</div>
                        <div>Seed: {task.seed ?? 'auto'}</div>
                        <div>CFG: {task.guidance_scale ?? 'default'}</div>
                        <div>Steps: {task.num_inference_steps ?? 'default'}</div>
                        <div>Sampler: {task.sampler || 'default'}</div>
                        <div>Scheduler: {task.scheduler || 'default'}</div>
                      </div>

                      {canVote && (
                        <Button
                          onClick={() => submitVote(task.id, candidateIDs)}
                          disabled={!allCandidatesVoted || submittingTaskId === task.id}
                          className={classNames(
                            'w-full rounded-md px-3 py-2 text-sm font-medium',
                            !allCandidatesVoted || submittingTaskId === task.id
                              ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                              : 'bg-blue-600 text-white hover:bg-blue-500',
                          )}
                        >
                          Submit Group Vote
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
                        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                          {task.candidates.map(candidate => (
                            <div key={candidate.id} className="overflow-hidden rounded-lg border border-gray-800 bg-gray-900">
                              <img src={candidate.image_url} alt={candidate.prompt} className="aspect-square w-full bg-black object-cover" />
                              <div className="space-y-3 p-3">
                                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-gray-400">
                                  <div>Seed: {candidate.seed}</div>
                                  <div>CFG: {candidate.guidance_scale}</div>
                                  <div>Steps: {candidate.num_inference_steps}</div>
                                  <div>Sampler: {candidate.sampler}</div>
                                </div>
                                <div className="grid grid-cols-3 gap-1.5">
                                  {voteOptions.map(option => {
                                    const isSelected = taskVotes[candidate.id] === option.value;
                                    return (
                                      <Button
                                        key={option.value}
                                        onClick={() => setCandidateVote(task.id, candidate.id, option.value)}
                                        disabled={submittingTaskId === task.id}
                                        title={`${option.label}: ${option.reward}`}
                                        className={classNames(
                                          'rounded-md px-2 py-1.5 text-xs font-medium',
                                          submittingTaskId === task.id
                                            ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                                            : isSelected
                                              ? option.className
                                              : 'bg-gray-800 text-gray-300 hover:bg-gray-700',
                                        )}
                                      >
                                        {option.label}
                                      </Button>
                                    );
                                  })}
                                </div>
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
