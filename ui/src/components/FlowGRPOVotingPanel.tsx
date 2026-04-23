'use client';

import { useEffect, useState } from 'react';
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

export default function FlowGRPOVotingPanel({ job, compact = false, limit = 5 }: Props) {
  const { tasks, status, refreshTasks } = useFlowGRPOVoteTasks(job.id, 3000, limit);
  const [submittingTaskId, setSubmittingTaskId] = useState<string | null>(null);
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [taskDraft, setTaskDraft] = useState<FlowGRPOLiveTaskConfig>(() => getTaskDefaults(job));

  useEffect(() => {
    setTaskDraft(getTaskDefaults(job));
  }, [job.id, job.job_config]);

  const updateDraft = <K extends keyof FlowGRPOLiveTaskConfig>(key: K, value: FlowGRPOLiveTaskConfig[K]) => {
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
    setIsCreatingTask(true);
    try {
      await apiClient.post(`/api/grpo/jobs/${job.id}/tasks`, taskDraft);
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

      {!compact && (
        <div className="rounded-xl border border-gray-800 bg-gray-950 p-4">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-gray-100">Create Live Task</h3>
            <p className="mt-1 text-sm text-gray-400">
              Prompts and sampling parameters are entered here at vote time, not in the saved job config.
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="mb-2 block text-sm text-gray-300">Prompt</label>
              <textarea
                value={taskDraft.prompt}
                onChange={event => updateDraft('prompt', event.target.value)}
                rows={3}
                className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                placeholder="Describe the sample request for this Flow-GRPO round"
              />
            </div>

            <div>
              <label className="mb-2 block text-sm text-gray-300">Negative Prompt</label>
              <textarea
                value={taskDraft.negative_prompt || ''}
                onChange={event => updateDraft('negative_prompt', event.target.value)}
                rows={2}
                className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                placeholder="Optional negative prompt"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div>
                <label className="mb-2 block text-sm text-gray-300">Candidates</label>
                <input
                  type="number"
                  min={2}
                  value={taskDraft.requested_candidates}
                  onChange={event => updateDraft('requested_candidates', Math.max(2, parseInt(event.target.value || '2', 10)))}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Width</label>
                <input
                  type="number"
                  min={64}
                  value={taskDraft.width}
                  onChange={event => updateDraft('width', Math.max(64, parseInt(event.target.value || '64', 10)))}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Height</label>
                <input
                  type="number"
                  min={64}
                  value={taskDraft.height}
                  onChange={event => updateDraft('height', Math.max(64, parseInt(event.target.value || '64', 10)))}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Seed</label>
                <input
                  type="number"
                  value={taskDraft.seed ?? ''}
                  onChange={event => {
                    const value = event.target.value.trim();
                    updateDraft('seed', value === '' ? null : parseInt(value, 10));
                  }}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                  placeholder="Optional"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Guidance Scale</label>
                <input
                  type="number"
                  min={0}
                  step="0.1"
                  value={taskDraft.guidance_scale}
                  onChange={event => updateDraft('guidance_scale', Math.max(0, parseFloat(event.target.value || '0')))}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Sample Steps</label>
                <input
                  type="number"
                  min={1}
                  value={taskDraft.num_inference_steps}
                  onChange={event => updateDraft('num_inference_steps', Math.max(1, parseInt(event.target.value || '1', 10)))}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Sampler</label>
                <select
                  value={taskDraft.sampler}
                  onChange={event => updateDraft('sampler', event.target.value)}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                >
                  <option value="flowmatch">FlowMatch</option>
                  <option value="ddpm">DDPM</option>
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm text-gray-300">Scheduler</label>
                <input
                  type="text"
                  value={taskDraft.scheduler || ''}
                  onChange={event => updateDraft('scheduler', event.target.value)}
                  className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                />
              </div>
            </div>

            <div className="flex justify-end">
              <Button
                onClick={createTask}
                disabled={isCreatingTask || !taskDraft.prompt.trim()}
                className={classNames(
                  'rounded-md px-4 py-2 text-sm font-medium',
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

      {tasks.map(task => {
        const canVote = task.status === 'open' && task.candidates.length > 0;
        return (
          <div key={task.id} className="rounded-xl border border-gray-800 bg-gray-950 p-4">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
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

            <div className="mb-4 grid grid-cols-2 gap-2 text-xs text-gray-400 md:grid-cols-4 xl:grid-cols-8">
              <div>Candidates: {task.requested_candidates}</div>
              <div>Size: {task.width && task.height ? `${task.width}x${task.height}` : 'default'}</div>
              <div>Seed: {task.seed ?? 'auto'}</div>
              <div>CFG: {task.guidance_scale ?? 'default'}</div>
              <div>Steps: {task.num_inference_steps ?? 'default'}</div>
              <div>Sampler: {task.sampler || 'default'}</div>
              <div>Scheduler: {task.scheduler || 'default'}</div>
              <div>Status: {statusLabels[task.status] || task.status}</div>
            </div>

            {!canVote && (
              <div className="rounded-lg border border-dashed border-gray-800 bg-gray-900/60 p-4 text-sm text-gray-400">
                {task.status === 'requested' && 'Waiting for the trainer to pick up this task.'}
                {task.status === 'generating' && 'The trainer is generating candidates for this task.'}
                {task.status === 'voted' && 'Vote received. The trainer is applying the Flow-GRPO update.'}
                {task.status === 'failed' && task.error ? `Task failed: ${task.error}` : 'Task failed.'}
              </div>
            )}

            {canVote && (
              <>
                <div
                  className={classNames(
                    'grid gap-4',
                    compact ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1 xl:grid-cols-2',
                  )}
                >
                  {task.candidates.map(candidate => (
                    <div key={candidate.id} className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900">
                      <img src={candidate.image_url} alt={candidate.prompt} className="h-auto w-full bg-black object-cover" />
                      <div className="space-y-3 p-4">
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
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

                <div className="mt-4 flex justify-end">
                  <Button
                    onClick={() => submitVote(task.id, { action: 'skip' })}
                    disabled={submittingTaskId === task.id}
                    className={classNames(
                      'rounded-md px-3 py-2 text-sm',
                      submittingTaskId === task.id
                        ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                        : 'bg-gray-800 text-gray-200 hover:bg-gray-700',
                    )}
                  >
                    Skip Task
                  </Button>
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
