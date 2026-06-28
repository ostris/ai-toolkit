'use client';

import { useEffect, useMemo, useState } from 'react';
import { Job } from '@prisma/client';
import { Button } from '@headlessui/react';
import classNames from 'classnames';
import { FaDice } from 'react-icons/fa';
import { EyeOff } from 'lucide-react';
import { apiClient } from '@/utils/api';
import useFlowGRPOVoteTasks, { FlowGRPOVoteTaskView } from '@/hooks/useFlowGRPOVoteTasks';
import { FlowGRPOLiveTaskConfig, JobConfig } from '@/types';
import SampleControlImage from '@/components/SampleControlImage';
import { getVotingInputImageMode } from '@/utils/modelCapabilities';

type Props = {
  job: Job;
  compact?: boolean;
};

type SeedMode = 'fixed' | 'random';

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
  { value: 'up', label: 'Up', className: 'bg-emerald-600 text-white hover:bg-emerald-500' },
  { value: 'down', label: 'Down', className: 'bg-red-600 text-white hover:bg-red-500' },
  { value: 'skip', label: 'Skip', className: 'bg-amber-700 text-white hover:bg-amber-600' },
];

const voteDisplay: Record<string, { label: string; className: string }> = {
  up: { label: 'Up', className: 'bg-emerald-600 text-white' },
  down: { label: 'Down', className: 'bg-red-600 text-white' },
  skip: { label: 'Skip', className: 'bg-amber-700 text-white' },
  reward: { label: 'Custom', className: 'bg-gray-700 text-gray-100' },
};

const activeTaskStatuses = new Set(['requested', 'generating', 'open', 'voted']);

type FlowGRPOLiveTaskDraft = Omit<
  FlowGRPOLiveTaskConfig,
  'width' | 'height' | 'seed' | 'guidance_scale' | 'num_inference_steps'
> & {
  width: string;
  height: string;
  seed: string;
  guidance_scale: string;
  num_inference_steps: string;
  seed_mode: SeedMode;
  ctrl_img_1: string | null;
  ctrl_img_2: string | null;
  ctrl_img_3: string | null;
};

const parseIntegerField = (value: string, fallback: number, min: number) => {
  const parsed = parseInt(value, 10);
  return Number.isFinite(parsed) ? Math.max(min, parsed) : fallback;
};

const parseFloatField = (value: string, fallback: number, min: number) => {
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? Math.max(min, parsed) : fallback;
};

const randomSeed = () => Math.floor(Math.random() * 2147483647);

const getTaskDefaults = (job: Job): FlowGRPOLiveTaskConfig => {
  const jobConfig = JSON.parse(job.job_config) as JobConfig;
  const process = jobConfig.config.process[0];
  return {
    prompt: '',
    negative_prompt: process.sample?.neg || '',
    width: process.sample?.width || 1024,
    height: process.sample?.height || 1024,
    seed: process.sample?.seed ?? null,
    ctrl_img: process.sample?.ctrl_img || null,
    ctrl_img_1: process.sample?.ctrl_img_1 || process.sample?.ctrl_img || null,
    ctrl_img_2: process.sample?.ctrl_img_2 || null,
    ctrl_img_3: process.sample?.ctrl_img_3 || null,
    guidance_scale: process.sample?.guidance_scale || 4,
    num_inference_steps: process.sample?.sample_steps || 30,
    sampler: process.sample?.sampler || 'flowmatch_step_with_logprob',
    scheduler: process.train?.noise_scheduler || 'flowmatch_step_with_logprob',
  };
};

const getTaskDraftDefaults = (job: Job): FlowGRPOLiveTaskDraft => {
  const defaults = getTaskDefaults(job);
  const defaultSeedValue =
    defaults.seed == null ? `${randomSeed()}` : `${defaults.seed}`;
  return {
    ...defaults,
    width: `${defaults.width}`,
    height: `${defaults.height}`,
    seed: defaultSeedValue,
    guidance_scale: `${defaults.guidance_scale}`,
    num_inference_steps: `${defaults.num_inference_steps}`,
    seed_mode: defaults.seed == null ? 'random' : 'fixed',
    ctrl_img_1: defaults.ctrl_img_1 || defaults.ctrl_img || null,
    ctrl_img_2: null,
    ctrl_img_3: null,
  };
};

const getTaskStatusMessage = (task: { status: string; error?: string | null }) => {
  if (task.status === 'requested') return 'Waiting for the trainer to pick up this task.';
  if (task.status === 'generating') return 'Generating candidates...';
  if (task.status === 'open') return 'Waiting for votes.';
  if (task.status === 'voted') return 'Applying vote...';
  if (task.status === 'processed') return 'Vote processed.';
  if (task.status === 'stale') return 'Task skipped because rollout trajectories became stale.';
  if (task.status === 'failed') return task.error ? `Task failed: ${task.error}` : 'Task failed.';
  return statusLabels[task.status] || task.status;
};

const cleanPrompt = (prompt: string) =>
  prompt
    .replace(/\s+--ctrl_img(_[123])?\s+\S+/g, '')
    .replace(/\s+--ctrl_idx\s+\S+/g, '')
    .trim();

const parseProgressSnapshot = (error: string | null | undefined) => {
  if (!error) return null;
  try {
    const parsed = JSON.parse(error) as {
      type?: string;
      completed_steps?: number;
      total_steps?: number;
      generated_candidates?: number;
      target_candidates?: number;
      it_per_sec?: number;
      elapsed_sec?: number;
      remaining_sec?: number;
    };
    if (parsed.type !== 'grpo_progress') return null;
    if (!Number.isFinite(parsed.completed_steps) || !Number.isFinite(parsed.total_steps)) return null;
    return {
      completedSteps: parsed.completed_steps as number,
      totalSteps: parsed.total_steps as number,
      generatedCandidates: Number.isFinite(parsed.generated_candidates) ? (parsed.generated_candidates as number) : null,
      targetCandidates: Number.isFinite(parsed.target_candidates) ? (parsed.target_candidates as number) : null,
      itPerSec: Number.isFinite(parsed.it_per_sec) ? (parsed.it_per_sec as number) : null,
      elapsedSec: Number.isFinite(parsed.elapsed_sec) ? (parsed.elapsed_sec as number) : null,
      remainingSec: Number.isFinite(parsed.remaining_sec) ? (parsed.remaining_sec as number) : null,
    };
  } catch {
    return null;
  }
};

const draftStorageKey = (jobID: string) => `flow-grpo-voting-draft:${jobID}`;

const formatIterationRate = (itPerSec: number | null) => {
  if (itPerSec == null || !Number.isFinite(itPerSec) || itPerSec <= 0) return null;
  if (itPerSec < 1) {
    return `${(1 / itPerSec).toFixed(2)} s/it`;
  }
  return `${itPerSec.toFixed(2)} it/s`;
};

const formatTqdmDuration = (seconds: number) => {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = `${m}`.padStart(h > 0 ? 2 : 1, '0');
  const ss = `${s}`.padStart(2, '0');
  if (h > 0) return `${h}:${mm}:${ss}`;
  return `${m}:${ss}`;
};

export default function FlowGRPOVotingPanel({ job, compact = false }: Props) {
  const jobConfig = JSON.parse(job.job_config) as JobConfig;
  const process = jobConfig.config.process[0];
  const inputImageMode = getVotingInputImageMode(process.model?.arch);
  const targetCandidates = process.grpo?.group_size || 4;
  const defaultTaskValues = useMemo(() => getTaskDefaults(job), [job]);

  const { tasks, status, refreshTasks } = useFlowGRPOVoteTasks(job.id, 3000);
  const [submittingTaskId, setSubmittingTaskId] = useState<string | null>(null);
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [candidateVotes, setCandidateVotes] = useState<Record<string, Record<string, string>>>({});
  const [hiddenHistoryTaskIDs, setHiddenHistoryTaskIDs] = useState<Set<string>>(new Set());
  const [taskDraft, setTaskDraft] = useState<FlowGRPOLiveTaskDraft>(() => getTaskDraftDefaults(job));

  useEffect(() => {
    const saved = sessionStorage.getItem(draftStorageKey(job.id));
    if (!saved) {
      setTaskDraft(getTaskDraftDefaults(job));
      return;
    }
    try {
      const parsed = JSON.parse(saved) as Partial<FlowGRPOLiveTaskDraft>;
      setTaskDraft({ ...getTaskDraftDefaults(job), ...parsed });
    } catch {
      setTaskDraft(getTaskDraftDefaults(job));
    }
  }, [job]);

  useEffect(() => {
    sessionStorage.setItem(draftStorageKey(job.id), JSON.stringify(taskDraft));
  }, [job.id, taskDraft]);

  useEffect(() => {
    setHiddenHistoryTaskIDs(new Set());
  }, [job.id]);

  const activeTasks = tasks.filter(task => activeTaskStatuses.has(task.status));
  const historyTasks = tasks.filter(task => !activeTaskStatuses.has(task.status) && !hiddenHistoryTaskIDs.has(task.id));

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

  const hideHistoryTask = async (taskID: string) => {
    setHiddenHistoryTaskIDs(current => new Set(current).add(taskID));
    try {
      await apiClient.post(`/api/grpo/jobs/${job.id}/tasks/${taskID}/hide`);
      refreshTasks();
    } catch (error) {
      console.error('Error hiding Flow-GRPO task:', error);
      setHiddenHistoryTaskIDs(current => {
        const next = new Set(current);
        next.delete(taskID);
        return next;
      });
    }
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
    const seedValue = taskDraft.seed.trim();
    const parsedSeed = seedValue === '' ? null : parseInt(seedValue, 10);
    const resolvedSeed = Number.isFinite(parsedSeed)
      ? (parsedSeed as number)
      : (defaultTaskValues.seed ?? randomSeed());
    const payload: FlowGRPOLiveTaskConfig = {
      prompt: taskDraft.prompt.trim(),
      negative_prompt: taskDraft.negative_prompt || '',
      width: parseIntegerField(taskDraft.width, defaultTaskValues.width, 64),
      height: parseIntegerField(taskDraft.height, defaultTaskValues.height, 64),
      seed: resolvedSeed,
      ctrl_img: inputImageMode === 'single' ? (taskDraft.ctrl_img_1 || null) : null,
      ctrl_img_1: taskDraft.ctrl_img_1 || null,
      ctrl_img_2: taskDraft.ctrl_img_2 || null,
      ctrl_img_3: taskDraft.ctrl_img_3 || null,
      guidance_scale: parseFloatField(taskDraft.guidance_scale, defaultTaskValues.guidance_scale, 0),
      num_inference_steps: parseIntegerField(taskDraft.num_inference_steps, defaultTaskValues.num_inference_steps, 1),
      sampler: taskDraft.sampler,
      scheduler: taskDraft.scheduler,
    };
    setIsCreatingTask(true);
    try {
      await apiClient.post(`/api/grpo/jobs/${job.id}/tasks`, payload);
      setTaskDraft(current => ({
        ...current,
        prompt: '',
        seed: current.seed_mode === 'random' ? `${randomSeed()}` : current.seed,
      }));
      refreshTasks();
    } catch (error) {
      console.error('Error creating Flow-GRPO vote task:', error);
    } finally {
      setIsCreatingTask(false);
    }
  };

  const renderTask = (task: FlowGRPOVoteTaskView, includeVoting: boolean) => {
    const canVote = includeVoting && task.status === 'open' && task.candidates.length > 1;
    const taskVotes = candidateVotes[task.id] || {};
    const candidateIDs = task.candidates.map(candidate => candidate.id);
    const allCandidatesVoted = canVote && candidateIDs.every(candidateID => taskVotes[candidateID]);
    const generatedCount = task.candidates.length;
    const voteByCandidateID = new Map(
      task.votes
        .filter(vote => vote.candidate_id)
        .map(vote => [vote.candidate_id as string, vote]),
    );
    const stepsPerCandidate = task.num_inference_steps ?? defaultTaskValues.num_inference_steps;
    const fallbackCompletedSteps = generatedCount * stepsPerCandidate;
    const fallbackTotalSteps = Math.max(1, targetCandidates * stepsPerCandidate);
    const progressSnapshot =
      task.status === 'generating' || task.status === 'voted'
        ? parseProgressSnapshot(task.error)
        : null;
    const isApplyingVote = task.status === 'voted';
    const completedSteps = isApplyingVote
      ? (progressSnapshot?.completedSteps ?? 0)
      : (progressSnapshot?.completedSteps ?? fallbackCompletedSteps);
    const totalSteps = isApplyingVote
      ? Math.max(1, progressSnapshot?.totalSteps ?? 1)
      : (progressSnapshot?.totalSteps ?? fallbackTotalSteps);
    const displayGenerated =
      progressSnapshot?.generatedCandidates != null
        ? Math.max(generatedCount, progressSnapshot.generatedCandidates)
        : generatedCount;
    const displayTarget = progressSnapshot?.targetCandidates ?? targetCandidates;
    const itPerSec = progressSnapshot?.itPerSec ?? null;
    const formattedRate = formatIterationRate(itPerSec);
    const elapsedSec = progressSnapshot?.elapsedSec ?? null;
    const remainingSec = progressSnapshot?.remainingSec ?? null;
    const formattedTqdmTime =
      elapsedSec != null && remainingSec != null
        ? `${formatTqdmDuration(elapsedSec)}<${formatTqdmDuration(remainingSec)}`
        : null;
    const progress =
      task.status === 'generating' || task.status === 'voted'
        ? Math.min(99, Math.round((completedSteps / totalSteps) * 100))
        : 100;

    return (
      <div key={task.id} className="rounded-xl border border-gray-800 bg-gray-950 p-4">
        <div className="mb-2 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-wide text-gray-500">Prompt</div>
            <div className="mt-1 text-sm text-gray-100">{cleanPrompt(task.prompt)}</div>
            {task.negative_prompt && (
              <div className="mt-1 text-xs text-gray-500">Negative: {task.negative_prompt}</div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div
              className={classNames(
                'rounded-full px-3 py-1 text-xs font-medium',
                task.status === 'open' && 'bg-blue-950 text-blue-200',
                task.status === 'generating' && 'bg-amber-950 text-amber-200',
                task.status === 'requested' && 'bg-gray-800 text-gray-200',
                task.status === 'voted' && 'bg-emerald-950 text-emerald-200',
                task.status === 'stale' && 'bg-orange-950 text-orange-200',
                task.status === 'failed' && 'bg-red-950 text-red-200',
                task.status === 'processed' && 'bg-gray-800 text-gray-200',
              )}
            >
              {statusLabels[task.status] || task.status}
            </div>
            {!includeVoting && (
              <Button
                onClick={() => hideHistoryTask(task.id)}
                title="Hide"
                aria-label="Hide"
                className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-gray-800 text-gray-300 hover:bg-gray-700"
              >
                <EyeOff className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>

        {(task.status === 'generating' || task.status === 'voted') && (
          <div className="mb-3">
            <div className="mb-1.5 flex items-center justify-between text-xs text-gray-400">
              <span>{isApplyingVote ? 'Applying vote' : `Generated ${displayGenerated}/${displayTarget} candidates`}</span>
              <span>
                Steps {completedSteps}/{totalSteps}
                {formattedTqdmTime ? `  •  ${formattedTqdmTime}` : ''}
                {formattedRate ? `  •  ${formattedRate}` : ''}
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded bg-gray-800">
              <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {!canVote && (
          <div className="rounded-lg border border-dashed border-gray-800 bg-gray-900/60 p-3 text-sm text-gray-400">
            {getTaskStatusMessage(task)}
          </div>
        )}

        {task.candidates.length > 0 && (
          <div className="mt-3 grid gap-3 [grid-template-columns:repeat(auto-fill,minmax(220px,1fr))]">
            {task.candidates.map(candidate => (
              <div key={candidate.id} className="overflow-hidden rounded-lg border border-gray-800 bg-gray-900">
                <img src={candidate.image_url} alt={candidate.prompt} className="aspect-[4/5] w-full bg-black object-cover" />
                <div className="space-y-2 p-3">
                  {!includeVoting && voteByCandidateID.has(candidate.id) && (
                    <div>
                      <span
                        className={classNames(
                          'inline-flex rounded-md px-2 py-1 text-xs font-medium',
                          voteDisplay[voteByCandidateID.get(candidate.id)?.value || 'reward']?.className ||
                            voteDisplay.reward.className,
                        )}
                      >
                        {voteDisplay[voteByCandidateID.get(candidate.id)?.value || 'reward']?.label ||
                          voteDisplay.reward.label}
                      </span>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-xs text-gray-400">
                    <div>Seed: {candidate.seed}</div>
                    <div>CFG: {candidate.guidance_scale}</div>
                    <div>Steps: {candidate.num_inference_steps}</div>
                    <div>Sampler: {candidate.sampler}</div>
                  </div>

                  {canVote && (
                    <div className="grid grid-cols-3 gap-1.5">
                      {voteOptions.map(option => {
                        const isSelected = taskVotes[candidate.id] === option.value;
                        return (
                          <Button
                            key={option.value}
                            onClick={() => setCandidateVote(task.id, candidate.id, option.value)}
                            disabled={submittingTaskId === task.id}
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
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {canVote && (
          <Button
            onClick={() => submitVote(task.id, candidateIDs)}
            disabled={!allCandidatesVoted || submittingTaskId === task.id}
            className={classNames(
              'mt-3 w-full rounded-md px-3 py-2 text-sm font-medium',
              !allCandidatesVoted || submittingTaskId === task.id
                ? 'cursor-not-allowed bg-gray-800 text-gray-500'
                : 'bg-blue-600 text-white hover:bg-blue-500',
            )}
          >
            Submit Group Vote
          </Button>
        )}
      </div>
    );
  };

  const activeSectionTitle = compact ? `Flow-GRPO: ${job.name}` : 'Generation & Voting';

  return (
    <div className="space-y-4">
      <div className={classNames('grid gap-4', !compact && 'lg:grid-cols-12 lg:items-start')}>
        {!compact && (
          <div
            className="lg:col-span-4"
            style={{ position: 'sticky', top: 0, alignSelf: 'flex-start', zIndex: 5 }}
          >
            <div className="max-h-[calc(100vh-108px)] overflow-auto rounded-xl border border-gray-800 bg-gray-950 p-4">
              <div className="space-y-3">
                <div>
                  <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Prompt</label>
                  <textarea
                    value={taskDraft.prompt}
                    onChange={event => updateDraft('prompt', event.target.value)}
                    rows={3}
                    className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    placeholder="Describe the sample request"
                  />
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Negative Prompt</label>
                  <textarea
                    value={taskDraft.negative_prompt || ''}
                    onChange={event => updateDraft('negative_prompt', event.target.value)}
                    rows={2}
                    className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500"
                    placeholder="Optional negative prompt"
                  />
                </div>

                <div>
                  <label className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-gray-400">Seed</label>
                  <div className="grid grid-cols-[auto_auto_1fr_auto] gap-2">
                    <Button
                      onClick={() => updateDraft('seed_mode', 'fixed')}
                      className={classNames(
                        'rounded-md px-3 py-2 text-sm font-medium',
                        taskDraft.seed_mode === 'fixed'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 text-gray-300 hover:bg-gray-700',
                      )}
                    >
                      Fixed
                    </Button>
                    <Button
                      onClick={() => {
                        updateDraft('seed_mode', 'random');
                        updateDraft('seed', `${randomSeed()}`);
                      }}
                      className={classNames(
                        'rounded-md px-3 py-2 text-sm font-medium',
                        taskDraft.seed_mode === 'random'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 text-gray-300 hover:bg-gray-700',
                      )}
                    >
                      Random
                    </Button>
                    <input
                      type="number"
                      value={taskDraft.seed}
                      onChange={event => updateDraft('seed', event.target.value)}
                      disabled={taskDraft.seed_mode === 'random'}
                      className="w-full rounded-lg border border-gray-800 bg-gray-900 px-3 py-2 text-sm text-gray-100 outline-none focus:border-blue-500 disabled:cursor-not-allowed disabled:text-gray-500"
                      placeholder={taskDraft.seed_mode === 'random' ? 'Auto seed' : 'Seed value'}
                    />
                    <Button
                      onClick={() => updateDraft('seed', `${randomSeed()}`)}
                      disabled={taskDraft.seed_mode === 'random'}
                      title="Randomize fixed seed"
                      className="rounded-md bg-gray-800 px-3 text-gray-200 hover:bg-gray-700 disabled:cursor-not-allowed disabled:text-gray-500"
                    >
                      <FaDice />
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
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

                {inputImageMode !== 'none' && (
                  <div>
                    <label className="mb-2 block text-xs font-medium uppercase tracking-wide text-gray-400">Input Images</label>
                    <div className="flex flex-wrap gap-2">
                      <SampleControlImage
                        src={taskDraft.ctrl_img_1}
                        instruction="Image 1"
                        onNewImageSelected={imagePath => updateDraft('ctrl_img_1', imagePath)}
                      />
                      {inputImageMode === 'multi' && (
                        <>
                          <SampleControlImage
                            src={taskDraft.ctrl_img_2}
                            instruction="Image 2"
                            onNewImageSelected={imagePath => updateDraft('ctrl_img_2', imagePath)}
                          />
                          <SampleControlImage
                            src={taskDraft.ctrl_img_3}
                            instruction="Image 3"
                            onNewImageSelected={imagePath => updateDraft('ctrl_img_3', imagePath)}
                          />
                        </>
                      )}
                    </div>
                  </div>
                )}

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

        <div className={classNames('space-y-5', !compact && 'lg:col-span-8')}>
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-300">{activeSectionTitle}</h2>
            <Button onClick={refreshTasks} className="rounded-md bg-gray-800 px-3 py-1 text-sm text-gray-200 hover:bg-gray-700">
              Refresh
            </Button>
          </div>

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
          {status !== 'error' && activeTasks.length === 0 && (
            <div className="rounded-xl border border-dashed border-gray-700 bg-gray-900/60 p-6 text-sm text-gray-400">
              No active tasks.
            </div>
          )}

          <div className="space-y-4">{activeTasks.map(task => renderTask(task, true))}</div>

          <div>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-300">History</h2>
            {historyTasks.length === 0 ? (
              <div className="rounded-xl border border-dashed border-gray-700 bg-gray-900/60 p-6 text-sm text-gray-400">
                No generation or voting history yet.
              </div>
            ) : (
              <div className="space-y-4">{historyTasks.map(task => renderTask(task, false))}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
