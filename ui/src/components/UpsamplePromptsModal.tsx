'use client';
import React, { useEffect, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Check, X, Loader2, AlertTriangle } from 'lucide-react';
import classNames from 'classnames';
import { Modal } from './Modal';
import { callScriptStream } from '@/utils/callScript';

export interface UpsamplePromptItem {
  index: number; // index into the samples array (write-back target)
  prompt: string; // current prompt text
  aspectRatio: string; // 'W:H' or 'auto'
}

export interface UpsamplePromptsModalState {
  prompts: UpsamplePromptItem[];
  onApply: (index: number, newPrompt: string) => void;
  onClose?: () => void;
}

export const upsamplePromptsModalState = createGlobalState<UpsamplePromptsModalState | null>(null);

export const openUpsamplePromptsModal = (
  prompts: UpsamplePromptItem[],
  onApply: (index: number, newPrompt: string) => void,
  onClose?: () => void,
) => {
  upsamplePromptsModalState.set({ prompts, onApply, onClose });
};

// Reduce/snap pixel dimensions to a clean 'W:H' (denominator <= 16), mirroring the
// captioner's compute_aspect_ratio. Returns 'auto' when dimensions are unknown.
export function toAspectRatio(width?: number, height?: number): string {
  if (!width || !height || width <= 0 || height <= 0) return 'auto';
  const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
  const g = gcd(width, height);
  const rw = width / g;
  const rh = height / g;
  const MAXD = 16;
  if (rw <= MAXD && rh <= MAXD) return `${rw}:${rh}`;
  const target = width / height;
  let best: { err: number; p: number; q: number } | null = null;
  for (let q = 1; q <= MAXD; q++) {
    const p = Math.max(1, Math.round(target * q));
    const err = Math.abs(p / q - target);
    if (!best || err < best.err) best = { err, p, q };
  }
  return `${best!.p}:${best!.q}`;
}

type RowStatus = 'idle' | 'queued' | 'running' | 'done' | 'failed';

const UpsamplePromptsModal: React.FC = () => {
  const [modalInfo, setModalInfo] = upsamplePromptsModalState.use();
  const isOpen = modalInfo !== null;

  const [selected, setSelected] = useState<Record<number, boolean>>({});
  const [creative, setCreative] = useState(false);
  const [instructions, setInstructions] = useState('');
  const [status, setStatus] = useState<Record<number, RowStatus>>({});
  const [upsampled, setUpsampled] = useState<Record<number, string>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [logOutput, setLogOutput] = useState('');
  const logRef = useRef<HTMLDivElement | null>(null);

  // Reset state when the modal opens/closes. Nothing is selected by default.
  useEffect(() => {
    setSelected({});
    setCreative(false);
    setInstructions('');
    setStatus({});
    setUpsampled({});
    setIsRunning(false);
    setIsDone(false);
    setHasError(false);
    setLogOutput('');
  }, [modalInfo]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logOutput]);

  const onClose = () => {
    if (isRunning) return;
    setModalInfo(null);
    modalInfo?.onClose?.();
  };

  const prompts = modalInfo?.prompts ?? [];
  const selectedCount = prompts.filter(p => selected[p.index]).length;

  const toggle = (index: number) => {
    if (isRunning) return;
    setSelected(prev => ({ ...prev, [index]: !prev[index] }));
  };

  const setAll = (value: boolean) => {
    if (isRunning) return;
    const next: Record<number, boolean> = {};
    prompts.forEach(p => (next[p.index] = value));
    setSelected(next);
  };

  const onRun = async () => {
    if (isRunning || !modalInfo) return;
    const submitted = modalInfo.prompts.filter(p => selected[p.index]);
    if (submitted.length === 0) return;

    setIsRunning(true);
    setIsDone(false);
    setHasError(false);
    setLogOutput('');
    setUpsampled({});

    const submittedIndices = submitted.map(p => p.index);
    setStatus(() => {
      const s: Record<number, RowStatus> = {};
      submittedIndices.forEach((idx, k) => (s[idx] = k === 0 ? 'running' : 'queued'));
      return s;
    });

    const payload = submitted.map(p => ({ prompt: p.prompt, aspect_ratio: p.aspectRatio }));

    // The script streams one compact JSON line per completed prompt on stdout.
    // stdout chunks don't align to line boundaries, so buffer and split ourselves.
    let buffer = '';
    const handleStdout = (chunk: string) => {
      buffer += chunk;
      let nl: number;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        try {
          const evt = JSON.parse(line);
          if (typeof evt.index !== 'number') continue;
          const orig = submittedIndices[evt.index];
          if (orig === undefined) continue;
          if (evt.caption) {
            const text = JSON.stringify(evt.caption, null, 2);
            setUpsampled(prev => ({ ...prev, [orig]: text }));
            setStatus(prev => ({ ...prev, [orig]: 'done' }));
            modalInfo.onApply(orig, text); // write back to the job config live
          } else {
            setStatus(prev => ({ ...prev, [orig]: 'failed' }));
          }
          const nextIdx = submittedIndices[evt.index + 1];
          if (nextIdx !== undefined) {
            setStatus(prev => (prev[nextIdx] === 'queued' ? { ...prev, [nextIdx]: 'running' } : prev));
          }
        } catch {
          // ignore non-JSON / partial lines
        }
      }
    };

    const append = (chunk: string) => setLogOutput(prev => prev + chunk);

    try {
      const finalEvent = await callScriptStream('upsample_ideogram4_caption.py', {
        args: {
          prompts: JSON.stringify(payload),
          stream: true,
          temperature: 0.7,
          quantize: true, // float8 (script default qtype) to reduce VRAM
          ...(creative ? { creative: true } : {}),
          ...(instructions.trim() ? { instructions: instructions.trim() } : {}),
        },
        onStdout: handleStdout,
        onStderr: append,
      });

      const ok = finalEvent?.type === 'exit' && finalEvent.ok === true;
      if (!ok) {
        setHasError(true);
        if (finalEvent?.type === 'error' && finalEvent.message) {
          append(`\n${finalEvent.message}\n`);
        } else if (finalEvent?.type === 'exit' && finalEvent.timedOut) {
          append('\nScript timed out.\n');
        } else if (finalEvent?.type === 'exit') {
          append(`\nScript exited with code ${finalEvent.exitCode}.\n`);
        }
      }
    } catch (err: any) {
      setHasError(true);
      append(`\n${err?.message || 'Unknown error'}\n`);
    } finally {
      // Anything still queued/running never reported a result -> failed.
      setStatus(prev => {
        const cp = { ...prev };
        submittedIndices.forEach(i => {
          if (cp[i] === 'running' || cp[i] === 'queued') cp[i] = 'failed';
        });
        return cp;
      });
      setIsRunning(false);
      setIsDone(true);
    }
  };

  const StatusIcon = ({ s }: { s: RowStatus | undefined }) => {
    if (s === 'done') return <Check className="w-4 h-4 text-emerald-400" />;
    if (s === 'failed') return <X className="w-4 h-4 text-rose-400" />;
    if (s === 'running') return <Loader2 className="w-4 h-4 text-amber-400 animate-spin" />;
    if (s === 'queued') return <Loader2 className="w-4 h-4 text-gray-500" />;
    return <span className="inline-block w-4 h-4" />;
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Upsample Prompts"
      size="lg"
      showCloseButton={!isRunning}
      closeOnOverlayClick={!isRunning}
    >
      <div>
        <div className="mb-3 flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-300">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>
            This loads a model and runs on the GPU. You need at least 13GB of free VRAM to run this, so make sure the
            GPU is idle (no training or other jobs running) before starting, or it may run out of memory.
          </span>
        </div>

        <div className="mb-2 flex items-center justify-between text-sm">
          <div className="text-gray-400">
            {isRunning && <span className="text-amber-400">Upsampling... please do not close this window.</span>}
            {isDone && hasError && <span className="text-rose-400">Finished with errors. See log below.</span>}
            {isDone && !hasError && <span className="text-emerald-400">Upsampling complete.</span>}
            {!isRunning && !isDone && (
              <span>
                Select prompts to upsample into structured Ideogram captions. {selectedCount}/{prompts.length} selected.
              </span>
            )}
          </div>
          {!isRunning && !isDone && (
            <div className="flex gap-2 flex-shrink-0">
              <button type="button" onClick={() => setAll(true)} className="text-xs text-gray-300 hover:text-gray-100">
                Select all
              </button>
              <span className="text-gray-600">|</span>
              <button type="button" onClick={() => setAll(false)} className="text-xs text-gray-300 hover:text-gray-100">
                None
              </button>
            </div>
          )}
        </div>

        <div className="mb-2 flex items-center gap-2">
          <button
            type="button"
            disabled={isRunning || isDone}
            onClick={() => setCreative(c => !c)}
            className={classNames(
              'px-2.5 py-1 text-xs rounded-md border transition-colors disabled:opacity-40 disabled:cursor-not-allowed',
              {
                'bg-purple-600 border-purple-500 text-white': creative,
                'border-gray-700 text-gray-300 hover:border-gray-500': !creative,
              },
            )}
          >
            Creative: {creative ? 'On' : 'Off'}
          </button>
          <span className="text-[11px] text-gray-500">
            {creative
              ? 'Expands the idea — places the subject in a scene and adds fitting details.'
              : 'Faithful — structures the prompt as given, with a minimal background.'}
          </span>
        </div>

        <div className="mb-3">
          <label className="block text-[11px] mb-1 text-gray-400">Additional instructions (optional)</label>
          <textarea
            value={instructions}
            onChange={e => setInstructions(e.target.value)}
            disabled={isRunning || isDone}
            rows={2}
            placeholder="e.g. keep it a close-up portrait, prefer a daytime setting, always include a 9:16 vertical framing..."
            className="w-full text-xs px-2 py-1.5 bg-gray-950 border border-gray-700 rounded-md text-gray-100 placeholder-gray-600 focus:ring-1 focus:ring-gray-600 focus:outline-none resize-none disabled:opacity-50"
          />
        </div>

        <div className="rounded-md bg-gray-950 border border-gray-800 max-h-[45vh] overflow-y-auto divide-y divide-gray-800">
          {prompts.map(p => {
            const s = status[p.index];
            const isSelected = !!selected[p.index];
            const display = upsampled[p.index] ?? p.prompt;
            const dim = (isRunning || isDone) && !s;
            return (
              <div key={p.index} className={classNames('flex items-start gap-2 px-3 py-2', { 'opacity-40': dim })}>
                <input
                  type="checkbox"
                  checked={isSelected}
                  disabled={isRunning || isDone}
                  onChange={() => toggle(p.index)}
                  className="mt-0.5 flex-shrink-0 accent-purple-600"
                />
                <div className="flex-shrink-0 mt-0.5">
                  <StatusIcon s={s} />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-gray-500 flex-shrink-0">#{p.index + 1}</span>
                    <span className="text-[10px] text-gray-500 flex-shrink-0">{p.aspectRatio}</span>
                    {upsampled[p.index] && (
                      <span className="text-[10px] text-emerald-400 flex-shrink-0">upsampled</span>
                    )}
                  </div>
                  <div
                    className={classNames('text-xs break-words line-clamp-2', {
                      'text-emerald-200': upsampled[p.index],
                      'text-gray-300': !upsampled[p.index],
                    })}
                    title={upsampled[p.index] ? upsampled[p.index] : p.prompt}
                  >
                    {display}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {(isRunning || isDone) && logOutput && (
          <div
            ref={logRef}
            className="mt-3 font-mono text-[10px] whitespace-pre-wrap break-all overflow-y-auto rounded-md p-2 max-h-32 bg-black text-gray-300"
          >
            {logOutput}
          </div>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={isRunning}
            className="px-4 py-2 text-sm text-gray-300 hover:text-gray-100 disabled:opacity-40 disabled:cursor-not-allowed rounded-md"
          >
            {isDone ? 'Close' : 'Cancel'}
          </button>
          {!isDone && (
            <button
              type="button"
              onClick={onRun}
              disabled={isRunning || selectedCount === 0}
              className="px-4 py-2 text-sm bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md"
            >
              {isRunning ? 'Upsampling...' : `Upsample (${selectedCount})`}
            </button>
          )}
        </div>
      </div>
    </Modal>
  );
};

export default UpsamplePromptsModal;
