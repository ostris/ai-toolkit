'use client';
import React, { useEffect, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Trash2 } from 'lucide-react';
import { Modal } from './Modal';
import { TextInput, SelectInput } from '@/components/formInputs';
import { getFilename } from '@/utils/basic';
import { callScriptStream } from '@/utils/callScript';
import { SelectOption } from '@/types';

export interface MergeLoRAFile {
  path: string;
}

export interface MergeLoRAsModalState {
  folderPath: string;
  outputName: string;
  availableLoRAs: MergeLoRAFile[];
  onClose?: () => void;
}

interface SelectedLoRA {
  path: string;
  strength: number;
}

export const mergeLoRAsModalState = createGlobalState<MergeLoRAsModalState | null>(null);

export const openMergeLoRAsModal = (
  folderPath: string,
  outputName: string,
  availableLoRAs: MergeLoRAFile[],
  onClose?: () => void,
) => {
  mergeLoRAsModalState.set({
    folderPath,
    outputName,
    availableLoRAs,
    onClose,
  });
};

const joinPath = (folder: string, name: string) => {
  const sep = folder.includes('\\') && !folder.includes('/') ? '\\' : '/';
  const trimmed = folder.replace(/[\\/]+$/, '');
  return `${trimmed}${sep}${name}.safetensors`;
};

const MergeLoRAsModal: React.FC = () => {
  const [modalInfo, setModalInfo] = mergeLoRAsModalState.use();
  const isOpen = modalInfo !== null;
  const [selectedLoRAs, setSelectedLoRAs] = useState<SelectedLoRA[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [logOutput, setLogOutput] = useState('');
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!modalInfo) {
      setSelectedLoRAs([]);
      setIsRunning(false);
      setIsDone(false);
      setHasError(false);
      setLogOutput('');
    }
  }, [modalInfo]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logOutput]);

  const onClose = () => {
    if (isRunning) return;
    setModalInfo(null);
    modalInfo?.onClose?.();
  };

  const onSubmit = async () => {
    if (isRunning || !modalInfo) return;
    setIsRunning(true);
    setIsDone(false);
    setHasError(false);
    setLogOutput('');

    const output = joinPath(modalInfo.folderPath, modalInfo.outputName);

    const append = (chunk: string) => setLogOutput(prev => prev + chunk);

    try {
      const finalEvent = await callScriptStream('merge_loras.py', {
        args: {
          loras: JSON.stringify(selectedLoRAs),
          output,
          save_dtype: 'bfloat16',
          device: 'cpu',
        },
        onStdout: append,
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
      setIsRunning(false);
      setIsDone(true);
    }
  };

  const loraLabel = (path: string) => getFilename(path).replace('.safetensors', '');

  const availableLoRAs = modalInfo?.availableLoRAs ?? [];
  const selectedPaths = selectedLoRAs.map(s => s.path);

  const options: SelectOption[] = availableLoRAs
    .filter(f => !selectedPaths.includes(f.path))
    .map(f => ({ value: f.path, label: loraLabel(f.path) }));

  const rescale = (items: SelectedLoRA[]): SelectedLoRA[] => {
    if (items.length === 0) return items;
    const strength = Math.round((1 / items.length) * 1000) / 1000;
    return items.map(s => ({ ...s, strength }));
  };

  const addLoRA = (path: string) => {
    if (!path || selectedPaths.includes(path)) return;
    setSelectedLoRAs(prev => rescale([...prev, { path, strength: 0 }]));
  };

  const removeLoRA = (path: string) => {
    setSelectedLoRAs(prev => rescale(prev.filter(s => s.path !== path)));
  };

  const updateStrength = (path: string, strength: number | null) => {
    setSelectedLoRAs(prev => prev.map(s => (s.path === path ? { ...s, strength: strength ?? 0 } : s)));
  };

  const showLog = isRunning || isDone;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Merge LoRAs"
      size="lg"
      showCloseButton={!isRunning}
      closeOnOverlayClick={!isRunning}
    >
      {showLog ? (
        <div>
          <div className="mb-2 text-sm">
            {isRunning && <span className="text-amber-400">Merging LoRAs... please do not close this window.</span>}
            {isDone && hasError && <span className="text-rose-400">Merge failed. See log below.</span>}
            {isDone && !hasError && <span className="text-emerald-400">Merge complete.</span>}
          </div>
          <div
            ref={logRef}
            className="font-mono text-xs whitespace-pre-wrap break-all overflow-y-auto rounded-md p-3 min-h-[400px] max-h-[60vh] bg-white text-gray-900 dark:bg-black dark:text-gray-100"
          >
            {logOutput || (isRunning ? 'Starting...\n' : '')}
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isRunning}
              className="px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed text-gray-100 rounded-md"
            >
              Close
            </button>
          </div>
        </div>
      ) : (
        <form
          onSubmit={e => {
            e.preventDefault();
            onSubmit();
          }}
        >
          <TextInput
            label="Output Filename"
            value={modalInfo?.outputName || ''}
            suffix=".safetensors"
            onChange={value => {
              setModalInfo({
                ...modalInfo,
                outputName: value,
              } as MergeLoRAsModalState);
            }}
            placeholder="Enter output filename"
          />

          <div className="mt-4">
            <SelectInput
              label="Add LoRA"
              multiple={false}
              value=""
              onChange={value => addLoRA(value)}
              options={options}
            />
          </div>

          {selectedLoRAs.length > 0 && (
            <div className="mt-4">
              <label className="block text-xs mb-1 text-gray-300">Selected LoRAs</label>
              <div className="bg-purple-500/10 rounded-xl p-2 max-h-48 overflow-y-auto space-y-1">
                {selectedLoRAs.map(s => (
                  <div key={s.path} className="flex items-center gap-2 px-2 py-0.5">
                    <div
                      className="flex-1 min-w-0 text-xs text-gray-200 overflow-hidden text-ellipsis whitespace-nowrap"
                      title={loraLabel(s.path)}
                    >
                      {loraLabel(s.path)}
                    </div>
                    <input
                      type="number"
                      value={s.strength}
                      onChange={e => {
                        const raw = e.target.value;
                        if (raw === '' || raw === '-') return;
                        const n = Number(raw);
                        if (!isNaN(n)) updateStrength(s.path, n);
                      }}
                      step="any"
                      className="w-20 flex-shrink-0 text-xs px-2 py-0.5 bg-gray-950 dark:bg-gray-800 border border-gray-700 rounded-sm text-gray-100 focus:ring-1 focus:ring-gray-600 focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => removeLoRA(s.path)}
                      className="flex-shrink-0 text-gray-400 hover:text-rose-400 p-0.5"
                      aria-label="Remove"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-6 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-300 hover:text-gray-100 rounded-md"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={selectedLoRAs.length === 0 || !modalInfo?.outputName}
              className="px-4 py-2 text-sm bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md"
            >
              Merge
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
};

export default MergeLoRAsModal;
