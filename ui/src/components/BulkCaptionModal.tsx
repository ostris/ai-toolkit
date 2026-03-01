'use client';
import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaComment } from 'react-icons/fa';

interface BulkCaptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStart: (options: { modelId: string; triggerWord: string; systemPrompt: string }) => void;
}

const MODEL_LITE = 'prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1';
const MODEL_FULL = 'prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it';

const MODEL_OPTIONS = [
  { value: MODEL_LITE, label: 'Qwen3-VL 4B (Lite · ~5–8 GB VRAM, fast)' },
  { value: MODEL_FULL, label: 'Qwen3-VL 8B (Full · ~10–14 GB VRAM, max quality)' },
];

const DEFAULT_SYSTEM_PROMPT = 'Describe the subject and overall scene in detail.';

export default function BulkCaptionModal({ isOpen, onClose, onStart }: BulkCaptionModalProps) {
  const [mounted, setMounted] = useState(false);
  const [triggerWord, setTriggerWord] = useState('');
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [modelId, setModelId] = useState(MODEL_LITE);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!isOpen) {
      setTriggerWord('');
      setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
      setModelId(MODEL_LITE);
    }
  }, [isOpen]);

  const handleStart = useCallback(() => {
    onStart({ modelId, triggerWord: triggerWord.trim(), systemPrompt: systemPrompt.trim() });
  }, [modelId, triggerWord, systemPrompt, onStart]);

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 w-full max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-6 pt-5 pb-4">
              <DialogTitle as="h3" className="text-base font-semibold text-gray-100 mb-4 flex items-center gap-2">
                <FaComment />
                Caption All Images
              </DialogTitle>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Model</label>
                  <select
                    value={modelId}
                    onChange={e => setModelId(e.target.value)}
                    className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2"
                    aria-label="Model"
                  >
                    {MODEL_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                  <p className="mt-1 text-xs text-gray-500">
                    Model auto-downloads on first use. 4B is faster; 8B produces higher-quality captions.
                  </p>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Trigger Word</label>
                  <input
                    type="text"
                    value={triggerWord}
                    onChange={e => setTriggerWord(e.target.value)}
                    placeholder="e.g. ohwx person"
                    className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2"
                    aria-label="Trigger word"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Every caption will begin with this trigger word. Leave blank to omit.
                  </p>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Caption Focus</label>
                  <textarea
                    value={systemPrompt}
                    onChange={e => setSystemPrompt(e.target.value)}
                    rows={3}
                    className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2 resize-none"
                    aria-label="Caption focus"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Tell the model what to focus on for every image in this dataset.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-gray-700 px-6 py-3 flex justify-end gap-3">
              <button
                type="button"
                onClick={onClose}
                className="rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-900"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleStart}
                className="rounded-md bg-blue-700 hover:bg-blue-500 px-3 py-2 text-sm font-semibold text-white flex items-center gap-2"
              >
                <FaComment className="w-3 h-3" />
                Caption Images
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
