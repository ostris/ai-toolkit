'use client';
import React, { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaComment } from 'react-icons/fa';
import { apiClient } from '@/utils/api';

interface CaptionModalProps {
  imageUrl: string;
  isOpen: boolean;
  onClose: () => void;
  onCaptionGenerated?: (caption: string) => void;
}

const MODEL_LITE = 'prithivMLmods/Qwen3-VL-4B-Instruct-abliterated-v1';
const MODEL_FULL = 'prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it';

const MODEL_OPTIONS = [
  { value: MODEL_LITE, label: 'Qwen3-VL-4B-Instruct-abliterated-v1 (~5–8 GB VRAM, fast)' },
  { value: MODEL_FULL, label: 'Qwen3-VL-8B-Instruct-abliterated-v1 (~10–14 GB VRAM, max quality)' },
];

const DEFAULT_SYSTEM_PROMPT =
  'Describe the subject and overall scene in detail.';

export default function CaptionModal({ imageUrl, isOpen, onClose, onCaptionGenerated }: CaptionModalProps) {
  const [mounted, setMounted] = useState(false);
  const [triggerWord, setTriggerWord] = useState('');
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [modelId, setModelId] = useState(MODEL_LITE);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedCaption, setGeneratedCaption] = useState<string | null>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!isOpen) {
      setError(null);
      setGeneratedCaption(null);
      setIsGenerating(false);
    }
  }, [isOpen]);

  const handleClose = useCallback(() => {
    if (!isGenerating) {
      onClose();
    }
  }, [isGenerating, onClose]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    setGeneratedCaption(null);
    try {
      const res = await apiClient.post('/api/img/ai-caption', {
        imgPath: imageUrl,
        triggerWord: triggerWord.trim(),
        systemPrompt: systemPrompt.trim(),
        modelId,
      });
      const caption = res.data?.caption ?? '';
      setGeneratedCaption(caption);
      onCaptionGenerated?.(caption);
      onClose();
    } catch (err: any) {
      const msg = err?.response?.data?.error || 'Failed to generate caption';
      setError(msg);
    } finally {
      setIsGenerating(false);
    }
  };

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={handleClose} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div
        className="fixed inset-0 z-10 w-screen overflow-y-auto"
        onPointerDown={e => e.stopPropagation()}
        onPointerUp={e => e.stopPropagation()}
        onPointerLeave={e => e.stopPropagation()}
        onPointerCancel={e => e.stopPropagation()}
      >
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 w-full max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-6 pt-5 pb-4">
              <DialogTitle as="h3" className="text-base font-semibold text-gray-100 mb-4 flex items-center gap-2">
                <FaComment />
                Generate Caption
              </DialogTitle>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Model</label>
                  <select
                    value={modelId}
                    onChange={e => setModelId(e.target.value)}
                    disabled={isGenerating}
                    className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2 disabled:opacity-50"
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
                    disabled={isGenerating}
                    className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2 disabled:opacity-50"
                    aria-label="Trigger word"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    The caption will begin with this trigger word. Leave blank to omit.
                  </p>
                </div>

                <div>
                  <label className="block text-sm text-gray-400 mb-1">Caption Focus</label>
                  <textarea
                    value={systemPrompt}
                    onChange={e => setSystemPrompt(e.target.value)}
                    rows={3}
                    disabled={isGenerating}
                    className="w-full bg-gray-700 text-white text-sm rounded px-3 py-2 resize-none disabled:opacity-50"
                    aria-label="Caption focus"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Tell the model what to focus on (e.g. "Describe clothing style and fit in detail.").
                  </p>
                </div>

                {generatedCaption !== null && (
                  <div>
                    <label className="block text-sm text-gray-400 mb-1">Generated Caption</label>
                    <div className="bg-gray-700 rounded px-3 py-2 text-sm text-gray-100 whitespace-pre-wrap">
                      {generatedCaption}
                    </div>
                    <p className="mt-1 text-xs text-gray-500">
                      Caption has been saved to the image.
                    </p>
                  </div>
                )}

                {error && <div className="text-sm text-red-400">{error}</div>}
              </div>
            </div>

            <div className="bg-gray-700 px-6 py-3 flex justify-end gap-3">
              <button
                type="button"
                onClick={handleClose}
                disabled={isGenerating}
                className="rounded-md bg-gray-800 px-3 py-2 text-sm font-semibold text-gray-200 hover:bg-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {generatedCaption !== null ? 'Close' : 'Cancel'}
              </button>
              <button
                type="button"
                onClick={handleGenerate}
                disabled={isGenerating}
                className="rounded-md bg-blue-700 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed px-3 py-2 text-sm font-semibold text-white flex items-center gap-2"
              >
                {isGenerating && (
                  <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                )}
                {isGenerating ? 'Generating…' : 'Generate Caption'}
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
}
