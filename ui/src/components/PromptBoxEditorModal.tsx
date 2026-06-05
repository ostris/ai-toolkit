'use client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react';
import { createGlobalState } from 'react-global-hooks';
import { SquareDashed, X } from 'lucide-react';
import classNames from 'classnames';
import { BoundingBoxEditor, extractBoxes } from './BoundingBoxOverlay';
import IdeogramCaptionSidebar, { isIdeogramCaption } from './IdeogramCaptionSidebar';

export interface PromptBoxEditorState {
  prompt: string; // current prompt text (plain or Ideogram JSON)
  aspectRatio?: string; // 'W:H' to shape the placeholder; defaults to 1:1
  title?: string;
  onApply: (newPrompt: string) => void;
}

export const promptBoxEditorState = createGlobalState<PromptBoxEditorState | null>(null);

export const openPromptBoxEditor = (state: PromptBoxEditorState) => {
  promptBoxEditorState.set(state);
};

function safeParse(text: string): any {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// Empty Ideogram caption skeleton, used to convert a plain prompt into a
// structured one so boxes/fields become editable.
function seedCaption(highLevel: string): string {
  return JSON.stringify(
    {
      high_level_description: highLevel.trim(),
      style_description: { aesthetics: '', lighting: '', photo: '', medium: '' },
      compositional_deconstruction: { background: '', elements: [] },
    },
    null,
    2,
  );
}

// Parse a 'W:H' string into a CSS aspect-ratio value. Falls back to 1/1.
function aspectStyle(aspectRatio?: string): string {
  if (!aspectRatio) return '1 / 1';
  const m = aspectRatio.trim().match(/^(\d+)\s*[:/]\s*(\d+)$/);
  if (!m) return '1 / 1';
  const w = parseInt(m[1], 10);
  const h = parseInt(m[2], 10);
  if (!w || !h) return '1 / 1';
  return `${w} / ${h}`;
}

const PromptBoxEditorModal: React.FC = () => {
  const [mounted, setMounted] = useState(false);
  const [modalInfo, setModalInfo] = promptBoxEditorState.use();
  const isOpen = modalInfo !== null;

  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);

  useEffect(() => setMounted(true), []);

  // Seed local state whenever the modal opens with a new prompt.
  useEffect(() => {
    const p = modalInfo?.prompt ?? '';
    setCaption(p);
    setSavedCaption(p);
    setSelectedBoxIndex(null);
    setIsDrawing(false);
    setShowBoxes(true);
  }, [modalInfo]);

  const isIdeogram = useMemo(() => isIdeogramCaption(caption), [caption]);
  const isDirty = caption.trim() !== savedCaption.trim();

  const apply = useCallback(() => {
    if (!modalInfo) return;
    modalInfo.onApply(caption);
    setSavedCaption(caption);
  }, [modalInfo, caption]);

  const onClose = useCallback(() => {
    // Persist edits on close so nothing is lost.
    if (modalInfo && caption.trim() !== savedCaption.trim()) {
      modalInfo.onApply(caption);
    }
    setModalInfo(null);
  }, [modalInfo, caption, savedCaption, setModalInfo]);

  // Mutate the caption JSON's element array, updating local state only. Returns
  // whatever the mutator returns.
  const editCaption = useCallback(
    (fn: (elements: any[], data: any) => any): any => {
      const data = safeParse(caption);
      if (!data) return undefined;
      const elements = data?.compositional_deconstruction?.elements;
      if (!Array.isArray(elements)) return undefined;
      const result = fn(elements, data);
      setCaption(JSON.stringify(data, null, 2));
      return result;
    },
    [caption],
  );

  const handleBoxChange = useCallback(
    (elementIndex: number, box: { y1: number; x1: number; y2: number; x2: number }) => {
      editCaption(els => {
        if (els[elementIndex]) els[elementIndex].bbox = [box.y1, box.x1, box.y2, box.x2];
      });
    },
    [editCaption],
  );

  const handleCreateBox = useCallback(
    (box: { y1: number; x1: number; y2: number; x2: number }) => {
      const newIndex = editCaption(els => {
        els.push({ type: 'obj', bbox: [box.y1, box.x1, box.y2, box.x2], desc: '' });
        return els.length - 1;
      });
      setSelectedBoxIndex(typeof newIndex === 'number' ? newIndex : null);
      setIsDrawing(false);
    },
    [editCaption],
  );

  const editBoxes = useMemo(() => extractBoxes(safeParse(caption)), [caption]);

  if (!mounted) return null;

  return createPortal(
    <Dialog open={isOpen} onClose={onClose} className="relative z-10">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />
      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-0 sm:p-4 text-center">
          <DialogPanel
            transition
            className="relative transform rounded-none sm:rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in w-full sm:w-auto sm:max-w-[95vw] sm:max-h-[95vh] data-closed:sm:translate-y-0 data-closed:sm:scale-95 flex flex-col sm:flex-row overflow-hidden"
          >
            {/* Layout preview area: a gradient stand-in for the (nonexistent) image */}
            <div className="relative flex-1 min-w-0 flex items-center justify-center bg-gray-900 overflow-hidden p-4 sm:p-8">
              <div
                className="relative h-[50vh] sm:h-[70vh] max-w-full rounded-lg overflow-hidden shadow-inner bg-gray-700"
                style={{ aspectRatio: aspectStyle(modalInfo?.aspectRatio) }}
              >
                {/* subtle grid so box positions are readable */}
                <div
                  className="absolute inset-0 opacity-20 pointer-events-none"
                  style={{
                    backgroundImage:
                      'linear-gradient(to right, rgba(255,255,255,0.4) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.4) 1px, transparent 1px)',
                    backgroundSize: '10% 10%',
                  }}
                />
                {!isIdeogram && (
                  <div className="absolute inset-0 flex items-center justify-center text-center px-6 pointer-events-none">
                    <span className="text-xs text-white/80">
                      No image — this is a layout preview.
                      <br />
                      Convert to a structured caption to place boxes.
                    </span>
                  </div>
                )}
                {isIdeogram && showBoxes && (
                  <BoundingBoxEditor
                    boxes={editBoxes}
                    selectedIndex={selectedBoxIndex}
                    drawing={isDrawing}
                    onSelect={setSelectedBoxIndex}
                    onChangeBox={handleBoxChange}
                    onCreateBox={handleCreateBox}
                  />
                )}
              </div>

              {/* Controls over the preview */}
              <div className="absolute top-2 right-2 flex items-center gap-2 z-20">
                {isIdeogram && (
                  <button
                    type="button"
                    onClick={() => {
                      const next = !showBoxes;
                      setShowBoxes(next);
                      if (!next) {
                        setSelectedBoxIndex(null);
                        setIsDrawing(false);
                      }
                    }}
                    title={showBoxes ? 'Hide bounding boxes' : 'Show & edit bounding boxes'}
                    className={classNames('bg-gray-900 rounded-full p-1 leading-[0px] hover:opacity-100', {
                      'opacity-100 text-blue-400': showBoxes,
                      'opacity-50': !showBoxes,
                    })}
                  >
                    <SquareDashed />
                  </button>
                )}
                <button
                  type="button"
                  onClick={onClose}
                  title="Close"
                  className="bg-gray-900 rounded-full p-1 leading-[0px] opacity-50 hover:opacity-100"
                >
                  <X />
                </button>
              </div>
            </div>

            {/* Right sidebar: structured caption editor (or plain prompt) */}
            <div className="bg-gray-950 w-full sm:w-96 shrink-0 flex flex-col gap-2 p-3 overflow-y-auto text-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-gray-300">{modalInfo?.title ?? 'Edit Prompt'}</span>
              </div>
              {isIdeogram ? (
                <IdeogramCaptionSidebar
                  caption={caption}
                  onChange={setCaption}
                  selectedIndex={selectedBoxIndex}
                  onSelectIndex={i => {
                    setSelectedBoxIndex(i);
                    if (i != null) setShowBoxes(true);
                  }}
                  isDrawing={isDrawing}
                  onToggleDrawing={() => setIsDrawing(d => !d)}
                  onSave={apply}
                  isDirty={isDirty}
                />
              ) : (
                <div className="flex flex-col gap-2">
                  <textarea
                    className="w-full min-h-[12rem] rounded border-2 border-gray-700 bg-gray-900 text-gray-100 text-sm p-2 resize-none outline-none focus:border-blue-500"
                    placeholder="Enter prompt..."
                    value={caption}
                    onChange={e => setCaption(e.target.value)}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setCaption(seedCaption(caption));
                      setShowBoxes(true);
                    }}
                    className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md border border-purple-500 bg-purple-600/20 text-purple-200 hover:bg-purple-600/30 text-xs transition-colors"
                  >
                    <SquareDashed className="w-3.5 h-3.5" /> Convert to structured caption
                  </button>
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={apply}
                      disabled={!isDirty}
                      className={classNames('px-4 py-1.5 rounded-md border text-xs font-medium transition-colors', {
                        'bg-green-600 border-green-500 text-white hover:bg-green-500': isDirty,
                        'border-gray-700 text-gray-500 cursor-default': !isDirty,
                      })}
                    >
                      {isDirty ? 'Save' : 'Saved'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>,
    document.body,
  );
};

export default PromptBoxEditorModal;
