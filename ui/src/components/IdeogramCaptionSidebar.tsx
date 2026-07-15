'use client';
import { useMemo, useState } from 'react';
import classNames from 'classnames';
import { Plus, Trash2, X, SquareDashed } from 'lucide-react';
import { boxColor, boxColorAlpha } from './BoundingBoxOverlay';

// Detects an Ideogram structured caption (the distinctive marker is the
// compositional_deconstruction block). Used by the viewer to decide whether to
// show this form editor instead of a plain textarea.
export function isIdeogramCaption(text: string): boolean {
  const t = text.trim();
  if (!t.startsWith('{')) return false;
  try {
    const d = JSON.parse(t);
    return !!d && typeof d === 'object' && typeof d.compositional_deconstruction === 'object';
  } catch {
    return false;
  }
}

// Official medium tokens. `photograph` uses the `photo` style key; every other
// medium uses `art_style`. Users may also type a custom medium.
const MEDIUM_OPTIONS = ['photograph', 'illustration', '3d_render', 'painting', 'graphic_design'];

// Map old/variant medium spellings (e.g. our old "Illustration." / "3D render.")
// to a canonical token so the dropdown can match them; unknown values stay custom.
function canonMedium(m: string): string {
  const key = (m || '').trim().replace(/\.+$/, '').trim().toLowerCase();
  const aliases: Record<string, string> = {
    photograph: 'photograph',
    photo: 'photograph',
    illustration: 'illustration',
    '3d render': '3d_render',
    '3d_render': '3d_render',
    '3d-render': '3d_render',
    '3drender': '3d_render',
    render: '3d_render',
    '3d': '3d_render',
    painting: 'painting',
    'graphic design': 'graphic_design',
    graphic_design: 'graphic_design',
    'graphic-design': 'graphic_design',
    graphic: 'graphic_design',
  };
  return aliases[key] ?? (m || '').trim();
}

// Photograph branch uses `photo`; everything else uses `art_style`. An unknown
// (empty/custom) medium defaults to the photo branch.
function isPhotoMedium(m: string): boolean {
  const c = canonMedium(m);
  if (c === 'photograph') return true;
  if (MEDIUM_OPTIONS.includes(c)) return false;
  return true;
}

// Normalize a color to an UPPERCASE #RRGGBB value (expands #RGB). Returns
// '#000000' for anything unparseable (used to feed <input type=color>).
function toHex6(c: string): string {
  const s = (c || '').trim();
  if (/^#[0-9a-fA-F]{6}$/.test(s)) return '#' + s.slice(1).toUpperCase();
  if (/^#[0-9a-fA-F]{3}$/.test(s)) {
    return (
      '#' +
      s
        .slice(1)
        .split('')
        .map(ch => ch + ch)
        .join('')
        .toUpperCase()
    );
  }
  return '#000000';
}

const clamp1000 = (n: number) => Math.max(0, Math.min(1000, Math.round(isNaN(n) ? 0 : n)));

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-gray-800 bg-gray-900/40 p-2.5">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-gray-400">{title}</div>
      {children}
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      {label && <span className="text-[10px] text-gray-400">{label}</span>}
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        spellCheck={false}
        onChange={e => onChange(e.target.value)}
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 outline-none focus:border-blue-500"
      />
    </label>
  );
}

function TextAreaField({
  label,
  value,
  onChange,
  rows = 3,
  placeholder,
}: {
  label?: string;
  value: string;
  onChange: (v: string) => void;
  rows?: number;
  placeholder?: string;
}) {
  return (
    <label className="flex flex-col gap-0.5">
      {label && <span className="text-[10px] text-gray-400">{label}</span>}
      <textarea
        value={value}
        rows={rows}
        placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 outline-none focus:border-blue-500 resize-none"
      />
    </label>
  );
}

// Medium picker: a dropdown of the official tokens plus a "Custom…" escape hatch
// that reveals a free-text input. Recognizes old/variant spellings via canonMedium.
function MediumField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const canon = canonMedium(value);
  const known = MEDIUM_OPTIONS.includes(canon);
  const [custom, setCustom] = useState(!known && (value || '').trim() !== '');
  const showCustom = custom || (!known && (value || '').trim() !== '');
  return (
    <label className="flex flex-col gap-0.5">
      <span className="text-[10px] text-gray-400">Medium</span>
      <select
        value={showCustom ? '__custom__' : canon}
        onChange={e => {
          const v = e.target.value;
          if (v === '__custom__') setCustom(true);
          else {
            setCustom(false);
            onChange(v);
          }
        }}
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 outline-none focus:border-blue-500"
      >
        {MEDIUM_OPTIONS.map(m => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
        <option value="__custom__">Custom…</option>
      </select>
      {showCustom && (
        <input
          type="text"
          value={value}
          placeholder="custom medium"
          spellCheck={false}
          onChange={e => onChange(e.target.value)}
          className="mt-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 outline-none focus:border-blue-500"
        />
      )}
    </label>
  );
}

function ColorPalette({ colors, max, onChange }: { colors: string[]; max: number; onChange: (c: string[]) => void }) {
  const setAt = (i: number, v: string) => onChange(colors.map((c, idx) => (idx === i ? v : c)));
  const removeAt = (i: number) => onChange(colors.filter((_, idx) => idx !== i));
  return (
    <div className="flex flex-wrap gap-1.5 items-center">
      {colors.map((c, i) => (
        <div key={i} className="flex items-center gap-1 bg-gray-800 border border-gray-700 rounded px-1 py-0.5">
          <input
            type="color"
            value={toHex6(c)}
            onChange={e => setAt(i, toHex6(e.target.value))}
            className="w-5 h-5 rounded cursor-pointer bg-transparent border-0 p-0"
            title="Pick color"
          />
          <input
            type="text"
            value={c}
            spellCheck={false}
            onChange={e => setAt(i, e.target.value)}
            className="w-[58px] bg-transparent text-[10px] text-gray-200 outline-none font-mono"
          />
          <button
            type="button"
            onClick={() => removeAt(i)}
            className="text-gray-500 hover:text-rose-400"
            title="Remove color"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      ))}
      {colors.length < max && (
        <button
          type="button"
          onClick={() => onChange([...colors, '#888888'])}
          className="flex items-center gap-1 text-[10px] text-gray-400 hover:text-gray-200 border border-dashed border-gray-700 rounded px-1.5 py-1"
        >
          <Plus className="w-3 h-3" /> Add
        </button>
      )}
    </div>
  );
}

interface Props {
  caption: string;
  onChange: (next: string) => void;
  selectedIndex: number | null;
  onSelectIndex: (i: number | null) => void;
  isDrawing: boolean;
  onToggleDrawing: () => void;
  onSave: () => void;
  isDirty: boolean;
}

export default function IdeogramCaptionSidebar({
  caption,
  onChange,
  selectedIndex,
  onSelectIndex,
  isDrawing,
  onToggleDrawing,
  onSave,
  isDirty,
}: Props) {
  const [showRaw, setShowRaw] = useState(false);

  const data = useMemo(() => {
    try {
      return JSON.parse(caption);
    } catch {
      return null;
    }
  }, [caption]);

  // Parse fresh from the current caption, mutate, and emit pretty JSON so the
  // string remains the single source of truth (shared with the image editor).
  const update = (mutator: (d: any) => void) => {
    let d: any;
    try {
      d = JSON.parse(caption);
    } catch {
      return;
    }
    mutator(d);
    onChange(JSON.stringify(d, null, 2));
  };

  if (!data) {
    return (
      <div className="rounded border border-rose-700 bg-rose-950/30 p-2 text-xs text-rose-300">
        Caption is not valid JSON, so the form editor is unavailable. Fix it in the raw text below.
        <textarea
          value={caption}
          onChange={e => onChange(e.target.value)}
          rows={8}
          className="mt-2 w-full bg-gray-800 border border-gray-700 rounded p-2 text-gray-100 outline-none resize-none"
        />
      </div>
    );
  }

  const style = data.style_description || {};
  const decon = data.compositional_deconstruction || {};
  const elements: any[] = Array.isArray(decon.elements) ? decon.elements : [];

  const setStyle = (key: string, value: string) =>
    update(d => {
      d.style_description = { ...(d.style_description || {}), [key]: value };
    });

  const setStylePalette = (cols: string[]) =>
    update(d => {
      const sd = { ...(d.style_description || {}) };
      if (cols.length) sd.color_palette = cols;
      else delete sd.color_palette;
      d.style_description = sd;
    });

  // Setting the medium may flip the branch (photograph ↔ non-photo). When it does,
  // migrate the render text from photo↔art_style so we never keep both keys.
  const setMedium = (value: string) =>
    update(d => {
      const sd = { ...(d.style_description || {}) };
      const wasPhoto = isPhotoMedium(sd.medium ?? '');
      const nowPhoto = isPhotoMedium(value);
      sd.medium = value;
      if (wasPhoto !== nowPhoto) {
        const val = sd.photo ?? sd.art_style ?? '';
        delete sd.photo;
        delete sd.art_style;
        if (nowPhoto) sd.photo = val;
        else sd.art_style = val;
      }
      d.style_description = sd;
    });

  // The single render field writes to `photo` or `art_style` per the current branch.
  const photoBranch = isPhotoMedium(style.medium ?? '');
  const renderValue = (photoBranch ? style.photo : style.art_style) ?? style.photo ?? style.art_style ?? '';
  const setRender = (value: string) =>
    update(d => {
      const sd = { ...(d.style_description || {}) };
      if (isPhotoMedium(sd.medium ?? '')) {
        sd.photo = value;
        delete sd.art_style;
      } else {
        sd.art_style = value;
        delete sd.photo;
      }
      d.style_description = sd;
    });

  const setElement = (i: number, mutator: (el: any) => void) =>
    update(d => {
      const els = d?.compositional_deconstruction?.elements;
      if (Array.isArray(els) && els[i]) mutator(els[i]);
    });

  const addElement = () => {
    update(d => {
      if (!d.compositional_deconstruction || typeof d.compositional_deconstruction !== 'object') {
        d.compositional_deconstruction = {};
      }
      if (!Array.isArray(d.compositional_deconstruction.elements)) {
        d.compositional_deconstruction.elements = [];
      }
      d.compositional_deconstruction.elements.push({ type: 'obj', desc: '' });
    });
    onSelectIndex(elements.length);
  };

  const removeElement = (i: number) => {
    update(d => {
      d?.compositional_deconstruction?.elements?.splice(i, 1);
    });
    onSelectIndex(null);
  };

  return (
    <div className="flex flex-col gap-4 text-sm">
      {/* Header — stays pinned while the form scrolls */}
      <div className="sticky -top-3 z-20 -mx-3 -mt-3 px-3 pt-3 pb-2 bg-gray-950/95 backdrop-blur border-b border-gray-800 flex items-center gap-2">
        <span className="text-xs font-semibold text-gray-200">Ideogram Caption</span>
        {isDirty && (
          <span
            className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_6px] shadow-blue-500/60"
            title="Unsaved changes"
          />
        )}
        <button
          type="button"
          onClick={onSave}
          disabled={!isDirty}
          className={classNames('ml-auto px-4 py-1.5 rounded-md border text-xs font-medium transition-colors', {
            'bg-green-600 border-green-500 text-white hover:bg-green-500 shadow-sm': isDirty,
            'border-gray-700 text-gray-500 cursor-default': !isDirty,
          })}
        >
          {isDirty ? 'Save' : 'Saved'}
        </button>
      </div>

      <TextAreaField
        label="High-level description"
        value={data.high_level_description ?? ''}
        onChange={v => update(d => (d.high_level_description = v))}
        rows={3}
        placeholder="One-sentence summary of the image..."
      />

      <Section title="Style">
        <TextField label="Aesthetics" value={style.aesthetics ?? ''} onChange={v => setStyle('aesthetics', v)} />
        <TextField label="Lighting" value={style.lighting ?? ''} onChange={v => setStyle('lighting', v)} />
        <MediumField value={style.medium ?? ''} onChange={setMedium} />
        <TextField
          label={photoBranch ? 'Photo (camera / film)' : 'Art style (rendering technique)'}
          value={renderValue}
          onChange={setRender}
          placeholder={photoBranch ? '35mm film still, shallow depth of field' : 'flat vector, clean edges'}
        />
        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-gray-400">Color palette (max 16)</span>
          <ColorPalette
            colors={Array.isArray(style.color_palette) ? style.color_palette : []}
            max={16}
            onChange={setStylePalette}
          />
        </div>
      </Section>

      <Section title="Background">
        <TextAreaField
          label=""
          value={decon.background ?? ''}
          onChange={v =>
            update(d => {
              d.compositional_deconstruction = { ...(d.compositional_deconstruction || {}), background: v };
            })
          }
          rows={4}
          placeholder="The scene shell: walls, floor, sky, ambient light..."
        />
      </Section>

      <Section title={`Elements (${elements.length})`}>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={addElement}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-gray-600 text-gray-300 hover:bg-gray-800 text-xs transition-colors"
          >
            <Plus className="w-3.5 h-3.5" /> Add element
          </button>
          <button
            type="button"
            onClick={onToggleDrawing}
            title="Draw a new box on the image"
            className={classNames('flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-xs transition-colors', {
              'bg-blue-600 border-blue-500 text-white': isDrawing,
              'border-gray-600 text-gray-300 hover:bg-gray-800': !isDrawing,
            })}
          >
            <SquareDashed className="w-3.5 h-3.5" /> {isDrawing ? 'Drawing… (esc to cancel)' : 'Draw box'}
          </button>
        </div>

        {elements.map((el, i) => {
          const isText = el.type === 'text';
          const selected = selectedIndex === i;
          const bbox = Array.isArray(el.bbox) && el.bbox.length === 4 ? el.bbox : null;
          const palette = Array.isArray(el.color_palette) ? el.color_palette : [];
          return (
            <div
              key={i}
              className={classNames('rounded-md border p-2.5 flex flex-col gap-2.5 transition-colors', {
                'ring-1 ring-blue-500/40': selected,
              })}
              style={{
                backgroundColor: boxColorAlpha(i, selected ? 0.22 : 0.12),
                borderColor: selected ? boxColor(i) : boxColorAlpha(i, 0.4),
              }}
            >
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => onSelectIndex(selected ? null : i)}
                  className={classNames(
                    'flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-md transition-colors',
                    {
                      'bg-blue-600 text-white': selected,
                      'bg-gray-800 text-gray-400 hover:text-gray-200': !selected,
                    },
                  )}
                  title="Select (highlights its box on the image)"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm border border-black/30"
                    style={{ backgroundColor: boxColor(i) }}
                  />
                  #{i + 1}
                </button>
                <div className="flex items-center rounded-md border border-gray-700 overflow-hidden">
                  <button
                    type="button"
                    onClick={() =>
                      setElement(i, e => {
                        e.type = 'obj';
                        delete e.text;
                      })
                    }
                    className={classNames('text-[11px] px-3 py-1 transition-colors', {
                      'bg-cyan-600 text-white': !isText,
                      'text-gray-300 hover:bg-gray-800': isText,
                    })}
                  >
                    obj
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      setElement(i, e => {
                        e.type = 'text';
                        if (e.text == null) e.text = '';
                      })
                    }
                    className={classNames('text-[11px] px-3 py-1 border-l border-gray-700 transition-colors', {
                      'bg-amber-600 text-white': isText,
                      'text-gray-300 hover:bg-gray-800': !isText,
                    })}
                  >
                    text
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => removeElement(i)}
                  className="ml-auto text-gray-500 hover:text-rose-400 transition-colors"
                  title="Delete element"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>

              {isText && (
                <TextAreaField
                  label="Text (rendered in image)"
                  value={el.text ?? ''}
                  onChange={v => setElement(i, e => (e.text = v))}
                  rows={2}
                />
              )}

              <TextAreaField
                label="Description"
                value={el.desc ?? ''}
                onChange={v => setElement(i, e => (e.desc = v))}
                rows={3}
              />

              <div className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-gray-400">Bounding box {bbox ? '' : '(none)'}</span>
                  {bbox ? (
                    <button
                      type="button"
                      onClick={() => setElement(i, e => delete e.bbox)}
                      className="text-[10px] text-rose-400 hover:text-rose-300"
                    >
                      remove
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setElement(i, e => (e.bbox = [250, 250, 750, 750]))}
                      className="text-[10px] text-blue-400 hover:text-blue-300"
                    >
                      + add box
                    </button>
                  )}
                </div>
                {bbox && (
                  <div className="grid grid-cols-4 gap-1">
                    {['y1', 'x1', 'y2', 'x2'].map((lbl, k) => (
                      <label key={k} className="flex flex-col">
                        <span className="text-[9px] text-gray-500">{lbl}</span>
                        <input
                          type="number"
                          min={0}
                          max={1000}
                          value={bbox[k]}
                          onChange={e =>
                            setElement(i, el2 => {
                              const bb = [...el2.bbox];
                              bb[k] = clamp1000(parseInt(e.target.value, 10));
                              el2.bbox = bb;
                            })
                          }
                          className="bg-gray-800 border border-gray-700 rounded px-1 py-0.5 text-[11px] text-gray-100 outline-none focus:border-blue-500"
                        />
                      </label>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-1">
                <span className="text-[10px] text-gray-400">Colors (max 5)</span>
                <ColorPalette
                  colors={palette}
                  max={5}
                  onChange={cols =>
                    setElement(i, e => {
                      if (cols.length) e.color_palette = cols;
                      else delete e.color_palette;
                    })
                  }
                />
              </div>
            </div>
          );
        })}
      </Section>

      {/* Advanced raw-JSON escape hatch */}
      <div className="border-t border-gray-800 pt-2">
        <button
          type="button"
          onClick={() => setShowRaw(s => !s)}
          className="text-[11px] text-gray-500 hover:text-gray-300"
        >
          {showRaw ? '▾ Hide raw JSON' : '▸ Raw JSON'}
        </button>
        {showRaw && (
          <textarea
            value={caption}
            onChange={e => onChange(e.target.value)}
            rows={10}
            spellCheck={false}
            className="mt-1 w-full bg-gray-800 border border-gray-700 rounded p-2 text-[11px] font-mono text-gray-100 outline-none resize-none"
          />
        )}
      </div>
    </div>
  );
}
