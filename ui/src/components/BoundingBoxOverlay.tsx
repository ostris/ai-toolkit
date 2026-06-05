import { useRef, useState } from 'react';
import classNames from 'classnames';

// Per-box color so multiple boxes are individually distinguishable (not all one
// color). Hue is spread by the golden-angle so adjacent indices look distinct;
// stable per index so the same box keeps its color across the editor/overlay and
// matches its #n chip in the sidebar.
export function boxColor(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}, 80%, 60%)`;
}

// Same hue as boxColor but with an alpha, for tinting backgrounds (e.g. the
// sidebar element cards) so they visually match their box on the image.
export function boxColorAlpha(index: number, alpha: number): string {
  const hue = (index * 137.508) % 360;
  return `hsla(${hue.toFixed(1)}, 80%, 60%, ${alpha})`;
}

// A single bbox parsed from an Ideogram-style caption/prompt. Stored coords are
// normalized 0-1000 in [y1, x1, y2, x2] order (top-left origin).
export interface OverlayBox {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
  label: string;
  type: 'obj' | 'text';
}

// Same as OverlayBox but carries the element's index in the caption's element
// array, so edits can be written back to the right element.
export interface EditableBox extends OverlayBox {
  elementIndex: number;
}

// Returns the list of boxes if the text is an Ideogram bbox-JSON caption/prompt
// with at least one bbox, otherwise null (normal captions/prompts get no overlay).
export function parseBoundingBoxes(text: string): OverlayBox[] | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) return null;
  let data: any;
  try {
    data = JSON.parse(trimmed);
  } catch {
    return null;
  }
  const elements = data?.compositional_deconstruction?.elements;
  if (!Array.isArray(elements)) return null;
  const boxes: OverlayBox[] = [];
  for (const el of elements) {
    const bb = el?.bbox;
    if (Array.isArray(bb) && bb.length === 4 && bb.every((n: any) => typeof n === 'number')) {
      const isText = el.type === 'text';
      const label = (isText ? el.text : el.desc) ?? '';
      boxes.push({ y1: bb[0], x1: bb[1], y2: bb[2], x2: bb[3], label: `${label}`, type: isText ? 'text' : 'obj' });
    }
  }
  return boxes.length > 0 ? boxes : null;
}

// Build editable boxes (tagged with their element index) from an already-parsed
// caption object. Returns [] if the object has no usable elements/boxes.
export function extractBoxes(data: any): EditableBox[] {
  const elements = data?.compositional_deconstruction?.elements;
  if (!Array.isArray(elements)) return [];
  const boxes: EditableBox[] = [];
  elements.forEach((el: any, idx: number) => {
    const bb = el?.bbox;
    if (Array.isArray(bb) && bb.length === 4 && bb.every((n: any) => typeof n === 'number')) {
      const isText = el.type === 'text';
      const label = (isText ? el.text : el.desc) ?? '';
      boxes.push({
        y1: bb[0],
        x1: bb[1],
        y2: bb[2],
        x2: bb[3],
        label: `${label}`,
        type: isText ? 'text' : 'obj',
        elementIndex: idx,
      });
    }
  });
  return boxes;
}

// Parse the caption for EDITING: returns the parsed object plus its boxes. null if
// the text is not an Ideogram bbox-JSON with at least one box.
export function parseCaptionForEditing(text: string): { data: any; boxes: EditableBox[] } | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{')) return null;
  let data: any;
  try {
    data = JSON.parse(trimmed);
  } catch {
    return null;
  }
  const boxes = extractBoxes(data);
  return boxes.length > 0 ? { data, boxes } : null;
}

export interface BoxCoords {
  y1: number;
  x1: number;
  y2: number;
  x2: number;
}

// Clamp/round/normalize coords to integer 0-1000 with y1<y2, x1<x2.
function normalizeBox(b: { x1: number; y1: number; x2: number; y2: number }): BoxCoords {
  const cl = (v: number) => Math.max(0, Math.min(1000, Math.round(v)));
  const x1 = cl(Math.min(b.x1, b.x2));
  const x2 = cl(Math.max(b.x1, b.x2));
  const y1 = cl(Math.min(b.y1, b.y2));
  const y2 = cl(Math.max(b.y1, b.y2));
  return { y1, x1, y2, x2 };
}

type DragMode = 'move' | 'nw' | 'ne' | 'sw' | 'se';
const MIN_SIZE = 8; // minimum normalized box span so boxes can't collapse

// Editable overlay: click to select, drag a box body to move, drag a corner
// handle to resize, ✕ to delete. With `drawing` on, drag on empty space to
// create a new box. All geometry commits fire on pointer-up. Place inside the
// same `relative` image-wrapping container as BoundingBoxOverlay; coordinate
// math uses this layer's own rect, so it's correct at any zoom.
export function BoundingBoxEditor({
  boxes,
  selectedIndex,
  drawing,
  onSelect,
  onChangeBox,
  onCreateBox,
}: {
  boxes: EditableBox[];
  selectedIndex: number | null;
  drawing: boolean;
  onSelect: (elementIndex: number | null) => void;
  onChangeBox: (elementIndex: number, box: BoxCoords) => void;
  onCreateBox: (box: BoxCoords) => void;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<{ index: number; box: OverlayBox } | null>(null);
  const [drawDraft, setDrawDraft] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);

  const toNorm = (clientX: number, clientY: number) => {
    const rect = rootRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1000, ((clientX - rect.left) / rect.width) * 1000)),
      y: Math.max(0, Math.min(1000, ((clientY - rect.top) / rect.height) * 1000)),
    };
  };

  const startDrag = (e: React.PointerEvent, box: EditableBox, mode: DragMode) => {
    // Stop the event from reaching the zoom/pan wrapper so dragging never pans.
    e.preventDefault();
    e.stopPropagation();
    onSelect(box.elementIndex);
    const root = rootRef.current;
    if (!root) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const start: OverlayBox = { ...box };
    let current: OverlayBox = { ...box };

    const move = (ev: PointerEvent) => {
      const rect = root.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const dx = ((ev.clientX - startX) / rect.width) * 1000;
      const dy = ((ev.clientY - startY) / rect.height) * 1000;
      let { x1, y1, x2, y2 } = start;
      if (mode === 'move') {
        const w = x2 - x1;
        const h = y2 - y1;
        const nx1 = Math.max(0, Math.min(1000 - w, x1 + dx));
        const ny1 = Math.max(0, Math.min(1000 - h, y1 + dy));
        x1 = nx1;
        y1 = ny1;
        x2 = nx1 + w;
        y2 = ny1 + h;
      } else {
        if (mode.includes('w')) x1 = Math.max(0, Math.min(x2 - MIN_SIZE, x1 + dx));
        if (mode.includes('e')) x2 = Math.min(1000, Math.max(x1 + MIN_SIZE, x2 + dx));
        if (mode.includes('n')) y1 = Math.max(0, Math.min(y2 - MIN_SIZE, y1 + dy));
        if (mode.includes('s')) y2 = Math.min(1000, Math.max(y1 + MIN_SIZE, y2 + dy));
      }
      current = { ...start, x1, y1, x2, y2 };
      setDraft({ index: box.elementIndex, box: current });
    };

    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      setDraft(null);
      onChangeBox(box.elementIndex, normalizeBox(current));
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  // Pointer-down on the empty root: draw a new box (drawing mode) or deselect.
  const onRootPointerDown = (e: React.PointerEvent) => {
    if (!drawing) {
      onSelect(null);
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    if (!rootRef.current) return;
    const p0 = toNorm(e.clientX, e.clientY);
    let cur = { x1: p0.x, y1: p0.y, x2: p0.x, y2: p0.y };
    setDrawDraft(cur);

    const move = (ev: PointerEvent) => {
      const p = toNorm(ev.clientX, ev.clientY);
      cur = { x1: p0.x, y1: p0.y, x2: p.x, y2: p.y };
      setDrawDraft({ ...cur });
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      setDrawDraft(null);
      const nb = normalizeBox(cur);
      if (nb.x2 - nb.x1 >= MIN_SIZE && nb.y2 - nb.y1 >= MIN_SIZE) onCreateBox(nb);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const handles: { mode: DragMode; cls: string }[] = [
    { mode: 'nw', cls: 'top-0 left-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize' },
    { mode: 'ne', cls: 'top-0 right-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize' },
    { mode: 'sw', cls: 'bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize' },
    { mode: 'se', cls: 'bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize' },
  ];

  return (
    <div
      ref={rootRef}
      onPointerDown={onRootPointerDown}
      className={classNames('absolute inset-0', drawing ? 'cursor-crosshair' : '')}
    >
      {boxes.map(box => {
        const pos = draft && draft.index === box.elementIndex ? draft.box : box;
        const selected = selectedIndex === box.elementIndex;
        const showSelected = selected && !drawing;
        const color = boxColor(box.elementIndex);
        return (
          <div
            key={box.elementIndex}
            onPointerDown={drawing ? undefined : e => startDrag(e, box, 'move')}
            className={classNames('absolute border-2', {
              'pointer-events-none': drawing,
              'cursor-move touch-none': !drawing,
              'ring-2 ring-blue-400': showSelected,
            })}
            style={{
              left: `${pos.x1 / 10}%`,
              top: `${pos.y1 / 10}%`,
              width: `${(pos.x2 - pos.x1) / 10}%`,
              height: `${(pos.y2 - pos.y1) / 10}%`,
              borderColor: showSelected ? '#ffffff' : color,
            }}
          >
            {box.label && (
              <span
                title={box.label}
                className="absolute top-0 left-0 max-w-full px-1 py-0.5 text-[9px] leading-tight font-medium whitespace-pre-line break-words line-clamp-2 text-gray-900 pointer-events-none"
                style={{ backgroundColor: color }}
              >
                {box.label}
              </span>
            )}
            {selected &&
              !drawing &&
              handles.map(h => (
                <div
                  key={h.mode}
                  onPointerDown={e => startDrag(e, { ...box, ...pos }, h.mode)}
                  className={classNames(
                    'absolute w-3 h-3 rounded-sm border border-gray-900 bg-white touch-none',
                    h.cls,
                  )}
                />
              ))}
          </div>
        );
      })}
      {drawDraft && (
        <div
          className="absolute border-2 border-dashed border-white bg-white/10 pointer-events-none"
          style={{
            left: `${Math.min(drawDraft.x1, drawDraft.x2) / 10}%`,
            top: `${Math.min(drawDraft.y1, drawDraft.y2) / 10}%`,
            width: `${Math.abs(drawDraft.x2 - drawDraft.x1) / 10}%`,
            height: `${Math.abs(drawDraft.y2 - drawDraft.y1) / 10}%`,
          }}
        />
      )}
    </div>
  );
}

// Absolute overlay layer of boxes. Place inside a `relative` container that wraps
// the image (e.g. inside react-zoom-pan-pinch's TransformComponent) so boxes
// track the image during zoom/pan. Coords are percentages of the frame, so this
// is resolution-independent. pointer-events-none keeps zoom/pan working through it.
export default function BoundingBoxOverlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {boxes.map((b, i) => {
        const color = boxColor(i);
        return (
          <div
            key={i}
            className="absolute border-2"
            style={{
              left: `${b.x1 / 10}%`,
              top: `${b.y1 / 10}%`,
              width: `${(b.x2 - b.x1) / 10}%`,
              height: `${(b.y2 - b.y1) / 10}%`,
              borderColor: color,
            }}
          >
            {b.label && (
              <span
                title={b.label}
                className="absolute top-0 left-0 max-w-full px-1 py-0.5 text-[9px] leading-tight font-medium whitespace-pre-line break-words line-clamp-3 text-gray-900"
                style={{ backgroundColor: color }}
              >
                {b.label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
