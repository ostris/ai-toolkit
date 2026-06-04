import classNames from 'classnames';

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

// Absolute overlay layer of boxes. Place inside a `relative` container that wraps
// the image (e.g. inside react-zoom-pan-pinch's TransformComponent) so boxes
// track the image during zoom/pan. Coords are percentages of the frame, so this
// is resolution-independent. pointer-events-none keeps zoom/pan working through it.
export default function BoundingBoxOverlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {boxes.map((b, i) => (
        <div
          key={i}
          className={classNames('absolute border', {
            'border-cyan-400': b.type === 'obj',
            'border-amber-400': b.type === 'text',
          })}
          style={{
            left: `${b.x1 / 10}%`,
            top: `${b.y1 / 10}%`,
            width: `${(b.x2 - b.x1) / 10}%`,
            height: `${(b.y2 - b.y1) / 10}%`,
          }}
        >
          {b.label && (
            <span
              title={b.label}
              className={classNames(
                'absolute top-0 left-0 max-w-full px-1 py-0.5 text-[9px] leading-tight font-medium whitespace-pre-line break-words line-clamp-3 text-gray-900',
                {
                  'bg-cyan-400/90': b.type === 'obj',
                  'bg-amber-400/90': b.type === 'text',
                },
              )}
            >
              {b.label}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
