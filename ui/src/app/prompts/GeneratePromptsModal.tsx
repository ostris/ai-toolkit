'use client';

import { useMemo, useState } from 'react';
import { X, Sparkles, Shuffle } from 'lucide-react';
import classNames from 'classnames';

export interface GeneratedPromptDraft {
  text: string;
  title?: string;
  tags: string[];
  setIds: string[];
  width?: number;
  height?: number;
  seed?: number;
  network_multiplier?: number;
}

export interface PromptSet {
  id: string;
  name: string;
  color?: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  onGenerate: (drafts: GeneratedPromptDraft[]) => void;
  sets: PromptSet[];
}

// ───────── style library ─────────
interface StylePack {
  id: string;
  label: string;
  description: string;
  modifiers: string[];
  subjects: string[];
  scenes: string[];
  // Hint for safe-for-work / mature framing
  mature?: boolean;
}

const STYLES: StylePack[] = [
  {
    id: 'abstract',
    label: 'Abstract',
    description: 'Non-representational shapes, color fields, geometry',
    modifiers: [
      'flat color blocks, geometric composition',
      'bauhaus shapes, primary palette, hard edges',
      'fluid ink bleeds, marbled surfaces',
      'cubist fragmentation, overlapping planes',
      'risograph print, two-color overlay',
      'kandinsky-inspired geometry, dancing lines',
    ],
    subjects: ['a figure', 'two figures', 'a face', 'a cityscape', 'a still life arrangement'],
    scenes: [
      'reduced to overlapping color planes',
      'rendered as concentric shapes',
      'broken into a grid of color swatches',
      'dissolving into noise textures',
    ],
  },
  {
    id: 'retro',
    label: 'Retro',
    description: '70s/80s film grain, faded color, analog vibe',
    modifiers: [
      '1970s film grain, faded kodachrome',
      '1980s polaroid, soft flash, light leaks',
      'VHS scan lines, low saturation, motion blur',
      'shag carpet decor, warm tungsten light',
      'vintage magazine ad aesthetic',
    ],
    subjects: ['a person', 'a couple', 'a family', 'a teen', 'a musician'],
    scenes: [
      'standing in a wood-paneled living room',
      'leaning against a station wagon at a gas station',
      'at a diner booth with neon signage',
      'on a roller rink under disco lights',
      'in a sunlit kitchen with floral wallpaper',
    ],
  },
  {
    id: 'wacky',
    label: 'Wacky',
    description: 'Surreal, absurd, cartoon physics, oddly specific props',
    modifiers: [
      'tilt-shift, fish-eye lens, exaggerated perspective',
      'rubber-hose cartoon physics, bouncy proportions',
      'oversaturated pop colors, googly eyes',
      'collage of clip-art textures, hand-cut paper edges',
      'breakdancing pose mid-air with spaghetti props',
    ],
    subjects: [
      'a giant rubber chicken',
      'a pug wearing a tuxedo',
      'a snail driving a tiny convertible',
      'an octopus playing chess against itself',
      'a sandwich with arms running for office',
    ],
    scenes: [
      'in the middle of a corporate boardroom presenting a pie chart',
      'racing down a bowling lane on a skateboard',
      'getting interviewed on a morning news show',
      'launching from a slingshot into a pool of pudding',
    ],
  },
  {
    id: 'fantasy',
    label: 'Fantasy',
    description: 'High-fantasy, mythic, painterly, dramatic light',
    modifiers: [
      'painterly oils, volumetric god rays',
      'frank frazetta inspired, dramatic backlighting',
      'storybook illustration, soft edges, vignette',
      'D&D campaign cover art, cinematic composition',
      'arcane glow, particle effects, depth fog',
    ],
    subjects: [
      'an elven ranger',
      'a stoic paladin in plate armor',
      'a hooded wizard',
      'a dragon-rider',
      'a forest spirit',
    ],
    scenes: [
      'overlooking a valley of ruined towers at dusk',
      'crossing a bridge of bone over a glowing chasm',
      'meditating in a moss-covered library',
      'parleying with a sphinx at the cliff edge',
      'channeling lightning beside a runestone',
    ],
  },
  {
    id: 'futuristic',
    label: 'Futuristic',
    description: 'Sci-fi, cyberpunk, neon, hard-surface tech',
    modifiers: [
      'cyberpunk neon haze, anamorphic lens flares',
      'sleek minimalist architecture, polished concrete',
      'holographic UI overlays, volumetric fog',
      'blade-runner color grade, rain reflections',
      'kitbash hard-surface mech panels',
    ],
    subjects: [
      'a courier with augmented forearm',
      'a hacker in mirrored visor',
      'an android barista',
      'a corporate exec in carbon-weave suit',
      'a street samurai',
    ],
    scenes: [
      'crossing a neon-soaked Tokyo intersection in heavy rain',
      'in a rooftop noodle bar above a megastructure',
      'on a transit platform with hovering drones',
      'inside a glassy atrium with holographic koi',
      'on a rusting orbital station window-side',
    ],
  },
  {
    id: 'spicy',
    label: 'Spicy',
    description: 'Tasteful glamour / boudoir / flirty editorial (SFW-leaning)',
    mature: true,
    modifiers: [
      'editorial fashion lighting, low key',
      'boudoir window-light, soft shadows',
      'silk and lace texture detail, shallow depth of field',
      'noir nightclub palette, smoke haze',
      'high-fashion magazine cover, glossy retouch',
    ],
    subjects: ['a model', 'a dancer', 'a singer', 'a lounge guest'],
    scenes: [
      'reclining on a velvet chaise in a candlelit room',
      'leaning against a backstage mirror with bulb lights',
      'at a hotel balcony at golden hour',
      'in a smoky jazz club booth',
      'silhouetted in a doorway with backlight',
    ],
  },
  {
    id: 'minimalist',
    label: 'Minimalist',
    description: 'Empty space, single subject, soft palette',
    modifiers: [
      'negative space, off-center composition, pastel palette',
      'studio backdrop, single key light',
      'paper cut-out aesthetic, two flat colors',
      'line art, no shading, weight variation',
    ],
    subjects: ['a coffee cup', 'a chair', 'a cyclist', 'a runner', 'a reader'],
    scenes: [
      'against an off-white wall with one shadow',
      'centered on a beige seamless background',
      'isolated on a soft gradient',
    ],
  },
  {
    id: 'cinematic',
    label: 'Cinematic',
    description: 'Movie-still framing, anamorphic, color grade',
    modifiers: [
      'anamorphic 2.39:1, cinematic color grade',
      'kodak vision3 stock, halation, gentle grain',
      'roger deakins lighting, single source key',
      'rim light, atmospheric haze, lens flares',
    ],
    subjects: ['a detective', 'a astronaut', 'a chef', 'a soldier', 'a lone traveler'],
    scenes: [
      'walking down a rain-slicked alley at night',
      'staring out of a diner window at dawn',
      'silhouetted against a desert horizon',
      'standing on a windswept rooftop',
    ],
  },
];

// ───────── gender prompt fragments ─────────
const GENDER_FRAGMENTS: Record<string, { subjectPrefix: string; tag: string }> = {
  any: { subjectPrefix: '', tag: '' },
  female: { subjectPrefix: 'a woman, ', tag: 'female' },
  male: { subjectPrefix: 'a man, ', tag: 'male' },
  nonbinary: { subjectPrefix: 'an androgynous person, ', tag: 'nonbinary' },
  couple: { subjectPrefix: 'a couple, ', tag: 'couple' },
  group: { subjectPrefix: 'a group of people, ', tag: 'group' },
};

const GENDER_OPTIONS = [
  { value: 'any', label: 'Any / unspecified' },
  { value: 'female', label: 'Female' },
  { value: 'male', label: 'Male' },
  { value: 'nonbinary', label: 'Non-binary' },
  { value: 'couple', label: 'Couple' },
  { value: 'group', label: 'Group' },
];

// ───────── common preset dimensions ─────────
const DIMENSION_PRESETS = [
  { label: 'Square 1024', w: 1024, h: 1024 },
  { label: 'Square 768', w: 768, h: 768 },
  { label: 'Portrait 832×1216', w: 832, h: 1216 },
  { label: 'Landscape 1216×832', w: 1216, h: 832 },
  { label: 'Wide 1536×640', w: 1536, h: 640 },
  { label: 'Inherit (blank)', w: 0, h: 0 },
];

// ───────── model arch presets (label-only; gets stored as a tag) ─────────
const MODEL_TAGS = [
  'FLUX',
  'SDXL',
  'SD1.5',
  'Qwen-Image',
  'HiDream',
  'Wan 2.1',
  'Wan 2.2',
  'LTX',
  'Chroma',
];

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomSeed(): number {
  return Math.floor(Math.random() * 2147483647);
}

function buildPrompt(style: StylePack, gender: string): string {
  const fragment = GENDER_FRAGMENTS[gender] || GENDER_FRAGMENTS.any;
  const subject = fragment.subjectPrefix
    ? `${fragment.subjectPrefix}${pickRandom(style.scenes)}`
    : `${pickRandom(style.subjects)} ${pickRandom(style.scenes)}`;
  const modifier = pickRandom(style.modifiers);
  return `${subject}, ${modifier}`;
}

export default function GeneratePromptsModal({ open, onClose, onGenerate, sets }: Props) {
  const [count, setCount] = useState(8);
  const [selectedStyles, setSelectedStyles] = useState<string[]>(['cinematic', 'futuristic']);
  const [selectedGenders, setSelectedGenders] = useState<string[]>(['any']);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [dimensionPreset, setDimensionPreset] = useState<string>('Square 1024');
  const [customW, setCustomW] = useState<string>('');
  const [customH, setCustomH] = useState<string>('');
  const [includeSeed, setIncludeSeed] = useState(false);
  const [loraScale, setLoraScale] = useState<string>('');
  const [setId, setSetId] = useState<string>('');
  const [includeMature, setIncludeMature] = useState(false);

  const resolvedDims = useMemo(() => {
    if (customW && customH) {
      const w = parseInt(customW, 10);
      const h = parseInt(customH, 10);
      if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) return { w, h };
    }
    const preset = DIMENSION_PRESETS.find(p => p.label === dimensionPreset);
    if (!preset || (preset.w === 0 && preset.h === 0)) return null;
    return { w: preset.w, h: preset.h };
  }, [dimensionPreset, customW, customH]);

  const toggleArr = (arr: string[], v: string, setter: (next: string[]) => void) => {
    setter(arr.includes(v) ? arr.filter(x => x !== v) : [...arr, v]);
  };

  if (!open) return null;

  const availableStyles = STYLES.filter(s => includeMature || !s.mature);

  const generate = () => {
    const usableStyles = STYLES.filter(s => selectedStyles.includes(s.id));
    if (usableStyles.length === 0) {
      alert('Pick at least one style.');
      return;
    }
    const genders = selectedGenders.length > 0 ? selectedGenders : ['any'];
    const loraNum = loraScale.trim() === '' ? undefined : Number(loraScale);

    const drafts: GeneratedPromptDraft[] = [];
    for (let i = 0; i < count; i++) {
      const style = pickRandom(usableStyles);
      const gender = pickRandom(genders);
      const text = buildPrompt(style, gender);
      const tags = [style.id];
      const genderTag = GENDER_FRAGMENTS[gender]?.tag;
      if (genderTag) tags.push(genderTag);
      selectedModels.forEach(m => tags.push(m.toLowerCase().replace(/[^a-z0-9]+/g, '-')));

      const draft: GeneratedPromptDraft = {
        text,
        title: `${style.label} #${i + 1}`,
        tags,
        setIds: setId ? [setId] : [],
      };
      if (resolvedDims) {
        draft.width = resolvedDims.w;
        draft.height = resolvedDims.h;
      }
      if (includeSeed) draft.seed = randomSeed();
      if (loraNum !== undefined && Number.isFinite(loraNum)) draft.network_multiplier = loraNum;
      drafts.push(draft);
    }

    onGenerate(drafts);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg w-[95vw] max-w-3xl max-h-[90vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center px-4 py-3 border-b border-gray-800">
          <Sparkles className="w-5 h-5 text-amber-400 mr-2" />
          <div className="text-lg flex-1">Generate Prompts</div>
          <button className="text-gray-300 hover:text-white" onClick={onClose}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 space-y-4 text-sm">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-xs text-gray-400 mb-1">Number of prompts</div>
              <input
                type="number"
                min={1}
                max={100}
                value={count}
                onChange={e => setCount(Math.max(1, Math.min(100, parseInt(e.target.value) || 1)))}
                className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              />
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Add to set (optional)</div>
              <select
                value={setId}
                onChange={e => setSetId(e.target.value)}
                className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              >
                <option value="">— no set —</option>
                {sets.map(s => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1 flex items-center justify-between">
              <span>Styles</span>
              <label className="flex items-center gap-1 text-[11px]">
                <input
                  type="checkbox"
                  checked={includeMature}
                  onChange={e => setIncludeMature(e.target.checked)}
                />
                Include "spicy" (tasteful editorial / boudoir, SFW-leaning)
              </label>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {availableStyles.map(s => {
                const sel = selectedStyles.includes(s.id);
                return (
                  <button
                    key={s.id}
                    onClick={() => toggleArr(selectedStyles, s.id, setSelectedStyles)}
                    className={classNames(
                      'border rounded p-2 text-left transition',
                      sel
                        ? 'border-blue-500 bg-blue-900/30 text-white'
                        : 'border-gray-700 bg-gray-950 text-gray-300 hover:border-gray-600',
                    )}
                  >
                    <div className="text-sm font-medium">{s.label}</div>
                    <div className="text-[11px] text-gray-400">{s.description}</div>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1">Subject / gender</div>
            <div className="flex flex-wrap gap-1">
              {GENDER_OPTIONS.map(g => {
                const sel = selectedGenders.includes(g.value);
                return (
                  <button
                    key={g.value}
                    onClick={() => toggleArr(selectedGenders, g.value, setSelectedGenders)}
                    className={classNames(
                      'text-xs px-2 py-1 rounded-full border',
                      sel
                        ? 'bg-blue-700 border-blue-500 text-white'
                        : 'bg-gray-950 border-gray-700 text-gray-300 hover:border-gray-600',
                    )}
                  >
                    {g.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1">Target model tags (optional)</div>
            <div className="flex flex-wrap gap-1">
              {MODEL_TAGS.map(m => {
                const sel = selectedModels.includes(m);
                return (
                  <button
                    key={m}
                    onClick={() => toggleArr(selectedModels, m, setSelectedModels)}
                    className={classNames(
                      'text-xs px-2 py-1 rounded-full border',
                      sel
                        ? 'bg-purple-700 border-purple-500 text-white'
                        : 'bg-gray-950 border-gray-700 text-gray-300 hover:border-gray-600',
                    )}
                  >
                    {m}
                  </button>
                );
              })}
            </div>
            <div className="text-[11px] text-gray-500 mt-1">
              Adds tags to each generated prompt so you can filter by target model later.
            </div>
          </div>

          <div>
            <div className="text-xs text-gray-400 mb-1">Image dimensions</div>
            <div className="flex flex-wrap gap-2 items-center">
              <select
                value={dimensionPreset}
                onChange={e => {
                  setDimensionPreset(e.target.value);
                  setCustomW('');
                  setCustomH('');
                }}
                className="bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              >
                {DIMENSION_PRESETS.map(p => (
                  <option key={p.label} value={p.label}>
                    {p.label}
                  </option>
                ))}
              </select>
              <span className="text-gray-500 text-xs">or custom:</span>
              <input
                type="text"
                inputMode="numeric"
                placeholder="width"
                value={customW}
                onChange={e => setCustomW(e.target.value.replace(/\D/g, ''))}
                className="w-24 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              />
              <span className="text-gray-500">×</span>
              <input
                type="text"
                inputMode="numeric"
                placeholder="height"
                value={customH}
                onChange={e => setCustomH(e.target.value.replace(/\D/g, ''))}
                className="w-24 bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <label className="flex items-center gap-2 text-gray-200">
              <input type="checkbox" checked={includeSeed} onChange={e => setIncludeSeed(e.target.checked)} />
              Assign a random seed to each
            </label>
            <div>
              <div className="text-xs text-gray-400 mb-1">LoRA scale (optional)</div>
              <input
                type="text"
                inputMode="decimal"
                value={loraScale}
                placeholder="e.g. 1.0"
                onChange={e => setLoraScale(e.target.value.replace(/[^0-9.\-]/g, ''))}
                className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-gray-100"
              />
            </div>
          </div>
        </div>

        <div className="px-4 py-3 border-t border-gray-800 flex items-center">
          <div className="text-xs text-gray-400 flex items-center gap-1">
            <Shuffle className="w-3 h-3" /> Random combinations of styles × subjects × scenes × modifiers.
          </div>
          <div className="flex-1"></div>
          <button
            onClick={onClose}
            className="px-3 py-1 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded mr-2"
          >
            Cancel
          </button>
          <button
            onClick={generate}
            className="px-3 py-1 text-sm text-white bg-amber-600 hover:bg-amber-500 rounded flex items-center gap-1"
          >
            <Sparkles className="w-3 h-3" /> Generate {count}
          </button>
        </div>
      </div>
    </div>
  );
}
