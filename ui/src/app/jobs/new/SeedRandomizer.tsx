'use client';

import { useState } from 'react';
import { Button } from '@headlessui/react';
import { LuDices, LuChevronDown, LuChevronUp } from 'react-icons/lu';
import { NumberInput } from '@/components/formInputs';

interface SeedRandomizerProps {
  value: number;
  onChange: (value: number) => void;
}

/**
 * Seed input + a dice button that rolls a random integer in a user-supplied
 * range and writes it to the seed field.
 *
 * The min/max range itself is UI-only (not persisted to the job config); the
 * underlying sample.seed is still a single integer that the trainer uses for
 * every sample of the run (unless walk_seed is enabled, which auto-increments
 * the seed per prompt). For true "fresh random per sample at training time"
 * we'd need a Python-side change to BaseSDTrainProcess's sample loop.
 */
export default function SeedRandomizer({ value, onChange }: SeedRandomizerProps) {
  const [rangeOpen, setRangeOpen] = useState(false);
  // Default range covers the common "I want a few-digit seed" use case.
  const [rangeMin, setRangeMin] = useState<number>(0);
  const [rangeMax, setRangeMax] = useState<number>(999999);

  const rollSeed = () => {
    const lo = Math.max(0, Math.floor(Math.min(rangeMin, rangeMax)));
    const hi = Math.max(lo, Math.floor(Math.max(rangeMin, rangeMax)));
    // inclusive on both ends
    const next = lo + Math.floor(Math.random() * (hi - lo + 1));
    onChange(next);
  };

  return (
    <div>
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <NumberInput
            label="Seed"
            value={value}
            onChange={(v: number | null) => onChange(v ?? 0)}
            placeholder="eg. 0"
            min={0}
            required
          />
        </div>
        <Button
          type="button"
          onClick={rollSeed}
          title={`Roll a random seed in [${rangeMin}, ${rangeMax}]`}
          className="flex items-center gap-1 rounded-md border border-gray-600 bg-gray-800 px-2.5 py-1.5 text-xs text-gray-100 hover:bg-gray-700"
        >
          <LuDices className="h-4 w-4" />
          Randomize
        </Button>
      </div>
      <Button
        type="button"
        onClick={() => setRangeOpen(v => !v)}
        className="mt-1 inline-flex items-center gap-1 text-[11px] text-gray-400 hover:text-gray-200"
      >
        {rangeOpen ? <LuChevronUp className="h-3 w-3" /> : <LuChevronDown className="h-3 w-3" />}
        Random seed range
      </Button>
      {rangeOpen && (
        <div className="mt-1 rounded-md border border-gray-700 bg-gray-950/40 px-2 py-2">
          <div className="grid grid-cols-2 gap-2">
            <NumberInput
              label="Min"
              value={rangeMin}
              onChange={(v: number | null) => setRangeMin(v ?? 0)}
              placeholder="0"
              min={0}
            />
            <NumberInput
              label="Max"
              value={rangeMax}
              onChange={(v: number | null) => setRangeMax(v ?? 0)}
              placeholder="999999"
              min={0}
            />
          </div>
          <p className="mt-1 text-[10px] text-gray-500">
            Click <strong>Randomize</strong> to roll a seed in [{rangeMin}, {rangeMax}]. Range is UI-only — the
            saved seed is the rolled value, not the range.
          </p>
        </div>
      )}
    </div>
  );
}
