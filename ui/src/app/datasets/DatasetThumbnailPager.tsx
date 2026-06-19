'use client';

import { useState } from 'react';
import { apiClient } from '@/utils/api';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import Link from 'next/link';

interface Props {
  datasetName: string;
  initialThumbs: string[];
  statsLoading: boolean;
  pageSize?: number;
}

interface CacheEntry {
  loading: boolean;
  paths: string[];
  loaded: boolean;
  error?: string;
}

const cache: Record<string, CacheEntry> = {};

export default function DatasetThumbnailPager({
  datasetName,
  initialThumbs,
  statsLoading,
  pageSize = 6,
}: Props) {
  const [page, setPage] = useState(0);
  const [, setTick] = useState(0);

  const entry: CacheEntry = cache[datasetName] || {
    loading: false,
    paths: initialThumbs,
    loaded: false,
  };
  // Keep the cache hydrated from props for the very first render.
  if (!cache[datasetName]) cache[datasetName] = entry;

  // Keep cache.paths in sync with initialThumbs until a full fetch happens.
  if (!entry.loaded && entry.paths.length < initialThumbs.length) {
    entry.paths = initialThumbs;
  }

  const fetchAll = async () => {
    if (entry.loading || entry.loaded) return;
    entry.loading = true;
    setTick(t => t + 1);
    try {
      const res = await apiClient.post('/api/datasets/listImages', { datasetName });
      const imgs: { img_path: string }[] = res.data?.images || [];
      // Filter to still-image extensions for previews; videos/audio not previewable here.
      const stillExts = ['.png', '.jpg', '.jpeg', '.webp'];
      const paths = imgs
        .map(i => i.img_path)
        .filter(p => stillExts.some(ext => p.toLowerCase().endsWith(ext)));
      entry.paths = paths.length > 0 ? paths : entry.paths;
      entry.loaded = true;
    } catch (err: any) {
      entry.error = err?.response?.data?.error || 'Failed to load images';
    } finally {
      entry.loading = false;
      setTick(t => t + 1);
    }
  };

  const totalKnown = entry.paths.length;
  const totalPages = Math.max(1, Math.ceil(totalKnown / pageSize));
  const startIdx = page * pageSize;
  const visible = entry.paths.slice(startIdx, startIdx + pageSize);

  const goLeft = () => {
    if (page > 0) {
      setPage(p => p - 1);
    }
  };

  const goRight = async () => {
    // If we're near the end of what we have, fetch the full set so
    // "next" can keep going past the stats-provided thumbs.
    if (!entry.loaded && page + 1 >= totalPages - 1) {
      await fetchAll();
    }
    setPage(p => {
      const newTotal = Math.max(1, Math.ceil(entry.paths.length / pageSize));
      return Math.min(p + 1, newTotal - 1);
    });
  };

  if (statsLoading && totalKnown === 0) {
    return <span className="text-gray-500 text-xs">…</span>;
  }
  if (totalKnown === 0) {
    return <span className="text-gray-500 text-xs italic">no images</span>;
  }

  const canGoLeft = page > 0;
  // We can always try to go right unless we know we're on the last page of the full list.
  const canGoRight = entry.loaded ? page < totalPages - 1 : true;
  const hasMore = !entry.loaded;

  return (
    <div className="flex items-center gap-2 py-1">
      <button
        type="button"
        onClick={goLeft}
        disabled={!canGoLeft}
        className="text-gray-400 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed"
        title="Previous page"
      >
        <ChevronLeft className="w-5 h-5" />
      </button>

      <Link href={`/datasets/${datasetName}`} className="flex flex-wrap gap-1 flex-1">
        {visible.map(p => (
          <img
            key={p}
            src={`/api/img/${encodeURIComponent(p)}`}
            alt=""
            loading="lazy"
            className="h-14 w-14 object-cover rounded border border-gray-800 flex-shrink-0"
          />
        ))}
        {entry.loading && (
          <span className="h-14 w-14 flex items-center justify-center text-gray-400">
            <Loader2 className="w-5 h-5 animate-spin" />
          </span>
        )}
      </Link>

      <button
        type="button"
        onClick={goRight}
        disabled={!canGoRight || entry.loading}
        className="text-gray-400 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed"
        title={hasMore ? 'Next (loads full list)' : 'Next page'}
      >
        <ChevronRight className="w-5 h-5" />
      </button>

      <div className="text-[11px] text-gray-500 tabular-nums whitespace-nowrap min-w-[64px] text-right">
        {entry.loading ? (
          'loading…'
        ) : (
          <>
            {startIdx + 1}–{Math.min(startIdx + pageSize, totalKnown)} / {totalKnown}
            {hasMore && '+'}
          </>
        )}
      </div>
    </div>
  );
}
