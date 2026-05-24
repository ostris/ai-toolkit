'use client';
import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

// Module-level batcher: many cards mount at once when the virtualized grid scrolls;
// instead of N HTTP requests, queue paths and flush them as a single batch.
type Resolver = { resolve: (caption: string) => void; reject: (err: unknown) => void };
const pending = new Map<string, Resolver[]>();
const cache = new Map<string, string>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_DELAY_MS = 30;
const MAX_BATCH = 200;

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(flush, FLUSH_DELAY_MS);
}

async function flush() {
  flushTimer = null;
  if (pending.size === 0) return;

  // Drain up to MAX_BATCH paths; if more arrived, reschedule.
  const paths: string[] = [];
  for (const path of pending.keys()) {
    paths.push(path);
    if (paths.length >= MAX_BATCH) break;
  }
  const batchResolvers = paths.map(p => ({ path: p, resolvers: pending.get(p)! }));
  for (const p of paths) pending.delete(p);

  try {
    const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths });
    const captions: Record<string, string> = res.data?.captions ?? {};
    for (const { path, resolvers } of batchResolvers) {
      const value = captions[path] ?? '';
      cache.set(path, value);
      for (const r of resolvers) r.resolve(value);
    }
  } catch (err) {
    for (const { resolvers } of batchResolvers) {
      for (const r of resolvers) r.reject(err);
    }
  }

  if (pending.size > 0) scheduleFlush();
}

function requestCaption(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const list = pending.get(path);
    if (list) {
      list.push({ resolve, reject });
    } else {
      pending.set(path, [{ resolve, reject }]);
    }
    scheduleFlush();
  });
}

export function invalidateCaption(path: string) {
  cache.delete(path);
}

export function setCachedCaption(path: string, caption: string) {
  cache.set(path, caption);
}

// Fetches caption for a path, using the module-level batcher + cache.
// `refreshKey` busts the cache (e.g. after external edits or auto-captioning poll).
export default function useCaptionBatch(imgPath: string | null, refreshKey: number = 0) {
  const [caption, setCaption] = useState<string>(() => (imgPath ? (cache.get(imgPath) ?? '') : ''));
  const [isLoaded, setIsLoaded] = useState<boolean>(() => Boolean(imgPath && cache.has(imgPath)));
  const lastPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imgPath) {
      setCaption('');
      setIsLoaded(false);
      return;
    }

    if (refreshKey > 0) invalidateCaption(imgPath);

    const cached = cache.get(imgPath);
    if (cached !== undefined) {
      setCaption(cached);
      setIsLoaded(true);
      lastPathRef.current = imgPath;
      return;
    }

    let cancelled = false;
    lastPathRef.current = imgPath;
    setIsLoaded(false);
    requestCaption(imgPath)
      .then(value => {
        if (cancelled || lastPathRef.current !== imgPath) return;
        setCaption(value);
        setIsLoaded(true);
      })
      .catch(err => {
        if (cancelled) return;
        console.error('Error fetching caption:', err);
        setIsLoaded(true);
      });

    return () => {
      cancelled = true;
    };
  }, [imgPath, refreshKey]);

  return { caption, isLoaded, setCaption };
}
