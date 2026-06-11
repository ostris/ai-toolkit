'use client';
import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

// Module-level batcher: many cards mount at once when the virtualized grid scrolls;
// instead of N HTTP requests, queue paths and flush them as a single batch.
// Entries are keyed by extension + path so different caption extensions don't
// collide in the cache or the pending batch.
type Resolver = { resolve: (caption: string) => void; reject: (err: unknown) => void };
type Pending = { path: string; ext: string; resolvers: Resolver[] };
const pending = new Map<string, Pending>();
const cache = new Map<string, string>();
let flushTimer: ReturnType<typeof setTimeout> | null = null;
const FLUSH_DELAY_MS = 30;
const MAX_BATCH = 200;

function normExt(ext: string | undefined): string {
  return (ext || 'txt').replace(/^\.+/, '').trim() || 'txt';
}

function keyFor(path: string, ext: string): string {
  return `${ext}\n${path}`;
}

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(flush, FLUSH_DELAY_MS);
}

async function flush() {
  flushTimer = null;
  if (pending.size === 0) return;

  // Drain up to MAX_BATCH entries; if more arrived, reschedule.
  const keys: string[] = [];
  for (const key of pending.keys()) {
    keys.push(key);
    if (keys.length >= MAX_BATCH) break;
  }
  const drained = keys.map(k => pending.get(k)!);
  for (const k of keys) pending.delete(k);

  // Group by extension; each extension is a separate batch request.
  const byExt = new Map<string, Pending[]>();
  for (const entry of drained) {
    const group = byExt.get(entry.ext);
    if (group) group.push(entry);
    else byExt.set(entry.ext, [entry]);
  }

  await Promise.all(
    Array.from(byExt.entries()).map(async ([ext, entries]) => {
      const paths = entries.map(e => e.path);
      try {
        const res = await apiClient.post('/api/caption/getBatch', { imgPaths: paths, ext });
        const captions: Record<string, string> = res.data?.captions ?? {};
        for (const { path, ext: e, resolvers } of entries) {
          const value = captions[path] ?? '';
          cache.set(keyFor(path, e), value);
          for (const r of resolvers) r.resolve(value);
        }
      } catch (err) {
        for (const { resolvers } of entries) {
          for (const r of resolvers) r.reject(err);
        }
      }
    }),
  );

  if (pending.size > 0) scheduleFlush();
}

function requestCaption(path: string, ext: string, signal?: AbortSignal): Promise<string> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const key = keyFor(path, ext);
    const resolver: Resolver = { resolve, reject };
    const entry = pending.get(key);
    if (entry) {
      entry.resolvers.push(resolver);
    } else {
      pending.set(key, { path, ext, resolvers: [resolver] });
    }
    if (signal) {
      const onAbort = () => {
        // Remove this resolver from the pending batch. If no other card is
        // still waiting on the same key, drop the entry entirely so the next
        // batch doesn't include it.
        const e = pending.get(key);
        if (e) {
          const idx = e.resolvers.indexOf(resolver);
          if (idx >= 0) e.resolvers.splice(idx, 1);
          if (e.resolvers.length === 0) pending.delete(key);
        }
        reject(new DOMException('Aborted', 'AbortError'));
      };
      signal.addEventListener('abort', onAbort, { once: true });
    }
    scheduleFlush();
  });
}

export function invalidateCaption(path: string, ext?: string) {
  cache.delete(keyFor(path, normExt(ext)));
}

export function setCachedCaption(path: string, caption: string, ext?: string) {
  cache.set(keyFor(path, normExt(ext)), caption);
}

// Fetches caption for a path, using the module-level batcher + cache.
// `refreshKey` busts the cache (e.g. after external edits or auto-captioning poll).
export default function useCaptionBatch(imgPath: string | null, refreshKey: number = 0, ext: string = 'txt') {
  const captionExt = normExt(ext);
  const [caption, setCaption] = useState<string>(() => (imgPath ? (cache.get(keyFor(imgPath, captionExt)) ?? '') : ''));
  const [isLoaded, setIsLoaded] = useState<boolean>(() => Boolean(imgPath && cache.has(keyFor(imgPath, captionExt))));
  const lastPathRef = useRef<string | null>(null);

  useEffect(() => {
    if (!imgPath) {
      setCaption('');
      setIsLoaded(false);
      return;
    }

    if (refreshKey > 0) invalidateCaption(imgPath, captionExt);

    const cached = cache.get(keyFor(imgPath, captionExt));
    if (cached !== undefined) {
      setCaption(cached);
      setIsLoaded(true);
      lastPathRef.current = imgPath;
      return;
    }

    let cancelled = false;
    const controller = new AbortController();
    lastPathRef.current = imgPath;
    setIsLoaded(false);
    requestCaption(imgPath, captionExt, controller.signal)
      .then(value => {
        if (cancelled || lastPathRef.current !== imgPath) return;
        setCaption(value);
        setIsLoaded(true);
      })
      .catch(err => {
        if (err?.name === 'AbortError' || cancelled) return;
        console.error('Error fetching caption:', err);
        setIsLoaded(true);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [imgPath, refreshKey, captionExt]);

  return { caption, isLoaded, setCaption };
}
