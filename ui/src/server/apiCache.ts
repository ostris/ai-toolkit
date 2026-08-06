type CacheEntry = {
  promise: Promise<unknown>;
  timestamp: number;
};

const cache = new Map<string, CacheEntry>();

const DEFAULT_STALE_TIME_MS = 5000;

/**
 * Universal cache for slow API work. Results are cached per key + params and
 * reused until they go stale. Concurrent callers while a fetch is in flight
 * share the same promise instead of triggering duplicate work.
 *
 * @param key        Unique name for this cached operation (e.g. 'cpu-info')
 * @param fetcher    Function that produces a fresh value
 * @param staleTimeMs How long a cached value stays fresh (default 5000ms)
 * @param params     Optional params; different params get separate cache entries
 */
export async function cached<T>(
  key: string,
  fetcher: () => Promise<T>,
  staleTimeMs: number = DEFAULT_STALE_TIME_MS,
  params: unknown = null,
): Promise<T> {
  const cacheKey = params === null ? key : `${key}:${JSON.stringify(params)}`;
  const now = Date.now();

  const entry = cache.get(cacheKey);
  if (entry && now - entry.timestamp < staleTimeMs) {
    return entry.promise as Promise<T>;
  }

  const promise = fetcher();
  cache.set(cacheKey, { promise, timestamp: now });

  // Drop failed fetches so the next call retries instead of caching the error
  promise.catch(() => {
    if (cache.get(cacheKey)?.promise === promise) {
      cache.delete(cacheKey);
    }
  });

  return promise;
}

/** Remove a cached entry (all param variants if no params given). */
export function invalidateCache(key: string, params: unknown = null): void {
  if (params !== null) {
    cache.delete(`${key}:${JSON.stringify(params)}`);
    return;
  }
  cache.delete(key);
  for (const cacheKey of cache.keys()) {
    if (cacheKey.startsWith(`${key}:`)) {
      cache.delete(cacheKey);
    }
  }
}
