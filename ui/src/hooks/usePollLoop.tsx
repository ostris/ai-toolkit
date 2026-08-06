'use client';

import { useEffect, useRef } from 'react';

/**
 * Polling loop that schedules the next run only after the current one
 * settles, so a slow server can't stack overlapping requests the way
 * setInterval does. Worst case the UI just updates late.
 *
 * - `fn` should return its promise so we know when the request finished.
 * - Errors from `fn` are caught here so a failed poll never kills the loop.
 * - `intervalMs` of null/0 runs `fn` once with no polling (matches the old
 *   reloadInterval=null behavior).
 * - Changing `deps` (or `intervalMs`) restarts the loop, running `fn`
 *   immediately.
 */
export default function usePollLoop(
  fn: () => void | Promise<unknown>,
  intervalMs: number | null | undefined,
  deps: unknown[] = [],
) {
  // always call the latest fn without restarting the loop on every render
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    let stopped = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        await fnRef.current();
      } catch (error) {
        console.error('Poll error:', error);
      }
      if (stopped || !intervalMs) return;
      timeout = setTimeout(tick, intervalMs);
    };

    tick();

    return () => {
      stopped = true;
      if (timeout) clearTimeout(timeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);
}
