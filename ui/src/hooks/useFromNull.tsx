import { useEffect, useRef } from 'react';

export function useFromNull(effect: () => void | (() => void), deps: Array<any | null | undefined>) {
  const prevDepsRef = useRef<(any | null | undefined)[]>([]);

  useEffect(() => {
    const shouldRun = deps.some((dep, i) => prevDepsRef.current[i] == null && dep != null);

    if (shouldRun) {
      const cleanup = effect();
      prevDepsRef.current = deps;
      return cleanup;
    }

    prevDepsRef.current = deps;
  }, deps);
}
