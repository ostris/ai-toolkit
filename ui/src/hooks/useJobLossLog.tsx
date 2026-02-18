'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';

export interface LossPoint {
  step: number;
  wall_time?: number;
  value: number | null;
}

type SeriesMap = Record<string, LossPoint[]>;

export type MetricFilter = 'loss' | 'learning_rate' | 'diff_guidance' | 'all' | 'other';

function categorizeMetric(key: string): 'loss' | 'learning_rate' | 'diff_guidance' | 'other' {
  if (key === 'learning_rate') return 'learning_rate';
  if (key === 'diff_guidance_norm') return 'diff_guidance';
  if (/loss/i.test(key)) return 'loss';
  return 'other';
}

function matchesFilter(key: string, filter: MetricFilter): boolean {
  if (filter === 'all') return true;
  const category = categorizeMetric(key);
  if (filter === 'other') return category === 'other';
  return category === filter;
}

function isLossKey(key: string) {
  // treat anything containing "loss" as a loss-series
  // (covers loss, train_loss, val_loss, loss/xyz, etc.)
  return /loss/i.test(key);
}

export default function useJobLossLog(
  jobID: string, 
  reloadInterval: null | number = null,
  metricFilter: MetricFilter = 'loss'
) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);

  // track last step per key so polling is incremental per series
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const filteredKeys = useMemo(() => {
    const filtered = (keys ?? []).filter(k => matchesFilter(k, metricFilter));
    // if keys table is empty early on, fall back to just "loss"
    if (filtered.length === 0 && metricFilter === 'loss') return ['loss'];
    return filtered.sort();
  }, [keys, metricFilter]);

  const refreshLoss = useCallback(async () => {
    if (!jobID) return;

    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const loadStatus: 'loading' | 'refreshing' = didInitialLoadRef.current ? 'refreshing' : 'loading';
    setStatus(loadStatus);

    try {
      // Step 1: get key list (we can do this by calling endpoint once; it returns keys)
      // Keep it cheap: limit=1.
      const first = await apiClient
        .get(`/api/jobs/${jobID}/loss`, { params: { key: 'loss', limit: 1 } })
        .then(res => res.data as { keys?: string[] });

      const newKeys = first.keys ?? [];
      setKeys(newKeys);

      const wantedKeys = (newKeys.filter(k => matchesFilter(k, metricFilter)).length 
        ? newKeys.filter(k => matchesFilter(k, metricFilter)) 
        : (metricFilter === 'loss' ? ['loss'] : [])
      ).sort();

      // Step 2: fetch each loss key incrementally (since_step per key if polling)
      const requests = wantedKeys.map(k => {
        const params: Record<string, any> = { key: k };

        if (reloadInterval && lastStepByKeyRef.current[k] != null) {
          params.since_step = lastStepByKeyRef.current[k];
        }

        // keep default limit from server (or set explicitly if you want)
        // params.limit = 2000;

        return apiClient
          .get(`/api/jobs/${jobID}/loss`, { params })
          .then(res => res.data as { key: string; points?: LossPoint[] });
      });

      const results = await Promise.all(requests);

      setSeries(prev => {
        const next: SeriesMap = { ...prev };

        for (const r of results) {
          const k = r.key;
          const newPoints = (r.points ?? []).filter(p => p.value !== null);

          if (!didInitialLoadRef.current) {
            // initial: replace
            next[k] = newPoints;
          } else if (newPoints.length) {
            const existing = next[k] ?? [];
            const prevLast = existing.length ? existing[existing.length - 1].step : null;
            const filtered = prevLast == null ? newPoints : newPoints.filter(p => p.step > prevLast);
            next[k] = filtered.length ? [...existing, ...filtered] : existing;
          } else {
            // no new points: keep existing
            next[k] = next[k] ?? [];
          }

          // update last step per key
          const finalArr = next[k] ?? [];
          lastStepByKeyRef.current[k] = finalArr.length
            ? finalArr[finalArr.length - 1].step
            : (lastStepByKeyRef.current[k] ?? null);
        }

        // remove stale keys that no longer exist (rare, but keeps UI clean)
        for (const existingKey of Object.keys(next)) {
          if (matchesFilter(existingKey, metricFilter) && !wantedKeys.includes(existingKey)) {
            delete next[existingKey];
            delete lastStepByKeyRef.current[existingKey];
          }
        }

        return next;
      });

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching loss logs:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, reloadInterval, metricFilter]);

  useEffect(() => {
    // reset when job changes
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setKeys([]);
    setStatus('idle');

    refreshLoss();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refreshLoss();
      }, reloadInterval);

      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refreshLoss]);

  return { series, keys, filteredKeys, status, refreshLoss, setSeries };
}
