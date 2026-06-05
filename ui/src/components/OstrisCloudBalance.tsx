'use client';

import { useEffect, useState } from 'react';
import { Wallet } from 'lucide-react';
import { apiClient } from '@/utils/api';

interface BalanceResponse {
  enabled: boolean;
  appUrl?: string;
  balance?: {
    units: string;
    usd: number;
  } | null;
  error?: string;
}

export default function OstrisCloudBalance() {
  const [data, setData] = useState<BalanceResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchBalance = async () => {
      try {
        const res = await apiClient.get<BalanceResponse>('/api/ostris_cloud');
        if (!cancelled) setData(res.data);
      } catch {
        // ignore — keep last known value
      }
    };

    fetchBalance();
    const interval = setInterval(fetchBalance, 60_000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (!data || !data.enabled || !data.appUrl) return null;

  const usd = data.balance?.usd;
  const hasAmount = typeof usd === 'number';
  const display = hasAmount
    ? usd.toLocaleString(undefined, {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
    : null;

  return (
    <a
      href={`${data.appUrl}/billing`}
      target="_blank"
      rel="noreferrer"
      className="block px-4 py-2 hover:bg-gray-800 transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Wallet className="w-4 h-4 text-blue-400" />
          <span className="text-[11px] uppercase tracking-wide text-gray-400">
            <span className="font-extrabold">Ostris</span> <span className="font-thin">Cloud</span>
          </span>
        </div>
        {display ? (
          <span className="text-xs font-medium text-gray-100 tabular-nums">{display}</span>
        ) : (
          <span className="inline-block h-3 w-10 rounded bg-gray-700 animate-pulse" aria-label="Loading balance" />
        )}
      </div>
    </a>
  );
}
