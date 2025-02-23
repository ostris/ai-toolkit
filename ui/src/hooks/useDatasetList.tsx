'use client';

import { useEffect, useState } from 'react';

export default function useDatasetList() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshDatasets = () => {
    setStatus('loading');
    fetch('/api/datasets/list')
      .then(res => res.json())
      .then(data => {
        console.log('Datasets:', data);
        // sort
        data.sort((a: string, b: string) => a.localeCompare(b));
        setDatasets(data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    refreshDatasets();
  }, []);

  return { datasets, setDatasets, status, refreshDatasets };
}
