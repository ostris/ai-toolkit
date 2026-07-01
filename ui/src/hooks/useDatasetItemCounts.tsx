'use client';

import { useEffect, useMemo, useState } from 'react';
import { apiClient } from '@/utils/api';

type DatasetItemCounts = Record<string, number | null>;

export default function useDatasetItemCounts(datasetPaths: string[]) {
  const [counts, setCounts] = useState<DatasetItemCounts>({});

  const datasetPathKey = useMemo(() => {
    return Array.from(new Set(datasetPaths.filter(datasetPath => datasetPath.trim() !== ''))).join('\0');
  }, [datasetPaths]);

  useEffect(() => {
    const uniqueDatasetPaths = datasetPathKey === '' ? [] : datasetPathKey.split('\0');

    if (uniqueDatasetPaths.length === 0) {
      setCounts({});
      return;
    }

    let isMounted = true;

    apiClient
      .post('/api/datasets/countItems', { datasetPaths: uniqueDatasetPaths })
      .then(res => {
        if (isMounted) {
          setCounts(res.data.counts ?? {});
        }
      })
      .catch(error => {
        if (isMounted) {
          console.error('Error counting dataset items:', error);
          setCounts({});
        }
      });

    return () => {
      isMounted = false;
    };
  }, [datasetPathKey]);

  return counts;
}
