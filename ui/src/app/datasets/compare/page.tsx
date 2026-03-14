'use client';

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { FaChevronLeft } from 'react-icons/fa';
import { LuLoader } from 'react-icons/lu';
import { TopBar, MainContent } from '@/components/layout';
import CompareView from '@/components/CompareView';
import { apiClient } from '@/utils/api';

interface ImagePair {
  left: string;
  right: string;
  filename: string;
}

function getBasename(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1];
}

function DatasetCompareContent() {
  const searchParams = useSearchParams();
  const left = searchParams.get('left') || '';
  const right = searchParams.get('right') || '';

  const [pairs, setPairs] = useState<ImagePair[]>([]);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!left || !right) {
      setError('Both left and right dataset names are required.');
      setStatus('error');
      return;
    }

    setStatus('loading');

    Promise.all([
      apiClient.post('/api/datasets/listImages', { datasetName: left }).then(res => res.data.images as { img_path: string }[]),
      apiClient.post('/api/datasets/listImages', { datasetName: right }).then(res => res.data.images as { img_path: string }[]),
    ])
      .then(([leftImages, rightImages]) => {
        // Build a map of basename -> full path for each side
        const leftMap = new Map<string, string>();
        leftImages.forEach(img => leftMap.set(getBasename(img.img_path), img.img_path));

        const rightMap = new Map<string, string>();
        rightImages.forEach(img => rightMap.set(getBasename(img.img_path), img.img_path));

        // Pair by matching basenames, sorted
        const allNames = Array.from(leftMap.keys()).sort();
        const imagePairs: ImagePair[] = allNames
          .filter(name => rightMap.has(name))
          .map(name => ({
            left: leftMap.get(name)!,
            right: rightMap.get(name)!,
            filename: name,
          }));

        setPairs(imagePairs);
        setStatus('success');
      })
      .catch(err => {
        setError(err?.response?.data?.error || 'Failed to load images.');
        setStatus('error');
      });
  }, [left, right]);

  if (status === 'loading') {
    return (
      <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
        <LuLoader className="animate-spin w-8 h-8 mb-4 text-gray-400" />
        <h3 className="text-lg font-semibold mb-2">Loading Comparison</h3>
        <p className="text-sm opacity-75">Fetching images from both datasets...</p>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-red-950/20 text-red-100 mx-auto max-w-md text-center">
        <h3 className="text-lg font-semibold mb-2">Error</h3>
        <p className="text-sm opacity-75">{error}</p>
      </div>
    );
  }

  return (
    <CompareView
      pairs={pairs}
      leftLabel={left}
      rightLabel={right}
      mode="dataset"
      leftDataset={left}
      rightDataset={right}
    />
  );
}

export default function DatasetComparePage() {
  return (
    <>
      <TopBar>
        <Link href="/datasets" className="text-gray-300 hover:text-white px-2">
          <FaChevronLeft />
        </Link>
        <div className="flex-1 min-w-0 px-2">
          <h1 className="text-lg font-semibold text-gray-100">Compare Datasets</h1>
        </div>
      </TopBar>

      <MainContent>
        <Suspense
          fallback={
            <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
              <LuLoader className="animate-spin w-8 h-8 mb-4 text-gray-400" />
              <h3 className="text-lg font-semibold mb-2">Loading...</h3>
            </div>
          }
        >
          <DatasetCompareContent />
        </Suspense>
      </MainContent>
    </>
  );
}
