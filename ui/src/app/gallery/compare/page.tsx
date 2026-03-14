'use client';

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { FaChevronLeft } from 'react-icons/fa';
import { LuLoader } from 'react-icons/lu';
import { TopBar, MainContent } from '@/components/layout';
import CompareView from '@/components/CompareView';
import { apiClient } from '@/utils/api';

interface GalleryFolder {
  id: number;
  path: string;
  created_at: string;
}

interface ImagePair {
  left: string;
  right: string;
  center?: string;
  filename: string;
}

function getBasename(filePath: string): string {
  const parts = filePath.replace(/\\/g, '/').split('/');
  return parts[parts.length - 1];
}

function GalleryCompareContent() {
  const searchParams = useSearchParams();
  const leftId = searchParams.get('left') || '';
  const rightId = searchParams.get('right') || '';
  const centerId = searchParams.get('center') || '';

  const [pairs, setPairs] = useState<ImagePair[]>([]);
  const [leftLabel, setLeftLabel] = useState('');
  const [rightLabel, setRightLabel] = useState('');
  const [centerLabel, setCenterLabel] = useState('');
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!leftId || !rightId) {
      setError('Both left and right folder IDs are required.');
      setStatus('error');
      return;
    }

    setStatus('loading');

    // First resolve folder IDs to paths
    apiClient
      .get('/api/gallery/list')
      .then(res => res.data as GalleryFolder[])
      .then(folders => {
        const leftFolder = folders.find(f => f.id === parseInt(leftId, 10));
        const rightFolder = folders.find(f => f.id === parseInt(rightId, 10));
        const centerFolder = centerId ? folders.find(f => f.id === parseInt(centerId, 10)) : null;

        if (!leftFolder || !rightFolder) {
          setError('One or more gallery folders not found.');
          setStatus('error');
          return;
        }
        if (centerId && !centerFolder) {
          setError('Center gallery folder not found.');
          setStatus('error');
          return;
        }

        setLeftLabel(leftFolder.path);
        setRightLabel(rightFolder.path);
        if (centerFolder) setCenterLabel(centerFolder.path);

        const fetches = [
          apiClient.get(`/api/gallery/images?folderPath=${encodeURIComponent(leftFolder.path)}`).then(res => res.data.images as { img_path: string }[]),
          apiClient.get(`/api/gallery/images?folderPath=${encodeURIComponent(rightFolder.path)}`).then(res => res.data.images as { img_path: string }[]),
          ...(centerFolder ? [apiClient.get(`/api/gallery/images?folderPath=${encodeURIComponent(centerFolder.path)}`).then(res => res.data.images as { img_path: string }[])] : []),
        ];

        return Promise.all(fetches);
      })
      .then(result => {
        if (!result) return;
        const [leftImages, rightImages, centerImages] = result;

        const leftMap = new Map<string, string>();
        leftImages.forEach(img => leftMap.set(getBasename(img.img_path), img.img_path));

        const rightMap = new Map<string, string>();
        rightImages.forEach(img => rightMap.set(getBasename(img.img_path), img.img_path));

        const centerMap = new Map<string, string>();
        if (centerImages) {
          centerImages.forEach(img => centerMap.set(getBasename(img.img_path), img.img_path));
        }

        const allNames = Array.from(leftMap.keys()).sort();
        const imagePairs: ImagePair[] = allNames
          .filter(name => rightMap.has(name) && (!centerId || centerMap.has(name)))
          .map(name => ({
            left: leftMap.get(name)!,
            right: rightMap.get(name)!,
            ...(centerId ? { center: centerMap.get(name)! } : {}),
            filename: name,
          }));

        setPairs(imagePairs);
        setStatus('success');
      })
      .catch(err => {
        setError(err?.response?.data?.error || 'Failed to load images.');
        setStatus('error');
      });
  }, [leftId, rightId, centerId]);

  if (status === 'loading') {
    return (
      <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
        <LuLoader className="animate-spin w-8 h-8 mb-4 text-gray-400" />
        <h3 className="text-lg font-semibold mb-2">Loading Comparison</h3>
        <p className="text-sm opacity-75">Fetching images from both folders...</p>
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
      leftLabel={leftLabel}
      rightLabel={rightLabel}
      centerLabel={centerLabel || undefined}
      mode="gallery"
    />
  );
}

export default function GalleryComparePage() {
  return (
    <>
      <TopBar>
        <Link href="/gallery" className="text-gray-300 hover:text-white px-2">
          <FaChevronLeft />
        </Link>
        <div className="flex-1 min-w-0 px-2">
          <h1 className="text-lg font-semibold text-gray-100">Compare Folders</h1>
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
          <GalleryCompareContent />
        </Suspense>
      </MainContent>
    </>
  );
}
