'use client';

import { useEffect, useState, use, useMemo } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import FullscreenDropOverlay from '@/components/FullscreenDropOverlay';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [renameText, setRenameText] = useState(datasetName);
  const [isRenaming, setIsRenaming] = useState(false);

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    console.log('Fetching images for dataset:', dbName);
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        console.log('Images:', data.images);
        // sort
        data.images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
        setImgList(data.images);
        setStatus('success');
      })
      .catch((error: any) => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };

  const handleRename = async () => {
    if (!renameText.trim()) {
      alert('Please enter a valid name');
      return;
    }

    setIsRenaming(true);
    try {
      const response = await apiClient.post('/api/datasets/rename', {
        datasetName: datasetName,
        newBaseName: renameText.trim()
      });

      if (response.data.success) {
        console.log('Rename successful:', response.data);
        // Refresh the image list to show the renamed files
        refreshImageList(datasetName);
        alert(`Successfully renamed ${response.data.totalRenamed} files`);
      } else {
        alert('Rename failed: ' + (response.data.error || 'Unknown error'));
      }
    } catch (error: any) {
      console.error('Rename error:', error);
      alert('Failed to rename files: ' + (error.response?.data?.error || error.message));
    } finally {
      setIsRenaming(false);
    }
  };
  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
      setRenameText(datasetName);
    }
  }, [datasetName]);

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, imgList.length]);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Dataset: {datasetName}</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            Add Images
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {PageInfoContent}

        {/* Renaming Section - Only visible when files are successfully uploaded */}
        {status === 'success' && imgList.length > 0 && (
          <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold mb-3 text-gray-900 dark:text-gray-100">Renaming</h2>
            <div className="flex gap-3 items-center">
              <input
                type="text"
                value={renameText}
                onChange={(e) => setRenameText(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter base name for files"
                disabled={isRenaming}
              />
              <Button
                onClick={handleRename}
                disabled={isRenaming || !renameText.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
              >
                {isRenaming ? 'Renaming...' : 'Rename'}
              </Button>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
              All media files (images and videos) will be renamed to "{renameText}001.ext", "{renameText}002.ext", etc. (keeping original extensions)
            </p>
          </div>
        )}

        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {imgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                imageUrl={img.img_path}
                onDelete={() => refreshImageList(datasetName)}
              />
            ))}
          </div>
        )}
      </MainContent>
      <AddImagesModal />
      <FullscreenDropOverlay
        datasetName={datasetName}
        onComplete={() => refreshImageList(datasetName)}
      />
    </>
  );
}
