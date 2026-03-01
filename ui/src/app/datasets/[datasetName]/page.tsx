'use client';

import { useEffect, useState, use, useMemo, useCallback, useRef } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft, FaTrashAlt, FaTimes, FaObjectGroup } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import DatasetImageViewer from '@/components/DatasetImageViewer';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { isAudio, isVideo } from '@/utils/basic';
import FullscreenDropOverlay from '@/components/FullscreenDropOverlay';

interface ScoringStatus {
  status: 'idle' | 'running' | 'completed' | 'cancelled' | 'error';
  scored: number;
  total: number;
  error?: string;
}

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isSelectMode, setIsSelectMode] = useState<boolean>(false);
  const [isMergeMode, setIsMergeMode] = useState<boolean>(false);
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());
  const [scoringStatus, setScoringStatus] = useState<ScoringStatus | null>(null);
  const [scoreRefreshKey, setScoreRefreshKey] = useState<number>(0);
  const scoringPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const removeImageFromList = useCallback((imgPath: string) => {
    setImgList(prev => prev.filter(x => x.img_path !== imgPath));
  }, []);

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
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };
  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName]);

  const handleLongPress = useCallback((imgPath: string) => {
    setIsSelectMode(true);
    setSelectedImages(new Set([imgPath]));
  }, []);

  const handleSelect = useCallback((imgPath: string) => {
    setSelectedImages(prev => {
      const next = new Set(prev);
      if (next.has(imgPath)) {
        next.delete(imgPath);
      } else {
        next.add(imgPath);
      }
      return next;
    });
  }, []);

  const handleCancelSelect = useCallback(() => {
    setIsSelectMode(false);
    setIsMergeMode(false);
    setSelectedImages(new Set());
  }, []);

  useEffect(() => {
    if (isSelectMode && selectedImages.size === 0) {
      setIsSelectMode(false);
      setIsMergeMode(false);
    }
  }, [isSelectMode, selectedImages.size]);

  const handleBulkDelete = useCallback(async () => {
    const paths = Array.from(selectedImages);
    await Promise.all(
      paths.map(imgPath =>
        apiClient
          .post('/api/img/trash', { imgPath })
          .then(() => removeImageFromList(imgPath))
          .catch(error => console.error('Error moving image to trash:', error)),
      ),
    );
    setIsSelectMode(false);
    setSelectedImages(new Set());
  }, [selectedImages, removeImageFromList]);

  const handleMergeStart = useCallback((imgPath: string) => {
    setIsSelectMode(true);
    setIsMergeMode(true);
    setSelectedImages(new Set([imgPath]));
  }, []);

  const handleMergeClips = useCallback(async () => {
    const videoPaths = Array.from(selectedImages).filter(p => isVideo(p)).sort();
    if (videoPaths.length < 2) return;
    try {
      await apiClient.post('/api/video/merge', { videoPaths });
      refreshImageList(datasetName);
    } catch (error) {
      console.error('Error merging videos:', error);
    }
    setIsSelectMode(false);
    setIsMergeMode(false);
    setSelectedImages(new Set());
  }, [selectedImages, datasetName]);

  const stopScoringPoll = useCallback(() => {
    if (scoringPollRef.current) {
      clearInterval(scoringPollRef.current);
      scoringPollRef.current = null;
    }
  }, []);

  const startScoringPoll = useCallback(() => {
    stopScoringPoll();
    scoringPollRef.current = setInterval(async () => {
      try {
        const res = await apiClient.get(`/api/datasets/scoreImages?datasetName=${encodeURIComponent(datasetName)}`);
        const data: ScoringStatus = res.data;
        setScoringStatus(data);
        if (data.status !== 'running') {
          stopScoringPoll();
          if (data.status === 'completed') {
            setScoreRefreshKey(k => k + 1);
          }
        }
      } catch (error) {
        console.error('Error polling scoring status:', error);
        stopScoringPoll();
      }
    }, 1000);
  }, [datasetName, stopScoringPoll]);

  useEffect(() => {
    return () => stopScoringPoll();
  }, [stopScoringPoll]);

  const handleScoreImages = useCallback(async () => {
    try {
      const res = await apiClient.post('/api/datasets/scoreImages', { datasetName });
      const data: ScoringStatus = res.data;
      setScoringStatus(data);
      if (data.status === 'running') {
        startScoringPoll();
      }
    } catch (error: any) {
      console.error('Error starting scoring:', error);
    }
  }, [datasetName, startScoringPoll]);

  const handleCancelScoring = useCallback(async () => {
    try {
      await apiClient.delete(`/api/datasets/scoreImages?datasetName=${encodeURIComponent(datasetName)}`);
      setScoringStatus(null);
      stopScoringPoll();
    } catch (error) {
      console.error('Error cancelling scoring:', error);
    }
  }, [datasetName, stopScoringPoll]);

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
        {isSelectMode ? (
          <>
            <div>
              <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={handleCancelSelect}>
                <FaTimes />
              </Button>
            </div>
            <div>
              <h1 className="text-lg">{selectedImages.size} {isMergeMode ? 'clip' : 'image'}{selectedImages.size !== 1 ? 's' : ''} selected</h1>
            </div>
            <div className="flex-1"></div>
            <div>
              {isMergeMode ? (
                <Button
                  className="text-gray-200 bg-blue-700 px-3 py-1 rounded-md flex items-center gap-2 disabled:opacity-50"
                  onClick={handleMergeClips}
                  disabled={Array.from(selectedImages).filter(p => isVideo(p)).length < 2}
                >
                  <FaObjectGroup />
                  Merge Clips
                </Button>
              ) : (
                <Button
                  className="text-gray-200 bg-red-700 px-3 py-1 rounded-md flex items-center gap-2 disabled:opacity-50"
                  onClick={handleBulkDelete}
                  disabled={selectedImages.size === 0}
                >
                  <FaTrashAlt />
                  Delete Selected
                </Button>
              )}
            </div>
          </>
        ) : (
          <>
            <div>
              <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
                <FaChevronLeft />
              </Button>
            </div>
            <div>
              <h1 className="text-lg">Dataset: {datasetName}, Images: {imgList.length}</h1>
            </div>
            <div className="flex-1"></div>
            <div className="flex gap-2">
              {scoringStatus?.status === 'running' ? (
                <Button
                  className="text-gray-200 bg-red-700 px-3 py-1 rounded-md"
                  onClick={handleCancelScoring}
                >
                  Cancel Scoring
                </Button>
              ) : (
                <Button
                  className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
                  onClick={handleScoreImages}
                >
                  Score Images
                </Button>
              )}
              <Button
                className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
                onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
              >
                Add Images
              </Button>
            </div>
          </>
        )}
      </TopBar>
      <MainContent>
        {scoringStatus?.status === 'running' && (
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Scoring images...</span>
              <span>{scoringStatus.scored} / {scoringStatus.total}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: scoringStatus.total > 0 ? `${(scoringStatus.scored / scoringStatus.total) * 100}%` : '0%' }}
              />
            </div>
          </div>
        )}
        {isSelectMode && (
          <p className="text-xs text-gray-400 mb-3">
            {isMergeMode
              ? 'Click video clips to select them for merging. Press Cancel to exit.'
              : 'Click images to select or deselect. Press Cancel to exit select mode.'}
          </p>
        )}
        {PageInfoContent}
        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {imgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                imageUrl={img.img_path}
                currentDataset={datasetName}
                onDelete={() => removeImageFromList(img.img_path)}
                onSplit={() => refreshImageList(datasetName)}
                onMerge={() => handleMergeStart(img.img_path)}
                onEnlarge={() => setSelectedImage(img.img_path)}
                isSelectMode={isSelectMode}
                selected={selectedImages.has(img.img_path)}
                onLongPress={() => handleLongPress(img.img_path)}
                onSelect={() => handleSelect(img.img_path)}
                scoreRefreshKey={scoreRefreshKey}
              />
            ))}
          </div>
        )}
      </MainContent>
      <AddImagesModal />
      <DatasetImageViewer
        imgPath={selectedImage}
        images={imgList.map(img => img.img_path).filter(path => !isAudio(path))}
        onChange={setSelectedImage}
      />
      <FullscreenDropOverlay
        datasetName={datasetName}
        onComplete={() => refreshImageList(datasetName)}
      />
    </>
  );
}
