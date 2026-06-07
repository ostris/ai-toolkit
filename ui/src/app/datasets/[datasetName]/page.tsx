'use client';

import { useEffect, useState, use, useMemo, useCallback } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import { VirtuosoGrid } from 'react-virtuoso';
import DatasetImageCard from '@/components/DatasetImageCard';
import DatasetImageViewer from '@/components/DatasetImageViewer';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import useSettings from '@/hooks/useSettings';
import { pathJoin } from '@/utils/basic';
import AutoCaptionButton from '@/components/AutoCaptionButton';
import { CreatableSelectInput } from '@/components/formInputs';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const [isAutoCaptioning, setIsAutoCaptioning] = useState(false);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const { settings, isSettingsLoaded } = useSettings();
  const [selectedImgPath, setSelectedImgPath] = useState<string | null>(null);
  const [captionExt, setCaptionExt] = useState<string>('txt');
  const [captionRefreshKeys, setCaptionRefreshKeys] = useState<Record<string, number>>({});
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        // Server already sorts; avoid the client-side sort that's expensive on large lists.
        setImgList(data.images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };
  useOpenImagesModalOnDrag(datasetName, () => refreshImageList(datasetName));

  const imgPaths = useMemo(() => imgList.map(img => img.img_path), [imgList]);

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
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
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-600/20';
      textColor = 'text-red-100';
      iconColor = 'text-red-400';
    }
    if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
      showIt = true;
      bgColor = 'bg-gray-800/50';
      textColor = 'text-gray-100';
      iconColor = 'text-gray-400';
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
        <div className="flex-shrink-0">
          <Button className="text-gray-500 dark:text-gray-300 px-2 sm:px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0 flex-shrink">
          <h1 className="text-base sm:text-lg truncate">
            <span className="hidden sm:inline">Dataset: </span>
            {datasetName}
          </h1>
        </div>
        <div className="flex-1"></div>
        <div className="flex-shrink-0 flex items-center gap-1 sm:gap-2">
          <div className="flex items-center gap-1">
            <label className="text-xs text-gray-400 hidden sm:inline whitespace-nowrap">Caption ext</label>
            <CreatableSelectInput
              className="w-44"
              value={captionExt}
              onChange={value => setCaptionExt(value)}
              options={[
                { value: 'txt', label: 'txt' },
                { value: 'json', label: 'json' },
                { value: 'caption', label: 'caption' },
              ]}
            />
          </div>
          <AutoCaptionButton
            datasetPath={`${pathJoin(settings.DATASETS_FOLDER, datasetName)}`}
            setIsAutoCaptioning={setIsAutoCaptioning}
            captionExt={captionExt}
          />
          <Button
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md text-sm sm:text-base whitespace-nowrap"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            <span className="sm:hidden">+ Add</span>
            <span className="hidden sm:inline">Add Images</span>
          </Button>
        </div>
      </TopBar>
      <MainContent ref={scrollParentCallback}>
        {PageInfoContent}
        {status === 'success' && imgList.length > 0 && scrollParent && (
          <VirtuosoGrid
            totalCount={imgList.length}
            customScrollParent={scrollParent}
            overscan={400}
            listClassName="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
            itemContent={index => {
              const img = imgList[index];
              if (!img) return null;
              return (
                <DatasetImageCard
                  alt="image"
                  isAutoCaptioning={isAutoCaptioning}
                  imageUrl={img.img_path}
                  onDelete={() => refreshImageList(datasetName)}
                  onImageClick={() => setSelectedImgPath(img.img_path)}
                  captionRefreshKey={captionRefreshKeys[img.img_path] || 0}
                  observerRoot={scrollParent}
                  captionExt={captionExt}
                />
              );
            }}
            computeItemKey={index => imgList[index]?.img_path ?? index}
          />
        )}
      </MainContent>
      <AddImagesModal />
      <DatasetImageViewer
        imgPath={selectedImgPath}
        imageList={imgPaths}
        onChange={setSelectedImgPath}
        refreshImages={() => refreshImageList(datasetName)}
        onCaptionSaved={path => setCaptionRefreshKeys(prev => ({ ...prev, [path]: (prev[path] || 0) + 1 }))}
        captionExt={captionExt}
      />
    </>
  );
}
