'use client';

import { useEffect, useState, use, useCallback, useRef } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft, FaTimes, FaPlay, FaPause, FaRandom, FaClock } from 'react-icons/fa';
import { FaUndoAlt, FaRedoAlt, FaComment, FaArrowsAlt, FaExpand } from 'react-icons/fa';
import Link from 'next/link';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { isVideo, isAudio, isImage } from '@/utils/basic';
import DatasetImageViewer from '@/components/DatasetImageViewer';
import GalleryCopyModal from '@/components/GalleryCopyModal';
import { Modal } from '@/components/Modal';

interface GalleryFolder {
  id: number;
  path: string;
  created_at: string;
}

interface ImageItem {
  img_path: string;
}

interface GalleryCardProps {
  imageUrl: string;
  captionRefreshKey: number;
  onEnlarge: () => void;
}

function GalleryImageCard({ imageUrl, captionRefreshKey, onEnlarge }: GalleryCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [inViewport, setInViewport] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [imageKey, setImageKey] = useState(Date.now());
  const [caption, setCaption] = useState('');
  const [savedCaption, setSavedCaption] = useState('');
  const [isCaptionLoaded, setIsCaptionLoaded] = useState(false);
  const [isCopyModalOpen, setIsCopyModalOpen] = useState(false);
  const isGettingCaption = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        setInViewport(entries[0].isIntersecting);
      },
      { threshold: 0.1 },
    );
    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  const fetchCaption = useCallback(
    (force = false) => {
      if (isGettingCaption.current || (!force && isCaptionLoaded)) return;
      isGettingCaption.current = true;
      apiClient
        .post('/api/gallery/caption', { imgPath: imageUrl })
        .then(res => {
          const data = res.data ? `${res.data}` : '';
          setCaption(data);
          setSavedCaption(data);
          setIsCaptionLoaded(true);
        })
        .catch(() => {
          setIsCaptionLoaded(true);
        })
        .finally(() => {
          isGettingCaption.current = false;
        });
    },
    [imageUrl, isCaptionLoaded],
  );

  useEffect(() => {
    if (inViewport) fetchCaption();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inViewport]);

  useEffect(() => {
    if (captionRefreshKey && inViewport && savedCaption === '') {
      isGettingCaption.current = false;
      fetchCaption(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captionRefreshKey]);

  const saveCaption = () => {
    const trimmed = caption.trim();
    if (trimmed === savedCaption) return;
    apiClient
      .post('/api/gallery/captionSave', { imgPath: imageUrl, caption: trimmed })
      .then(() => setSavedCaption(trimmed))
      .catch(err => console.error('Error saving caption:', err));
  };

  const rotateImage = (direction: 'left' | 'right') => {
    apiClient
      .post('/api/gallery/rotate', { imgPath: imageUrl, direction })
      .then(() => {
        setLoaded(false);
        setImageKey(prev => prev + 1);
      })
      .catch(err => console.error('Error rotating image:', err));
  };

  const isCaptionCurrent = caption.trim() === savedCaption;
  const isItImage = isImage(imageUrl);

  return (
    <div className="flex flex-col">
      <div ref={cardRef} className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {inViewport && (
            <>
              {isItImage && (
                <img
                  key={imageKey}
                  src={`/api/gallery/img/${encodeURIComponent(imageUrl)}?v=${imageKey}`}
                  alt={imageUrl}
                  onLoad={() => setLoaded(true)}
                  onClick={onEnlarge}
                  className={`w-full h-full object-contain transition-opacity duration-300 cursor-pointer ${loaded ? 'opacity-100' : 'opacity-0'}`}
                />
              )}
              {isVideo(imageUrl) && (
                <video
                  src={`/api/gallery/img/${encodeURIComponent(imageUrl)}`}
                  className="w-full h-full object-contain"
                  autoPlay={false}
                  loop
                  controls
                />
              )}
            </>
          )}
          <div className="absolute top-1 right-1 flex space-x-2 z-10">
            {isItImage && (
              <>
                <button className="bg-gray-800 rounded-full p-2" onClick={onEnlarge} aria-label="Enlarge image">
                  <FaExpand />
                </button>
                <button className="bg-gray-800 rounded-full p-2" onClick={() => rotateImage('left')} aria-label="Rotate left">
                  <FaUndoAlt />
                </button>
                <button className="bg-gray-800 rounded-full p-2" onClick={() => rotateImage('right')} aria-label="Rotate right">
                  <FaRedoAlt />
                </button>
              </>
            )}
            <button className="bg-gray-800 rounded-full p-2" onClick={() => setIsCopyModalOpen(true)} aria-label="Copy to dataset">
              <FaArrowsAlt />
            </button>
          </div>
          {inViewport && (
            <div className="text-xs text-gray-100 bg-gray-950 mt-1 absolute bottom-0 left-0 p-1 opacity-25 hover:opacity-90 transition-opacity duration-300 w-full">
              {imageUrl}
            </div>
          )}
        </div>
      </div>
      <div className="relative w-full" style={{ height: '75px' }}>
        <div
          className={`absolute inset-x-0 top-0 p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px] hover:h-[150px] transition-[height] duration-300 ease-in-out z-20 overflow-hidden border-2 ${
            isCaptionCurrent ? 'border-transparent' : 'border-blue-500'
          }`}
        >
          {inViewport && isCaptionLoaded ? (
            <form
              className="h-full"
              onSubmit={e => {
                e.preventDefault();
                saveCaption();
              }}
              onBlur={saveCaption}
            >
              <textarea
                className="w-full h-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none"
                value={caption}
                onChange={e => setCaption(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    saveCaption();
                  }
                }}
              />
            </form>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400">
              {isCaptionLoaded ? 'Scroll into view to edit' : 'Loading caption...'}
            </div>
          )}
        </div>
      </div>
      <GalleryCopyModal
        isOpen={isCopyModalOpen}
        onClose={() => setIsCopyModalOpen(false)}
        imageUrl={imageUrl}
        onComplete={() => setIsCopyModalOpen(false)}
      />
    </div>
  );
}

export default function GalleryFolderPage({ params }: { params: { folderId: string } }) {
  const usableParams = use(params as any) as { folderId: string };
  const folderId = parseInt(usableParams.folderId, 10);

  const [folder, setFolder] = useState<GalleryFolder | null>(null);
  const [imgList, setImgList] = useState<ImageItem[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [captionRefreshKey, setCaptionRefreshKey] = useState(0);

  // Slideshow state
  const [isSlideshowOpen, setIsSlideshowOpen] = useState(false);
  const [slideshowDuration, setSlideshowDuration] = useState(3);
  const [slideshowRandom, setSlideshowRandom] = useState(false);
  const [isSlideshow, setIsSlideshow] = useState(false);
  const [slideshowIndex, setSlideshowIndex] = useState(0);
  const slideshowTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [slideshowOrder, setSlideshowOrder] = useState<number[]>([]);

  const imageOnlyList = imgList.filter(img => isImage(img.img_path));

  // Load folder info and images
  useEffect(() => {
    if (!folderId) return;
    // Load folder details
    apiClient
      .get('/api/gallery/list')
      .then(res => res.data)
      .then((data: GalleryFolder[]) => {
        const f = data.find(x => x.id === folderId);
        if (f) setFolder(f);
      })
      .catch(() => {});
  }, [folderId]);

  useEffect(() => {
    if (!folder) return;
    setStatus('loading');
    apiClient
      .get(`/api/gallery/images?folderPath=${encodeURIComponent(folder.path)}`)
      .then(res => res.data)
      .then((data: { images: ImageItem[] }) => {
        const sorted = [...data.images].sort((a, b) => a.img_path.localeCompare(b.img_path));
        setImgList(sorted);
        setStatus('success');
      })
      .catch(() => setStatus('error'));
  }, [folder]);

  // Slideshow logic
  const stopSlideshow = useCallback(() => {
    if (slideshowTimer.current) {
      clearInterval(slideshowTimer.current);
      slideshowTimer.current = null;
    }
    setIsSlideshow(false);
  }, []);

  const startSlideshow = useCallback(() => {
    if (imageOnlyList.length === 0) return;
    const order = imageOnlyList.map((_, i) => i);
    if (slideshowRandom) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    setSlideshowOrder(order);
    setSlideshowIndex(0);
    setSelectedImage(imageOnlyList[order[0]].img_path);
    setIsSlideshow(true);
  }, [imageOnlyList, slideshowRandom]);

  useEffect(() => {
    if (!isSlideshow) return;
    if (slideshowTimer.current) clearInterval(slideshowTimer.current);
    slideshowTimer.current = setInterval(() => {
      setSlideshowIndex(prev => {
        const next = prev + 1;
        if (next >= slideshowOrder.length) {
          stopSlideshow();
          return prev;
        }
        setSelectedImage(imageOnlyList[slideshowOrder[next]].img_path);
        return next;
      });
    }, slideshowDuration * 1000);
    return () => {
      if (slideshowTimer.current) clearInterval(slideshowTimer.current);
    };
  }, [isSlideshow, slideshowDuration, slideshowOrder, imageOnlyList, stopSlideshow]);

  useEffect(() => {
    if (!selectedImage && isSlideshow) {
      stopSlideshow();
    }
  }, [selectedImage, isSlideshow, stopSlideshow]);

  const handleImageChange = useCallback(
    (nextPath: string | null) => {
      setSelectedImage(nextPath);
      if (!nextPath && isSlideshow) stopSlideshow();
    },
    [isSlideshow, stopSlideshow],
  );

  const imagePaths = imgList.map(x => x.img_path);

  let pageContent: React.ReactNode = null;
  if (status === 'loading') {
    pageContent = (
      <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
        <LuLoader className="animate-spin w-8 h-8 mb-4 text-gray-400" />
        <h3 className="text-lg font-semibold mb-2">Loading Images</h3>
        <p className="text-sm opacity-75">Please wait...</p>
      </div>
    );
  } else if (status === 'error') {
    pageContent = (
      <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-red-950/20 text-red-100 mx-auto max-w-md text-center">
        <LuBan className="w-8 h-8 mb-4 text-red-400" />
        <h3 className="text-lg font-semibold mb-2">Error Loading Images</h3>
      </div>
    );
  } else if (status === 'success' && imgList.length === 0) {
    pageContent = (
      <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
        <LuImageOff className="w-8 h-8 mb-4 text-gray-400" />
        <h3 className="text-lg font-semibold mb-2">No Images Found</h3>
        <p className="text-sm opacity-75">This folder contains no supported media files.</p>
      </div>
    );
  }

  return (
    <>
      <TopBar>
        <Link href="/gallery" className="text-gray-300 hover:text-white px-2">
          <FaChevronLeft />
        </Link>
        <div className="flex-1 min-w-0 px-2">
          <h1 className="text-lg font-semibold text-gray-100 truncate">{folder?.path ?? 'Gallery'}</h1>
        </div>
        {imageOnlyList.length > 0 && (
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1.5 rounded-md hover:bg-slate-500 transition-colors flex items-center gap-2 text-sm"
            onClick={() => setIsSlideshowOpen(true)}
          >
            <FaPlay className="w-3 h-3" />
            Slideshow
          </Button>
        )}
      </TopBar>

      <MainContent>
        {pageContent}
        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {imgList.map(img => (
              <GalleryImageCard
                key={img.img_path}
                imageUrl={img.img_path}
                captionRefreshKey={captionRefreshKey}
                onEnlarge={() => setSelectedImage(img.img_path)}
              />
            ))}
          </div>
        )}
      </MainContent>

      <DatasetImageViewer
        imgPath={selectedImage}
        images={imagePaths}
        onChange={handleImageChange}
        apiBase="/api/gallery/img"
      />

      {/* Slideshow settings modal */}
      <Modal isOpen={isSlideshowOpen} onClose={() => setIsSlideshowOpen(false)} title="Slideshow" size="sm">
        <div className="space-y-4 text-gray-200">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
              <FaClock className="w-4 h-4" />
              Duration per slide (seconds)
            </label>
            <input
              type="number"
              min={1}
              max={60}
              value={slideshowDuration}
              onChange={e => setSlideshowDuration(Math.max(1, parseInt(e.target.value) || 1))}
              className="w-full rounded-md bg-gray-700 border border-gray-600 text-gray-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="randomOrder"
              checked={slideshowRandom}
              onChange={e => setSlideshowRandom(e.target.checked)}
              className="accent-blue-500"
            />
            <label htmlFor="randomOrder" className="text-sm text-gray-300 cursor-pointer flex items-center gap-2">
              <FaRandom className="w-4 h-4" />
              Randomise order
            </label>
          </div>
          <div className="mt-4 flex justify-end gap-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none"
              onClick={() => setIsSlideshowOpen(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none flex items-center gap-2"
              onClick={() => {
                setIsSlideshowOpen(false);
                startSlideshow();
              }}
            >
              <FaPlay className="w-3 h-3" />
              Start Slideshow
            </button>
          </div>
        </div>
      </Modal>
    </>
  );
}
