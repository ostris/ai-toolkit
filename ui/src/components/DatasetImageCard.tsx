import React, { useEffect, useState, ReactNode, KeyboardEvent, useRef } from 'react';
import { FaTrashAlt, FaPlay } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio, encodeFilePathForUrl } from '@/utils/basic';
import useCaptionBatch, { setCachedCaption } from '@/hooks/useCaptionBatch';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  isAutoCaptioning: boolean;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onImageClick?: () => void;
  captionRefreshKey?: number;
  observerRoot?: Element | null;
  rootMargin?: string;
  captionExt?: string;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  isAutoCaptioning,
  children,
  className = '',
  onDelete = () => {},
  onImageClick,
  captionRefreshKey = 0,
  observerRoot = null,
  rootMargin = '200px 0px',
  captionExt = 'txt',
}) => {
  const [loaded, setLoaded] = useState<boolean>(false);
  const [showAudioPlayer, setShowAudioPlayer] = useState(true);
  const [pollTick, setPollTick] = useState(0);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [streamVideo, setStreamVideo] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);

  // Track actual viewport visibility — Virtuoso keeps a buffer of cards mounted
  // outside the visible region, so we can't rely on mount/unmount alone.
  useEffect(() => {
    const el = cardRef.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.target === el) {
            setIsVisible(entry.isIntersecting);
          }
        }
      },
      {
        root: observerRoot ?? null,
        threshold: 0.01,
        rootMargin,
      },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [observerRoot, rootMargin]);

  // Drive image loads through fetch + AbortController so scrolling past actually
  // cancels in-flight requests. Debounced 80ms so fast scroll-throughs never
  // start a request. Both images and videos pull the 300x300 thumb (the server
  // generates it on a miss); ?thumb=1 falls through to the real file only when
  // a thumb can't be made, so a video/* response means "no thumb available" —
  // abort before downloading the body and stream a <video> tag instead.
  useEffect(() => {
    if (isItAudio) return;
    if (!isVisible) return;

    const controller = new AbortController();
    let cancelled = false;
    let objectUrl: string | null = null;

    const timer = window.setTimeout(() => {
      fetch(`/api/img/${encodeFilePathForUrl(imageUrl)}?thumb=1`, { signal: controller.signal })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          if ((r.headers.get('content-type') || '').startsWith('video/')) {
            controller.abort();
            if (!cancelled) {
              setStreamVideo(true);
              setLoaded(true);
            }
            return null;
          }
          return r.blob();
        })
        .then(blob => {
          if (cancelled || !blob) return;
          objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          setLoaded(true);
        })
        .catch(err => {
          if (err?.name !== 'AbortError') console.error('Dataset image fetch failed:', err);
        });
    }, 80);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setBlobUrl(null);
      setStreamVideo(false);
      setLoaded(false);
    };
  }, [imageUrl, isItAudio, isVisible]);

  const combinedRefreshKey = captionRefreshKey + pollTick;
  const { caption: fetchedCaption, isLoaded: isCaptionLoaded } = useCaptionBatch(
    isVisible ? imageUrl : null,
    combinedRefreshKey,
    captionExt,
  );

  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const dirtyRef = useRef<boolean>(false);

  // Cards are keyed by image path, so this never needs resetting. Once loaded,
  // keep the textarea mounted through poll refreshes — swapping in the
  // "Loading caption..." placeholder unmounts it and resets scroll.
  const hasLoadedCaptionRef = useRef(false);
  if (isCaptionLoaded) hasLoadedCaptionRef.current = true;

  // Sync from the fetched caption, but don't clobber unsaved local edits.
  useEffect(() => {
    if (!isCaptionLoaded) return;
    if (dirtyRef.current) return;
    setCaption(fetchedCaption);
    setSavedCaption(fetchedCaption.trim());
  }, [fetchedCaption, isCaptionLoaded]);

  // Poll while auto-captioning so backend-written captions show up.
  useEffect(() => {
    if (!isAutoCaptioning) return;
    const interval = setInterval(() => setPollTick(t => t + 1), 5000);
    return () => clearInterval(interval);
  }, [isAutoCaptioning]);

  const saveCaption = () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) {
      dirtyRef.current = false;
      return;
    }
    apiClient
      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption, ext: captionExt })
      .then(() => {
        setSavedCaption(trimmedCaption);
        setCachedCaption(imageUrl, trimmedCaption, captionExt);
        dirtyRef.current = false;
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  // Save any pending edit if the card unmounts (e.g. scrolled out of the virtualized window).
  const latestRef = useRef({ caption, savedCaption, imageUrl, captionExt });
  useEffect(() => {
    latestRef.current = { caption, savedCaption, imageUrl, captionExt };
  });
  useEffect(() => {
    return () => {
      if (!dirtyRef.current) return;
      const { caption: c, savedCaption: s, imageUrl: url, captionExt: ext } = latestRef.current;
      const trimmed = c.trim();
      if (trimmed === s) return;
      apiClient
        .post('/api/img/caption', { imgPath: url, caption: trimmed, ext })
        .then(() => setCachedCaption(url, trimmed, ext))
        .catch(err => console.error('Error saving caption on unmount:', err));
    };
  }, []);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const handleCaptionChange = (value: string) => {
    dirtyRef.current = value.trim() !== savedCaption;
    setCaption(value);
  };

  const isCaptionCurrent = caption.trim() === savedCaption;

  return (
    <div ref={cardRef} className={`flex flex-col ${className}`}>
      <div className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div
          className={classNames('absolute inset-0 rounded-t-lg shadow-md bg-gray-900', {
            'animate-pulse': !isItAudio && !loaded,
          })}
        >
          {streamVideo && (
            <video
              src={`/api/img/${encodeFilePathForUrl(imageUrl)}`}
              className={classNames('w-full h-full object-contain', {
                'cursor-zoom-in': !!onImageClick,
              })}
              onClick={onImageClick}
              autoPlay={false}
              preload="metadata"
              playsInline
              loop
              muted
            />
          )}
          {isItAudio && !showAudioPlayer && (
            <div
              className="w-full h-full cursor-pointer flex items-center justify-center bg-gray-900"
              onClick={() => setShowAudioPlayer(true)}
            >
              <img
                src={`/api/audio/art/${encodeURIComponent(imageUrl)}`}
                alt={alt}
                className="w-full h-full object-contain"
                onError={e => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
          )}
          {isItAudio && showAudioPlayer && (
            <AudioPlayer src={`/api/img/${encodeFilePathForUrl(imageUrl)}`} title={imageUrl.replace(/^.*[\\/]/, '')} />
          )}
          {!isItAudio && blobUrl && (
            <img
              src={blobUrl}
              alt={alt}
              onClick={onImageClick}
              className={classNames('w-full h-full object-contain', {
                'cursor-zoom-in': !!onImageClick,
              })}
            />
          )}
          {isItAVideo && loaded && (
            <div className="absolute bottom-2 left-2 bg-gray-900/70 rounded-full p-2 pointer-events-none">
              <FaPlay className="w-3 h-3 text-white" />
            </div>
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          <div className="absolute top-1 right-1 flex space-x-2 z-10">
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                openConfirm({
                  title: `Delete ${isItAVideo ? 'video' : 'image'}`,
                  message: `Are you sure you want to delete this ${isItAVideo ? 'video' : 'image'}? This action cannot be undone.`,
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => {
                    apiClient
                      .post('/api/img/delete', { imgPath: imageUrl })
                      .then(() => {
                        console.log('Image deleted:', imageUrl);
                        onDelete();
                      })
                      .catch(error => {
                        console.error('Error deleting image:', error);
                      });
                  },
                });
              }}
            >
              <FaTrashAlt />
            </button>
          </div>
        </div>
      </div>
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
        })}
      >
        {isCaptionLoaded || hasLoadedCaptionRef.current ? (
          <form
            onSubmit={e => {
              e.preventDefault();
              saveCaption();
            }}
            onBlur={saveCaption}
          >
            <textarea
              className={classNames('w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none', {
                'opacity-50 cursor-not-allowed': isAutoCaptioning,
              })}
              value={caption}
              rows={3}
              readOnly={isAutoCaptioning}
              onChange={e => handleCaptionChange(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </form>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">Loading caption...</div>
        )}
      </div>
    </div>
  );
};

export default DatasetImageCard;
