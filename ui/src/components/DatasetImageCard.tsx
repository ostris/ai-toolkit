import React, { useEffect, useState, ReactNode, KeyboardEvent, useRef } from 'react';
import { FaTrashAlt } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio } from '@/utils/basic';
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
}) => {
  const [loaded, setLoaded] = useState<boolean>(false);
  const [showAudioPlayer, setShowAudioPlayer] = useState(true);
  const [pollTick, setPollTick] = useState(0);

  const combinedRefreshKey = captionRefreshKey + pollTick;
  const { caption: fetchedCaption, isLoaded: isCaptionLoaded } = useCaptionBatch(imageUrl, combinedRefreshKey);

  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const dirtyRef = useRef<boolean>(false);

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
      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption })
      .then(() => {
        setSavedCaption(trimmedCaption);
        setCachedCaption(imageUrl, trimmedCaption);
        dirtyRef.current = false;
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  // Save any pending edit if the card unmounts (e.g. scrolled out of the virtualized window).
  const latestRef = useRef({ caption, savedCaption, imageUrl });
  useEffect(() => {
    latestRef.current = { caption, savedCaption, imageUrl };
  });
  useEffect(() => {
    return () => {
      if (!dirtyRef.current) return;
      const { caption: c, savedCaption: s, imageUrl: url } = latestRef.current;
      const trimmed = c.trim();
      if (trimmed === s) return;
      apiClient
        .post('/api/img/caption', { imgPath: url, caption: trimmed })
        .then(() => setCachedCaption(url, trimmed))
        .catch(err => console.error('Error saving caption on unmount:', err));
    };
  }, []);

  const handleLoad = (): void => {
    setLoaded(true);
  };

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

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);
  const isItImage = !isItAVideo && !isItAudio;

  return (
    <div className={`flex flex-col ${className}`}>
      <div className="relative w-full" style={{ paddingBottom: '100%' }}>
        <div
          className={classNames('absolute inset-0 rounded-t-lg shadow-md bg-gray-900', {
            'animate-pulse': isItImage && !loaded,
          })}
        >
          {isItAVideo && (
            <video
              src={`/api/img/${encodeURIComponent(imageUrl)}`}
              className={`w-full h-full object-contain`}
              autoPlay={false}
              loop
              muted
              controls
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
            <AudioPlayer src={`/api/img/${encodeURIComponent(imageUrl)}`} title={imageUrl.replace(/^.*[\\/]/, '')} />
          )}
          {isItImage && (
            <img
              src={`/api/img/${encodeURIComponent(imageUrl)}`}
              alt={alt}
              onLoad={handleLoad}
              onClick={onImageClick}
              className={classNames('w-full h-full object-contain transition-opacity duration-300', {
                'opacity-100': loaded,
                'opacity-0': !loaded,
                'cursor-zoom-in': !!onImageClick,
              })}
            />
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
        {isCaptionLoaded ? (
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
