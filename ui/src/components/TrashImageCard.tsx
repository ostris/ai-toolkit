import React, { useRef, useEffect, useState } from 'react';
import { FaTrashAlt, FaUndoAlt, FaCheckCircle } from 'react-icons/fa';
import classNames from 'classnames';
import { isVideo, isAudio } from '@/utils/basic';
import AudioPlayer from './AudioPlayer';

interface TrashImageCardProps {
  imageUrl: string;
  alt: string;
  selected?: boolean;
  isSelectMode?: boolean;
  onRestore?: () => void;
  onDelete?: () => void;
  onLongPress?: () => void;
  onSelect?: () => void;
}

const TrashImageCard: React.FC<TrashImageCardProps> = ({
  imageUrl,
  alt,
  selected = false,
  isSelectMode = false,
  onRestore,
  onDelete,
  onLongPress,
  onSelect,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [inViewport, setInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const didLongPress = useRef<boolean>(false);

  useEffect(() => {
    return () => {
      if (longPressTimer.current) {
        clearTimeout(longPressTimer.current);
      }
    };
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setInViewport(true);
        } else {
          setInViewport(false);
        }
      },
      { threshold: 0.1 },
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);

  const handleLoad = (): void => {
    setLoaded(true);
  };

  const handlePointerDown = () => {
    didLongPress.current = false;
    longPressTimer.current = setTimeout(() => {
      didLongPress.current = true;
      onLongPress?.();
    }, 500);
  };

  const clearLongPress = () => {
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
  };

  const handlePointerUp = () => clearLongPress();
  const handlePointerLeave = () => clearLongPress();
  const handlePointerCancel = () => clearLongPress();

  const handleCardClick = (e: React.MouseEvent) => {
    if (didLongPress.current) {
      e.preventDefault();
      e.stopPropagation();
      didLongPress.current = false;
      return;
    }
    if (isSelectMode) {
      e.preventDefault();
      e.stopPropagation();
      onSelect?.();
    }
  };

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);
  const isItImage = !isItAVideo && !isItAudio;

  // derive display name (strip trash_ prefix from basename)
  const basename = imageUrl.replace(/^.*[\\/]/, '');
  const displayName = basename.startsWith('trash_') ? basename.slice('trash_'.length) : basename;

  return (
    <div
      className={classNames('flex flex-col', {
        'ring-2 ring-blue-500 rounded-lg': selected,
      })}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onPointerCancel={handlePointerCancel}
      onContextMenu={e => e.preventDefault()}
      onClick={handleCardClick}
    >
      <div
        ref={cardRef}
        className="relative w-full"
        style={{ paddingBottom: '100%' }}
      >
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {inViewport && (
            <>
              {isItAVideo && (
                <video
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  className="w-full h-full object-contain"
                  autoPlay={false}
                  loop
                  controls
                />
              )}
              {isItAudio && (
                <AudioPlayer
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  title={displayName}
                />
              )}
              {isItImage && (
                <img
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  alt={alt}
                  onLoad={handleLoad}
                  className={`w-full h-full object-contain transition-opacity duration-300 ${
                    loaded ? 'opacity-100' : 'opacity-0'
                  }`}
                />
              )}
            </>
          )}
          {/* Selection overlay */}
          {isSelectMode && (
            <div
              className={classNames(
                'absolute inset-0 rounded-t-lg transition-colors duration-150 flex items-start justify-start p-2',
                selected ? 'bg-blue-500/30' : 'bg-transparent',
              )}
            >
              <FaCheckCircle
                className={classNames('w-6 h-6 transition-colors duration-150', selected ? 'text-blue-400' : 'text-gray-500')}
              />
            </div>
          )}
          {!isSelectMode && (
            <div className="absolute top-1 right-1 flex space-x-2 z-10">
              <button
                className="bg-gray-800 rounded-full p-2"
                onClick={e => { e.stopPropagation(); onRestore?.(); }}
                aria-label="Restore file"
                title="Restore"
              >
                <FaUndoAlt />
              </button>
              <button
                className="bg-gray-800 rounded-full p-2 text-red-400 hover:text-red-300"
                onClick={e => { e.stopPropagation(); onDelete?.(); }}
                aria-label="Permanently delete"
                title="Permanently delete"
              >
                <FaTrashAlt />
              </button>
            </div>
          )}
          {inViewport && !isItAudio && (
            <div className="text-xs text-gray-100 bg-gray-950 mt-1 absolute bottom-0 left-0 p-1 opacity-25 hover:opacity-90 transition-opacity duration-300 w-full truncate">
              {displayName}
            </div>
          )}
        </div>
      </div>
      {/* Bottom bar to match DatasetImageCard height */}
      <div className="w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px] flex items-center justify-center text-gray-400 text-xs">
        <span className="truncate px-1">{displayName}</span>
      </div>
    </div>
  );
};

export default TrashImageCard;
