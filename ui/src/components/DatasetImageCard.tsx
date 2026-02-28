import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent } from 'react';
import { FaTrashAlt, FaEye, FaEyeSlash, FaExpand, FaUndoAlt, FaRedoAlt, FaCheckCircle, FaCut, FaObjectGroup, FaArrowsAlt } from 'react-icons/fa';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import VideoTrimModal from './VideoTrimModal';
import MoveImageModal from './MoveImageModal';
import { isVideo, isAudio } from '@/utils/basic';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onSplit?: () => void;
  onTrim?: () => void;
  onMerge?: () => void;
  onEnlarge?: () => void;
  onMove?: (operation: 'move' | 'copy') => void;
  currentDataset?: string;
  selected?: boolean;
  isSelectMode?: boolean;
  onLongPress?: () => void;
  onSelect?: () => void;
  scoreRefreshKey?: number;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  children,
  className = '',
  onDelete = () => {},
  onSplit,
  onTrim,
  onMerge,
  onEnlarge,
  onMove,
  currentDataset = '',
  selected = false,
  isSelectMode = false,
  onLongPress,
  onSelect,
  scoreRefreshKey,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [inViewport, setInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [imageKey, setImageKey] = useState<number>(Date.now());
  const [videoKey, setVideoKey] = useState<number>(Date.now());
  const [isVideoEditOpen, setIsVideoEditOpen] = useState<boolean>(false);
  const [isMoveModalOpen, setIsMoveModalOpen] = useState<boolean>(false);
  const [scores, setScores] = useState<Record<string, number> | null>(null);
  const isGettingCaption = useRef<boolean>(false);
  const isGettingScores = useRef<boolean>(false);
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const didLongPress = useRef<boolean>(false);

  useEffect(() => {
    return () => {
      if (longPressTimer.current) {
        clearTimeout(longPressTimer.current);
      }
    };
  }, []);

  const fetchCaption = async () => {
    if (isGettingCaption.current || isCaptionLoaded) return;
    isGettingCaption.current = true;
    apiClient
      .post(`/api/caption/get`, { imgPath: imageUrl })
      .then(res => res.data)
      .then(data => {
        console.log('Caption fetched:', data);
        if (data) {
          // fix issue where caption could be non string
          data = `${data}`;
        }
        setCaption(data || '');
        setSavedCaption(data || '');
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        console.error('Error fetching caption:', error);
      })
      .finally(() => {
        isGettingCaption.current = false;
      });
  };

  const fetchScores = () => {
    if (isGettingScores.current) return;
    isGettingScores.current = true;
    apiClient
      .get(`/api/datasets/imageScores?imgPath=${encodeURIComponent(imageUrl)}`)
      .then(res => res.data)
      .then(data => {
        setScores(data.scores || {});
      })
      .catch(error => {
        console.error('Error fetching scores:', error);
      })
      .finally(() => {
        isGettingScores.current = false;
      });
  };

  const saveCaption = () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    apiClient
      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption })
      .then(res => res.data)
      .then(data => {
        console.log('Caption saved:', data);
        setSavedCaption(trimmedCaption);
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  const rotateImage = (direction: 'left' | 'right') => {
    apiClient
      .post('/api/img/rotate', { imgPath: imageUrl, direction })
      .then(() => {
        setLoaded(false);
        setImageKey(prev => prev + 1);
      })
      .catch(error => {
        console.error('Error rotating image:', error);
      });
  };

  const handleVideoEdit = () => {
    setIsVideoEditOpen(true);
  };

  // Only fetch caption when the component is both in viewport and visible
  useEffect(() => {
    if (inViewport && isVisible) {
      fetchCaption();
      fetchScores();
    }
  }, [inViewport, isVisible]);

  // Re-fetch scores when scoreRefreshKey changes (e.g., after scoring completes)
  useEffect(() => {
    if (scoreRefreshKey !== undefined && inViewport && isVisible) {
      isGettingScores.current = false;
      fetchScores();
    }
  }, [scoreRefreshKey]);

  useEffect(() => {
    // Create intersection observer to check viewport visibility
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setInViewport(true);
          // Initialize isVisible to true when first coming into view
          if (!isVisible) {
            setIsVisible(true);
          }
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

  const toggleVisibility = (): void => {
    setIsVisible(prev => !prev);
    if (!isVisible && !isCaptionLoaded) {
      fetchCaption();
    }
  };

  const handleLoad = (): void => {
    setLoaded(true);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    // If Enter is pressed without Shift, prevent default behavior and save
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
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

  const isCaptionCurrent = caption.trim() === savedCaption;

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);
  const isItImage = !isItAVideo && !isItAudio;

  return (
    <div
      className={classNames(`flex flex-col ${className}`, {
        'ring-2 ring-blue-500 rounded-lg': selected,
      })}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onPointerCancel={handlePointerCancel}
      onContextMenu={e => e.preventDefault()}
      onClick={handleCardClick}
    >
      {/* Square image container */}
      <div
        ref={cardRef}
        className="relative w-full"
        style={{ paddingBottom: '100%' }} // Make it square
      >
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {inViewport && isVisible && (
            <>
              {isItAVideo && (
                <video
                  key={videoKey}
                  src={`/api/img/${encodeURIComponent(imageUrl)}?v=${videoKey}`}
                  className={`w-full h-full object-contain`}
                  autoPlay={false}
                  loop
                  controls
                />
              )}
              {isItAudio && (
                <AudioPlayer
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  title={imageUrl.replace(/^.*[\\/]/, '')}
                />
              )}
              {isItImage && (
                <img
                  key={imageKey}
                  src={`/api/img/${encodeURIComponent(imageUrl)}?v=${imageKey}`}
                  alt={alt}
                  onLoad={handleLoad}
                  onClick={isSelectMode ? undefined : onEnlarge}
                  className={`w-full h-full object-contain transition-opacity duration-300 ${
                    loaded ? 'opacity-100' : 'opacity-0'
                  } ${onEnlarge && !isSelectMode ? 'cursor-pointer' : ''}`}
                />
              )}
            </>
          )}
          {!isVisible && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 rounded-t-lg">
              <span className="text-white text-lg"></span>
            </div>
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
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
            {onEnlarge && isItImage && (
              <button
                className="bg-gray-800 rounded-full p-2"
                onClick={onEnlarge}
                aria-label="Enlarge image"
              >
                <FaExpand />
              </button>
            )}
            {isItImage && (
              <button
                className="bg-gray-800 rounded-full p-2"
                onClick={() => rotateImage('left')}
                aria-label="Rotate image left"
              >
                <FaUndoAlt />
              </button>
            )}
            {isItImage && (
              <button
                className="bg-gray-800 rounded-full p-2"
                onClick={() => rotateImage('right')}
                aria-label="Rotate image right"
              >
                <FaRedoAlt />
              </button>
            )}
            {isItAVideo && (
              <button
                className="bg-gray-800 rounded-full p-2"
                onClick={handleVideoEdit}
                aria-label="Edit video (trim or split)"
              >
                <FaCut />
              </button>
            )}
            {isItAVideo && onMerge && (
              <button
                className="bg-gray-800 rounded-full p-2"
                onClick={onMerge}
                aria-label="Merge video clips"
              >
                <FaObjectGroup />
              </button>
            )}
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => setIsMoveModalOpen(true)}
              aria-label="Move or copy to another dataset"
            >
              <FaArrowsAlt />
            </button>
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                    apiClient
                      .post('/api/img/trash', { imgPath: imageUrl })
                      .then(() => {
                        console.log('Image moved to trash:', imageUrl);
                        onDelete();
                      })
                      .catch(error => {
                        console.error('Error moving image to trash:', error);
                      });
                  }}
            >
              <FaTrashAlt />
            </button>
          </div>
          )}
        </div>
        {inViewport && isVisible && !isItAudio && (
          <div className="text-xs text-gray-100 bg-gray-950 mt-1 absolute bottom-0 left-0 p-1 opacity-25 hover:opacity-90 transition-opacity duration-300 w-full">
            {imageUrl}
          </div>
        )}
      </div>
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
        })}
      >
        {inViewport && isVisible && isCaptionLoaded && (
          <form
            onSubmit={e => {
              e.preventDefault();
              saveCaption();
            }}
            onBlur={saveCaption}
          >
            <textarea
              className="w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none"
              value={caption}
              rows={3}
              onChange={e => setCaption(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </form>
        )}
        {(!inViewport || !isVisible) && isCaptionLoaded && (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            {isVisible ? 'Scroll into view to edit caption' : 'Show content to edit caption'}
          </div>
        )}
        {!isCaptionLoaded && (
          <div className="w-full h-full flex items-center justify-center text-gray-400">Loading caption...</div>
        )}
      </div>
      {scores && Object.keys(scores).length > 0 && (
        <div className="w-full px-2 py-1 bg-gray-900 rounded-b-lg flex flex-wrap gap-1">
          {Object.entries(scores).map(([metric, value]) => (
            <span
              key={metric}
              className="text-xs bg-gray-700 text-gray-200 px-2 py-0.5 rounded-full"
              title={metric}
            >
              {metric}: {value.toFixed(2)}
            </span>
          ))}
        </div>
      )}
      {isItAVideo && (
        <VideoTrimModal
          videoUrl={imageUrl}
          isOpen={isVideoEditOpen}
          onClose={() => setIsVideoEditOpen(false)}
          onTrim={() => { setIsVideoEditOpen(false); setVideoKey(Date.now()); onTrim?.(); }}
          onSplit={() => { setIsVideoEditOpen(false); onSplit?.(); }}
        />
      )}
      <MoveImageModal
        isOpen={isMoveModalOpen}
        onClose={() => setIsMoveModalOpen(false)}
        imageUrl={imageUrl}
        currentDataset={currentDataset}
        onComplete={(operation) => {
          if (operation === 'move') {
            onDelete();
          }
          onMove?.(operation);
        }}
      />
    </div>
  );
};

export default DatasetImageCard;
