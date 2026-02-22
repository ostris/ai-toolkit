import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent } from 'react';
import { FaTrashAlt, FaEye, FaEyeSlash, FaExpand, FaUndoAlt, FaRedoAlt, FaCheckCircle, FaCut } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio } from '@/utils/basic';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onSplit?: () => void;
  onEnlarge?: () => void;
  selected?: boolean;
  isSelectMode?: boolean;
  onLongPress?: () => void;
  onSelect?: () => void;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  children,
  className = '',
  onDelete = () => {},
  onSplit,
  onEnlarge,
  selected = false,
  isSelectMode = false,
  onLongPress,
  onSelect,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [inViewport, setInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [imageKey, setImageKey] = useState<number>(Date.now());
  const isGettingCaption = useRef<boolean>(false);
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

  const handleSplitVideo = () => {
    openConfirm({
      title: 'Split Video',
      message: 'Enter the number of seconds per segment to split the video into.',
      inputTitle: 'Seconds per segment (e.g. 30)',
      confirmText: 'Split',
      type: 'info',
      onConfirm: (value?: string) => {
        const seconds = parseInt(value || '', 10);
        if (isNaN(seconds) || seconds < 1) {
          openConfirm({
            title: 'Invalid Input',
            message: 'Please enter a valid number of seconds (minimum 1).',
            confirmText: 'OK',
            type: 'warning',
          });
          return;
        }
        apiClient
          .post('/api/video/split', { videoPath: imageUrl, secondsPerSegment: seconds })
          .then(() => {
            onSplit?.();
          })
          .catch(error => {
            console.error('Error splitting video:', error);
          });
      },
    });
  };

  // Only fetch caption when the component is both in viewport and visible
  useEffect(() => {
    if (inViewport && isVisible) {
      fetchCaption();
    }
  }, [inViewport, isVisible]);

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
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  className={`w-full h-full object-contain`}
                  autoPlay={false}
                  loop
                  muted
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
                'absolute inset-0 rounded-t-lg transition-colors duration-150 flex items-start justify-start p-2 pointer-events-none',
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
                onClick={handleSplitVideo}
                aria-label="Split video"
              >
                <FaCut />
              </button>
            )}
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                    apiClient
                      .post('/api/img/delete', { imgPath: imageUrl })
                      .then(() => {
                        console.log('Image deleted:', imageUrl);
                        onDelete();
                      })
                      .catch(error => {
                        console.error('Error deleting image:', error);
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
    </div>
  );
};

export default DatasetImageCard;
