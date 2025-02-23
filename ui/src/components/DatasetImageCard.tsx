import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent } from 'react';
import { FaTrashAlt } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  children,
  className = '',
  onDelete = () => {},
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const isGettingCaption = useRef<boolean>(false);

  const fetchCaption = async () => {
    try {
      if (isGettingCaption.current || isCaptionLoaded) return;
      isGettingCaption.current = true;
      const response = await fetch(`/api/caption/${encodeURIComponent(imageUrl)}`);
      const data = await response.text();
      setCaption(data);
      setSavedCaption(data);
      setIsCaptionLoaded(true);
    } catch (error) {
      console.error('Error fetching caption:', error);
    }
  };

  const saveCaption = () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    fetch('/api/img/caption', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imgPath: imageUrl, caption: trimmedCaption }),
    })
      .then(res => res.json())
      .then(data => {
        console.log('Caption saved:', data);
        setSavedCaption(trimmedCaption);
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  useEffect(() => {
    isVisible && fetchCaption();
  }, [isVisible]);

  useEffect(() => {
    // Create intersection observer to check visibility
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
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

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    // If Enter is pressed without Shift, prevent default behavior and save
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption;

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Square image container */}
      <div
        ref={cardRef}
        className="relative w-full"
        style={{ paddingBottom: '100%' }} // Make it square
      >
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {isVisible && (
            <img
              src={`/api/img/${encodeURIComponent(imageUrl)}`}
              alt={alt}
              onLoad={handleLoad}
              className={`w-full h-full object-contain transition-opacity duration-300 ${
                loaded ? 'opacity-100' : 'opacity-0'
              }`}
            />
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          <div className="absolute top-1 right-1">
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                openConfirm({
                  title: 'Delete Image',
                  message: 'Are you sure you want to delete this image? This action cannot be undone.',
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => {
                    fetch('/api/img/delete', {
                      method: 'POST',
                      headers: {
                        'Content-Type': 'application/json',
                      },
                      body: JSON.stringify({ imgPath: imageUrl }),
                    })
                      .then(res => res.json())
                      .then(data => {
                        console.log('Image deleted:', data);
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

      {/* Text area below the image */}
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
        })}
      >
        {isVisible && isCaptionLoaded && (
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
      </div>
    </div>
  );
};

export default DatasetImageCard;
