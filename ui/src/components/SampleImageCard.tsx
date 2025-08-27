import React, { useRef, useEffect, useState, ReactNode } from 'react';
import { sampleImageModalState } from '@/components/SampleImageModal';
import { isVideo } from '@/utils/basic';

interface SampleImageCardProps {
  imageUrl: string;
  alt: string;
  numSamples: number;
  sampleImages: string[];
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
}

const SampleImageCard: React.FC<SampleImageCardProps> = ({
  imageUrl,
  alt,
  numSamples,
  sampleImages,
  children,
  className = '',
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);

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

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Square image container */}
      <div
        ref={cardRef}
        className="relative w-full cursor-pointer"
        style={{ paddingBottom: '100%' }} // Make it square
        onClick={() => sampleImageModalState.set({ imgPath: imageUrl, numSamples, sampleImages })}
      >
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {isVisible && (
            <>
              {isVideo(imageUrl) ? (
                <video
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  className={`w-full h-full object-cover`}
                  autoPlay={false}
                  loop
                  muted
                  playsInline
                />
              ) : (
                <img
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  alt={alt}
                  onLoad={handleLoad}
                  className={`w-full h-full object-cover transition-opacity duration-300 ${
                    loaded ? 'opacity-100' : 'opacity-0'
                  }`}
                />
              )}
            </>
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
        </div>
      </div>
    </div>
  );
};

export default SampleImageCard;
