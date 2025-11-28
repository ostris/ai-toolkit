import React, { useRef, useEffect, useState, ReactNode } from 'react';
import { isVideo } from '@/utils/basic';

interface SampleImageCardProps {
  imageUrl: string;
  alt: string;
  numSamples: number;
  sampleImages: string[];
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onClick?: () => void;
  /** pass your scroll container element (e.g. containerRef.current) */
  observerRoot?: Element | null;
  /** optional: tweak pre-load buffer */
  rootMargin?: string; // default '200px 0px'
}

const SampleImageCard: React.FC<SampleImageCardProps> = ({
  imageUrl,
  alt,
  numSamples,
  sampleImages,
  children,
  className = '',
  onClick = () => {},
  observerRoot = null,
  rootMargin = '200px 0px',
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Observe both enter and exit
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

  // Pause video when leaving viewport
  useEffect(() => {
    if (!isVideo(imageUrl)) return;
    const v = videoRef.current;
    if (!v) return;
    if (!isVisible && !v.paused) {
      try {
        v.pause();
      } catch {}
    }
  }, [isVisible, imageUrl]);

  const handleLoad = () => setLoaded(true);

  return (
    <div className={`flex flex-col ${className}`}>
      <div ref={cardRef} className="relative w-full cursor-pointer" style={{ paddingBottom: '100%' }} onClick={onClick}>
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {isVisible ? (
            isVideo(imageUrl) ? (
              <video
                ref={videoRef}
                src={`/aitoolkit/api/img/${encodeURIComponent(imageUrl)}`}
                className="w-full h-full object-cover"
                preload="none"
                playsInline
                muted
                loop
                controls={false}
              />
            ) : (
              <img
                src={`/aitoolkit/api/img/${encodeURIComponent(imageUrl)}`}
                alt={alt}
                onLoad={handleLoad}
                loading="lazy"
                decoding="async"
                className={`w-full h-full object-cover transition-opacity duration-300 ${
                  loaded ? 'opacity-100' : 'opacity-0'
                }`}
              />
            )
          ) : null}

          {children && isVisible && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
        </div>
      </div>
    </div>
  );
};

export default SampleImageCard;
