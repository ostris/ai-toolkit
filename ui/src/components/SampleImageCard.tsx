import React, { useRef, useEffect, useState, ReactNode } from 'react';
import { isVideo, isAudio } from '@/utils/basic';

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
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  const isItAudio = isAudio(imageUrl);
  const isItVideo = isVideo(imageUrl);
  const isImageType = !isItAudio && !isItVideo;

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

  // Drive image loads through fetch + AbortController so scrolling past actually
  // cancels in-flight requests (browsers don't reliably cancel <img> fetches when
  // the element unmounts). A short debounce skips requests entirely during fast
  // scrolls where the card is only briefly visible.
  useEffect(() => {
    if (!isImageType) return;
    if (!isVisible) return;

    const controller = new AbortController();
    let cancelled = false;
    let objectUrl: string | null = null;

    const timer = window.setTimeout(() => {
      fetch(`/api/img/${encodeURIComponent(imageUrl)}`, { signal: controller.signal })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.blob();
        })
        .then(blob => {
          if (cancelled) return;
          objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          setLoaded(true);
        })
        .catch(err => {
          if (err?.name !== 'AbortError') console.error('Sample image fetch failed:', err);
        });
    }, 80);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      setBlobUrl(null);
      setLoaded(false);
    };
  }, [isVisible, isImageType, imageUrl]);

  return (
    <div className={`flex flex-col ${className}`}>
      <div ref={cardRef} className="relative w-full cursor-pointer" style={{ paddingBottom: '100%' }} onClick={onClick}>
        <div
          className={`absolute inset-0 rounded-t-lg shadow-md bg-gray-900 ${
            isVisible && isImageType && !loaded ? 'animate-pulse' : ''
          }`}
        >
          {isVisible ? (
            isItAudio ? (
              <div className="w-full h-full flex items-center justify-center bg-gray-900">
                <img
                  src={`/api/audio/art/${encodeURIComponent(imageUrl)}`}
                  alt={alt}
                  className="w-full h-full object-cover"
                  onError={e => {
                    (e.target as HTMLImageElement).style.display = 'none';
                  }}
                />
              </div>
            ) : isItVideo ? (
              <video
                ref={videoRef}
                src={`/api/img/${encodeURIComponent(imageUrl)}`}
                className="w-full h-full object-cover"
                preload="none"
                playsInline
                muted
                loop
                autoPlay
                controls={false}
              />
            ) : blobUrl ? (
              <img
                src={blobUrl}
                alt={alt}
                decoding="async"
                className={`w-full h-full object-cover transition-opacity duration-300 ${
                  loaded ? 'opacity-100' : 'opacity-0'
                }`}
              />
            ) : null
          ) : null}

          {children && isVisible && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
        </div>
      </div>
    </div>
  );
};

export default SampleImageCard;
