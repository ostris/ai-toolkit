import React, { useRef, useEffect, useState, ReactNode } from 'react';
import { isVideo, isAudio, isText, encodeFilePathForUrl } from '@/utils/basic';

interface SampleImageCardProps {
  imageUrl: string;
  alt: string;
  numSamples: number;
  sampleImages: string[];
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
  selected?: boolean;
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
  selected = false,
  observerRoot = null,
  rootMargin = '200px 0px',
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  // videos with no pre-generated thumb (older samples) fall back to the <video> element
  const [videoFallback, setVideoFallback] = useState(false);

  const isItAudio = isAudio(imageUrl);
  const isItVideo = isVideo(imageUrl);
  const isItText = isText(imageUrl);
  const [text, setText] = useState<string | null>(null);

  // text samples: fetch the file body and render it in the card
  useEffect(() => {
    if (!isItText || !isVisible) return;
    const controller = new AbortController();
    fetch(`/api/img/${encodeFilePathForUrl(imageUrl)}`, { signal: controller.signal })
      .then(r => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(t => {
        setText(t);
        setLoaded(true);
      })
      .catch(err => {
        if (err?.name !== 'AbortError') console.error('Sample text fetch failed:', err);
      });
    return () => {
      controller.abort();
      setText(null);
      setLoaded(false);
    };
  }, [isItText, isVisible, imageUrl]);

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
    if (isItAudio || isItText) return;
    if (!isVisible) return;

    const controller = new AbortController();
    let cancelled = false;
    let objectUrl: string | null = null;

    const timer = window.setTimeout(() => {
      // ?thumb=1: the server sends the small pre-generated thumbnail when one
      // exists, otherwise the full file. Videos without a thumb come back as
      // video/* — abort the transfer and render the <video> element instead.
      fetch(`/api/img/${encodeFilePathForUrl(imageUrl)}?thumb=1`, { signal: controller.signal })
        .then(r => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          const contentType = r.headers.get('content-type') || '';
          if (isItVideo && !contentType.startsWith('image/')) {
            controller.abort();
            if (!cancelled) {
              setVideoFallback(true);
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
      setVideoFallback(false);
    };
  }, [isVisible, isItAudio, isItText, isItVideo, imageUrl]);

  return (
    <div className={`flex flex-col ${className}`}>
      <div
        ref={cardRef}
        className={`relative w-full cursor-pointer select-none rounded-t-lg transition-colors duration-200 ${
          selected ? 'bg-blue-500' : ''
        }`}
        style={{ paddingBottom: '100%' }}
        onClick={onClick}
      >
        <div
          className={`absolute rounded-t-lg shadow-md bg-gray-900 overflow-hidden transition-all duration-200 [container-type:inline-size] ${
            selected ? 'inset-2' : 'inset-0'
          } ${isVisible && !isItAudio && !loaded ? 'animate-pulse' : ''}`}
        >
          {isVisible ? (
            isItText ? (
              // font size scales with the card and the text length: a square of side W holds
              // ~W^2 / (0.6 f^2) characters at size f, so f ~ 120/sqrt(n) cqw fills the box
              <div
                className="w-full h-full overflow-hidden p-[3cqw] leading-snug text-gray-200 whitespace-pre-wrap break-words text-left bg-gray-900"
                style={{
                  fontSize: `clamp(2.5cqw, ${(120 / Math.sqrt(Math.max((text ?? '').length, 1))).toFixed(2)}cqw, 12cqw)`,
                }}
              >
                {text ?? ''}
              </div>
            ) : isItAudio ? (
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
            ) : isItVideo && videoFallback ? (
              <video
                ref={videoRef}
                src={`/api/img/${encodeFilePathForUrl(imageUrl)}`}
                className="w-full h-full object-cover"
                preload="none"
                playsInline
                muted
                loop
                autoPlay
                controls={false}
              />
            ) : blobUrl ? (
              <img src={blobUrl} alt={alt} className="w-full h-full object-cover" />
            ) : null
          ) : null}

          {children && isVisible && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
        </div>
      </div>
    </div>
  );
};

export default SampleImageCard;
