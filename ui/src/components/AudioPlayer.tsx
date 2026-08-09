'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

type AudioPlayerProps = {
  src: string;

  /** Fallbacks (used only if meta missing) */
  title?: string;
  subtitle?: string;

  /** Optional: default background image if no embedded album art is found */
  defaultAlbumArtUrl?: string;

  className?: string;
  autoPlay?: boolean;
  onPlay?: () => void;
  onPause?: () => void;
};

function clamp(n: number, a: number, b: number) {
  return Math.min(b, Math.max(a, n));
}
function fmtTime(sec: number) {
  if (!isFinite(sec) || sec < 0) return '0:00';
  const s = Math.floor(sec % 60);
  const m = Math.floor(sec / 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/**
 * Global “only one plays at a time” channel.
 */
const AUDIO_EXCLUSIVE_EVENT = 'app:exclusive-audio-play';
type ExclusivePlayDetail = { token: string };
function broadcastExclusivePlay(token: string) {
  window.dispatchEvent(new CustomEvent<ExclusivePlayDetail>(AUDIO_EXCLUSIVE_EVENT, { detail: { token } }));
}

/**
 * Build the server-side album-art URL from the audio src.
 * The audio src is `/api/img/{encodedPath}` — we extract the path
 * and point to `/api/audio/art/{encodedPath}` instead.
 */
function albumArtUrlFromSrc(src: string): string {
  const prefix = '/api/img/';
  if (src.startsWith(prefix)) {
    return `/api/audio/art/${src.slice(prefix.length)}`;
  }
  // Fallback: assume src is already an encoded path
  return `/api/audio/art/${encodeURIComponent(src)}`;
}

export default function AudioPlayer({
  src,
  title = 'Audio',
  subtitle,
  defaultAlbumArtUrl,
  className = '',
  autoPlay = false,
  onPlay,
  onPause,
}: AudioPlayerProps) {
  const tokenRef = useRef(`aud_${Math.random().toString(16).slice(2)}_${Date.now()}`);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const barRef = useRef<HTMLDivElement | null>(null);
  const waveRef = useRef<HTMLCanvasElement | null>(null);

  const [size, setSize] = useState(256);

  const [isReady, setIsReady] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isBuffering, setIsBuffering] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [duration, setDuration] = useState(0);
  const [t, setT] = useState(0);

  const [dragging, setDragging] = useState(false);
  const [dragValue, setDragValue] = useState(0);

  // Album art: served by /api/audio/art endpoint (fast, server-side extraction)
  const [albumArtUrl, setAlbumArtUrl] = useState<string | null>(null);

  // WebAudio analyser
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const timeRef = useRef<Uint8Array | null>(null);
  const rafRef = useRef<number | null>(null);

  // Smoothed energy
  const energyRef = useRef(0);

  // Resize observer (square)
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const r = entries[0]?.contentRect;
      if (!r) return;
      setSize(Math.max(1, Math.floor(Math.min(r.width, r.height))));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // DPR-aware canvas sizing
  useEffect(() => {
    const c = waveRef.current;
    if (!c) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cssW = c.clientWidth || 1;
    const cssH = c.clientHeight || 1;
    c.width = Math.max(1, Math.floor(cssW * dpr));
    c.height = Math.max(1, Math.floor(cssH * dpr));
  }, [size]);

  // Set album art URL from server endpoint
  useEffect(() => {
    const artUrl = albumArtUrlFromSrc(src);
    let cancelled = false;
    apiClient
      .head(artUrl)
      .then(() => {
        if (!cancelled) setAlbumArtUrl(artUrl);
      })
      .catch(() => {
        if (!cancelled) setAlbumArtUrl(defaultAlbumArtUrl ?? null);
      });
    return () => {
      cancelled = true;
    };
  }, [src, defaultAlbumArtUrl]);

  const progress = useMemo(() => {
    const cur = dragging ? dragValue : t;
    if (!duration) return 0;
    return clamp(cur / duration, 0, 1);
  }, [t, duration, dragging, dragValue]);

  const elapsed = dragging ? dragValue : t;
  const remaining = Math.max(0, (duration || 0) - elapsed);

  // Load src into <audio>
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    setErr(null);
    setIsReady(false);
    setIsBuffering(false);
    setIsPlaying(false);
    setDuration(0);
    setT(0);
    setDragging(false);
    setDragValue(0);

    el.src = src;
    el.preload = 'metadata';
    el.load();
  }, [src]);

  // Audio events
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    const onLoaded = () => {
      setDuration(isFinite(el.duration) ? el.duration : 0);
      setIsReady(true);
      if (autoPlay) void safePlay();
    };
    const onTime = () => {
      if (!dragging) setT(el.currentTime || 0);
    };
    const onPlayEvt = () => setIsPlaying(true);
    const onPauseEvt = () => setIsPlaying(false);
    const onWaiting = () => setIsBuffering(true);
    const onPlayingEvt = () => setIsBuffering(false);
    const onEnded = () => setIsPlaying(false);
    const onError = () => {
      setIsPlaying(false);
      setIsBuffering(false);
      setErr('Failed to load audio.');
    };

    el.addEventListener('loadedmetadata', onLoaded);
    el.addEventListener('timeupdate', onTime);
    el.addEventListener('play', onPlayEvt);
    el.addEventListener('pause', onPauseEvt);
    el.addEventListener('waiting', onWaiting);
    el.addEventListener('playing', onPlayingEvt);
    el.addEventListener('ended', onEnded);
    el.addEventListener('error', onError);

    return () => {
      el.removeEventListener('loadedmetadata', onLoaded);
      el.removeEventListener('timeupdate', onTime);
      el.removeEventListener('play', onPlayEvt);
      el.removeEventListener('pause', onPauseEvt);
      el.removeEventListener('waiting', onWaiting);
      el.removeEventListener('playing', onPlayingEvt);
      el.removeEventListener('ended', onEnded);
      el.removeEventListener('error', onError);
    };
  }, [dragging, autoPlay]);

  // Exclusive playback listener
  useEffect(() => {
    const handler = (e: Event) => {
      const ce = e as CustomEvent<ExclusivePlayDetail>;
      const other = ce.detail?.token;
      if (!other) return;
      if (other !== tokenRef.current) {
        if (audioRef.current && !audioRef.current.paused) audioRef.current.pause();
      }
    };
    window.addEventListener(AUDIO_EXCLUSIVE_EVENT, handler as EventListener);
    return () => window.removeEventListener(AUDIO_EXCLUSIVE_EVENT, handler as EventListener);
  }, []);

  // Keep the canvas progress overlay updating smoothly while playing.
  // This fixes “only updates on start/stop” by forcing continuous redraw with the current progress.
  useEffect(() => {
    if (isPlaying) startLoop();
    else stopLoop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, duration]);

  function ensureAudioGraph() {
    if (audioCtxRef.current) return;
    const el = audioRef.current;
    if (!el) return;

    const Ctx = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext | undefined;
    if (!Ctx) return;

    const ctx = new Ctx();
    const srcNode = ctx.createMediaElementSource(el);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.85;

    srcNode.connect(analyser);
    analyser.connect(ctx.destination);

    audioCtxRef.current = ctx;
    analyserRef.current = analyser;
    timeRef.current = new Uint8Array(analyser.fftSize);
  }

  function stopLoop() {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    energyRef.current = 0;
  }

  function startLoop() {
    if (rafRef.current != null) return;

    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);

      const analyser = analyserRef.current;
      const time = timeRef.current;

      let targetEnergy = 0;
      if (analyser && time) {
        analyser.getByteTimeDomainData(time as Uint8Array<any>);
        let sum = 0;
        for (let i = 0; i < time.length; i++) sum += Math.abs(time[i] - 128);
        targetEnergy = sum / time.length / 128; // 0..1
      }

      const prev = energyRef.current;
      const next = prev + (targetEnergy - prev) * 0.12;
      energyRef.current = next;

      // IMPORTANT: use real-time currentTime here so progress updates even if React state lags.
      const el = audioRef.current;
      const cur = dragging ? dragValue : (el?.currentTime ?? t);
      const prog = duration > 0 ? clamp(cur / duration, 0, 1) : 0;

      drawWave(next, analyser, time, prog);
    };

    rafRef.current = requestAnimationFrame(tick);
  }

  function drawPath(ctx: CanvasRenderingContext2D, time: Uint8Array, w: number, h: number, energy: number) {
    const mid = h * 0.5;
    const amp = h * (0.26 + energy * 0.3);
    ctx.beginPath();
    for (let i = 0; i < time.length; i++) {
      const x = (i / (time.length - 1)) * w;
      const v = (time[i] - 128) / 128;
      const y = mid + v * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  }

  /**
   * Waveform is ALWAYS yellow.
   * Progress is shown as a subtle background shade behind it (matching percentage).
   */
  function drawWave(energy: number, analyser: AnalyserNode | null, time: Uint8Array | null, prog: number) {
    const c = waveRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;

    const w = c.width;
    const h = c.height;
    ctx.clearRect(0, 0, w, h);

    if (!analyser || !time) return;

    analyser.getByteTimeDomainData(time as Uint8Array<any>);

    // progress background (behind waveform), semi-transparent
    const playedX = Math.floor(w * prog);
    ctx.fillStyle = 'rgba(251,191,36,0.10)'; // faint amber wash
    ctx.fillRect(0, 0, playedX, h);

    // waveform ALWAYS yellow, semi-transparent
    drawPath(ctx, time, w, h, energy);
    ctx.strokeStyle = 'rgba(251,191,36,0.70)'; // #fbbf24 @ alpha
    ctx.lineWidth = Math.max(2, Math.floor(h * 0.02));
    ctx.stroke();
  }

  async function safePlay() {
    const el = audioRef.current;
    if (!el) return;
    setErr(null);

    broadcastExclusivePlay(tokenRef.current);

    try {
      ensureAudioGraph();
      if (audioCtxRef.current && audioCtxRef.current.state === 'suspended') {
        await audioCtxRef.current.resume();
      }
      await el.play();
      onPlay?.();
      startLoop();
    } catch {
      setErr('Playback was blocked or failed.');
    }
  }

  function pause() {
    const el = audioRef.current;
    if (!el) return;
    el.pause();
    onPause?.();
    stopLoop();
  }

  function togglePlay() {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) void safePlay();
    else pause();
  }

  function seekTo(sec: number) {
    const el = audioRef.current;
    if (!el || !isFinite(duration) || duration <= 0) return;
    el.currentTime = clamp(sec, 0, duration);
    setT(el.currentTime);
  }

  function restart() {
    seekTo(0);
  }

  // Scrubber pointer interactions
  function barValueFromClientX(clientX: number) {
    const bar = barRef.current;
    if (!bar || !duration) return 0;
    const r = bar.getBoundingClientRect();
    const x = clamp((clientX - r.left) / r.width, 0, 1);
    return x * duration;
  }
  function onBarPointerDown(e: React.PointerEvent) {
    if (!duration) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setDragging(true);
    setDragValue(barValueFromClientX(e.clientX));
  }
  function onBarPointerMove(e: React.PointerEvent) {
    if (!dragging || !duration) return;
    setDragValue(barValueFromClientX(e.clientX));
  }
  function onBarPointerUp(e: React.PointerEvent) {
    if (!duration) return;
    const v = barValueFromClientX(e.clientX);
    setDragging(false);
    setDragValue(0);
    seekTo(v);
  }

  // Sizing / layout
  const pad = clamp(Math.round(size * 0.06), 12, 36);
  const titleSize = clamp(Math.round(size * 0.05), 12, 22);
  const subSize = clamp(Math.round(size * 0.035), 10, 16);
  const timeSize = clamp(Math.round(size * 0.04), 11, 18);

  const barH = clamp(Math.round(size * 0.035), 10, 16);
  const thumb = clamp(Math.round(size * 0.065), 16, 30);

  const playBtn = clamp(Math.round(size * 0.3), 76, 260);
  const playIcon = Math.round(playBtn * 0.4);

  const restartBtn = clamp(Math.round(playBtn * 0.55), 42, 140);
  const restartIcon = Math.round(restartBtn * 0.5);

  // bottom UI block fixed so scrub never gets pushed out
  const bottomBlock = Math.round(timeSize * 1.3 + barH + pad * 0.9);

  const bgUrl = albumArtUrl || (defaultAlbumArtUrl ?? null);

  return (
    <div ref={wrapRef} className={`relative h-full w-full overflow-hidden bg-gray-900 ${className}`}>
      {bgUrl ? (
        <>
          <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${bgUrl})` }} />
          <div className="absolute inset-0 bg-gray-900/50" />
        </>
      ) : null}

      <div className="relative z-10 flex h-full w-full flex-col" style={{ padding: pad }}>
        {/* Header */}
        <div className="min-h-0">
          <div
            className="truncate text-gray-200"
            style={{ fontSize: titleSize, lineHeight: 1.1, letterSpacing: '0.01em' }}
          >
            {title}
          </div>
          {subtitle ? (
            <div className="mt-1 truncate text-gray-400" style={{ fontSize: subSize, lineHeight: 1.15 }}>
              {subtitle}
            </div>
          ) : null}
          {err ? (
            <div className="mt-2 text-gray-300" style={{ fontSize: subSize }}>
              {err}
            </div>
          ) : !isReady ? (
            <div className="mt-2 text-gray-500" style={{ fontSize: subSize }}>
              Loading…
            </div>
          ) : null}
        </div>

        {/* Waveform + controls */}
        <div className="mt-3 flex-1 min-h-0">
          <div
            className="group relative w-full overflow-hidden rounded-lg border border-2 border-yellow-400 bg-gray-900/80"
            style={{ height: `calc(100% - ${bottomBlock}px)` }}
          >
            {/* semi-transparent background so album art shows through */}
            <canvas ref={waveRef} className="h-full w-full" />

            <div className="absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-3">
              <button
                onClick={restart}
                className={[
                  'rounded-full border border-gray-700 bg-gray-950/80 text-gray-200',
                  'transition group-hover:border-gray-600 group-hover:bg-gray-950/90',
                  'focus:outline-none focus:ring-2 focus:ring-gray-500/40',
                ].join(' ')}
                style={{ width: restartBtn, height: restartBtn }}
                aria-label="Restart"
                title="Restart"
              >
                <svg width={restartIcon} height={restartIcon} viewBox="0 0 24 24" className="mx-auto" aria-hidden>
                  <path d="M12 5a7 7 0 1 1-6.4 4H3l3.5-3.5L10 9H7.8A5 5 0 1 0 12 7v-2z" fill="currentColor" />
                </svg>
              </button>

              <button
                onClick={togglePlay}
                className={[
                  'rounded-full border border-gray-700 bg-gray-950/80 text-gray-200',
                  'transition group-hover:border-gray-600 group-hover:bg-gray-950/90',
                  'focus:outline-none focus:ring-2 focus:ring-gray-500/40',
                ].join(' ')}
                style={{ width: playBtn, height: playBtn }}
                aria-label={isPlaying ? 'Pause' : 'Play'}
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {!isPlaying ? (
                  <svg width={playIcon} height={playIcon} viewBox="0 0 24 24" className="mx-auto" aria-hidden>
                    <path d="M8.5 5.5v13l11-6.5-11-6.5z" fill="currentColor" />
                  </svg>
                ) : (
                  <svg width={playIcon} height={playIcon} viewBox="0 0 24 24" className="mx-auto" aria-hidden>
                    <path d="M7 6h3v12H7zM14 6h3v12h-3z" fill="currentColor" />
                  </svg>
                )}

                {isBuffering ? (
                  <div className="absolute -bottom-7 left-1/2 -translate-x-1/2 text-xs text-gray-300">Buffering…</div>
                ) : null}
              </button>
            </div>
          </div>

          {/* Bottom block: always visible */}
          <div className="mt-3">
            <div
              className="flex items-center justify-between tabular-nums text-gray-300"
              style={{ fontSize: timeSize }}
            >
              <div>{fmtTime(dragging ? dragValue : t)}</div>
              <div>{duration ? `-${fmtTime(remaining)}` : '0:00'}</div>
            </div>

            <div
              ref={barRef}
              className="mt-2 relative w-full cursor-pointer select-none rounded-full bg-gray-800"
              style={{ height: barH }}
              onPointerDown={onBarPointerDown}
              onPointerMove={onBarPointerMove}
              onPointerUp={onBarPointerUp}
              onPointerCancel={() => setDragging(false)}
              title="Scrub"
            >
              <div
                className="absolute left-0 top-0 h-full rounded-full bg-gray-600"
                style={{
                  width: `${progress * 100}%`,
                  transition: dragging ? 'none' : 'width 80ms linear',
                }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 rounded-full bg-gray-200"
                style={{
                  left: `calc(${progress * 100}% - ${Math.floor(thumb / 2)}px)`,
                  width: thumb,
                  height: thumb,
                  transform: `translateY(-50%) scale(${dragging ? 1.08 : 1})`,
                  transition: 'transform 120ms ease-out',
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <audio ref={audioRef} preload="metadata" />
    </div>
  );
}
