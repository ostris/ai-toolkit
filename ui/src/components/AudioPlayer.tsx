'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';

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
 * ID3 helpers (v2.2/v2.3/v2.4):
 * - robust album art extraction: APIC (v2.3/2.4) + PIC (v2.2)
 * - basic text frames: title/artist/album
 * - handles tag-level unsynchronisation
 *
 * Requires fetch() byte access; if CORS blocks, it will fall back gracefully.
 */
type Id3Meta = {
  title?: string;
  artist?: string;
  album?: string;
  pictureUrl?: string; // object URL
};

function synchsafeToInt(b0: number, b1: number, b2: number, b3: number) {
  return ((b0 & 0x7f) << 21) | ((b1 & 0x7f) << 14) | ((b2 & 0x7f) << 7) | (b3 & 0x7f);
}

function deUnsync(bytes: Uint8Array) {
  const out: number[] = [];
  for (let i = 0; i < bytes.length; i++) {
    const cur = bytes[i];
    out.push(cur);
    if (cur === 0xff && i + 1 < bytes.length && bytes[i + 1] === 0x00) i += 1;
  }
  return new Uint8Array(out);
}

function decodeText(encoding: number, bytes: Uint8Array) {
  let end = bytes.length;
  while (end > 0 && bytes[end - 1] === 0) end--;
  const b = bytes.slice(0, end);

  try {
    if (encoding === 0) return new TextDecoder('latin1').decode(b);
    if (encoding === 1) return new TextDecoder('utf-16').decode(b);
    if (encoding === 2) return new TextDecoder('utf-16be').decode(b);
    if (encoding === 3) return new TextDecoder('utf-8').decode(b);
  } catch {
    // ignore
  }
  return new TextDecoder('latin1').decode(b);
}

function readNullTerminated(bytes: Uint8Array, start: number, encoding: number) {
  if (encoding === 1 || encoding === 2) {
    let i = start;
    while (i + 1 < bytes.length && !(bytes[i] === 0 && bytes[i + 1] === 0)) i += 2;
    const textBytes = bytes.slice(start, i);
    return { text: decodeText(encoding, textBytes), next: i + 2 };
  } else {
    let i = start;
    while (i < bytes.length && bytes[i] !== 0) i++;
    const textBytes = bytes.slice(start, i);
    return { text: decodeText(encoding, textBytes), next: i + 1 };
  }
}

async function fetchBytes(src: string, start: number, endInclusive: number) {
  const wantLen = endInclusive - start + 1;

  try {
    const r = await fetch(src, { headers: { Range: `bytes=${start}-${endInclusive}` } });
    if (!r.ok) throw new Error('range not ok');
    const buf = await r.arrayBuffer();
    return new Uint8Array(buf);
  } catch {
    const r = await fetch(src);
    if (!r.ok) throw new Error('fetch not ok');
    const buf = await r.arrayBuffer();
    const u8 = new Uint8Array(buf);
    if (start === 0 && u8.length >= wantLen) return u8.slice(0, wantLen);
    return u8.slice(start, Math.min(u8.length, endInclusive + 1));
  }
}

async function extractId3MetaAndArt(src: string, maxTagBytes = 4_000_000): Promise<Id3Meta> {
  const head = await fetchBytes(src, 0, 64 * 1024 - 1).catch(() => null);
  if (!head || head.length < 10) return {};
  if (head[0] !== 0x49 || head[1] !== 0x44 || head[2] !== 0x33) return {};

  const verMajor = head[3]; // 2,3,4
  const flags = head[5];
  const tagSize = synchsafeToInt(head[6], head[7], head[8], head[9]);
  const tagEnd = 10 + tagSize;

  const need = Math.min(tagEnd, maxTagBytes);
  let tagBytes = head;
  if (head.length < need) {
    const more = await fetchBytes(src, 0, need - 1).catch(() => null);
    if (!more) return {};
    tagBytes = more;
  } else {
    tagBytes = head.slice(0, need);
  }

  const tagUnsync = (flags & 0x80) !== 0;
  const tagDataRaw = tagBytes.slice(10, Math.min(tagBytes.length, tagEnd));
  const tagData = tagUnsync ? deUnsync(tagDataRaw) : tagDataRaw;

  let offset = 0;

  // Extended header (v2.3/v2.4)
  if (verMajor === 3 || verMajor === 4) {
    const hasExt = (flags & 0x40) !== 0;
    if (hasExt && tagData.length >= 4) {
      let extSize = 0;
      if (verMajor === 4) extSize = synchsafeToInt(tagData[0], tagData[1], tagData[2], tagData[3]);
      else extSize = (tagData[0] << 24) | (tagData[1] << 16) | (tagData[2] << 8) | tagData[3];
      offset += 4 + Math.max(0, extSize);
    }
  }

  const meta: Id3Meta = {};
  const setIfEmpty = (k: keyof Id3Meta, v?: string) => {
    if (!v) return;
    if (!meta[k]) meta[k] = v;
  };

  while (offset < tagData.length) {
    if (tagData[offset] === 0x00) break;

    if (verMajor === 2) {
      if (offset + 6 > tagData.length) break;
      const id = new TextDecoder('latin1').decode(tagData.slice(offset, offset + 3));
      const size = (tagData[offset + 3] << 16) | (tagData[offset + 4] << 8) | tagData[offset + 5];
      offset += 6;
      if (!id.trim() || size <= 0 || offset + size > tagData.length) break;

      const frame = tagData.slice(offset, offset + size);

      if (id === 'TT2' || id === 'TP1' || id === 'TAL') {
        const enc = frame[0];
        const txt = decodeText(enc, frame.slice(1));
        if (id === 'TT2') setIfEmpty('title', txt);
        if (id === 'TP1') setIfEmpty('artist', txt);
        if (id === 'TAL') setIfEmpty('album', txt);
      }

      if (id === 'PIC' && frame.length > 6) {
        const enc = frame[0];
        const fmt = new TextDecoder('latin1').decode(frame.slice(1, 4)).toLowerCase();
        const imgType = fmt === 'png' ? 'image/png' : 'image/jpeg';
        let p = 5;
        const desc = readNullTerminated(frame, p, enc);
        p = desc.next;
        if (p < frame.length) {
          const img = frame.slice(p);
          if (img.length > 64) {
            const blob = new Blob([img], { type: imgType });
            meta.pictureUrl = URL.createObjectURL(blob);
          }
        }
      }

      offset += size;
    } else {
      if (offset + 10 > tagData.length) break;

      const id = new TextDecoder('latin1').decode(tagData.slice(offset, offset + 4));
      let size = 0;
      if (verMajor === 4)
        size = synchsafeToInt(tagData[offset + 4], tagData[offset + 5], tagData[offset + 6], tagData[offset + 7]);
      else
        size =
          (tagData[offset + 4] << 24) | (tagData[offset + 5] << 16) | (tagData[offset + 6] << 8) | tagData[offset + 7];

      const flag2 = tagData[offset + 9];
      offset += 10;

      if (!id.trim() || size <= 0 || offset + size > tagData.length) break;

      let frame = tagData.slice(offset, offset + size);

      const frameUnsync = verMajor === 4 && (flag2 & 0x02) !== 0;
      if (frameUnsync) frame = deUnsync(frame);

      if (id === 'TIT2' || id === 'TPE1' || id === 'TALB') {
        const enc = frame[0];
        const txt = decodeText(enc, frame.slice(1));
        if (id === 'TIT2') setIfEmpty('title', txt);
        if (id === 'TPE1') setIfEmpty('artist', txt);
        if (id === 'TALB') setIfEmpty('album', txt);
      }

      if (id === 'APIC' && frame.length > 10) {
        const enc = frame[0];
        const mimeZ = readNullTerminated(frame, 1, 0);
        const mime = mimeZ.text || 'image/jpeg';
        let p = mimeZ.next;
        if (p < frame.length) p += 1; // pictureType
        const desc = readNullTerminated(frame, p, enc);
        p = desc.next;

        if (p < frame.length) {
          const img = frame.slice(p);
          if (img.length > 64) {
            const blob = new Blob([img], { type: mime });
            if (!meta.pictureUrl) meta.pictureUrl = URL.createObjectURL(blob);
          }
        }
      }

      offset += size;
    }
  }

  return meta;
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

  // Meta + artwork
  const [metaTitle, setMetaTitle] = useState<string | null>(null);
  const [metaArtist, setMetaArtist] = useState<string | null>(null);
  const [metaAlbum, setMetaAlbum] = useState<string | null>(null);
  const [albumArtUrl, setAlbumArtUrl] = useState<string | null>(null);
  const albumArtBlobUrlRef = useRef<string | null>(null);

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

  // Extract meta + art
  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (albumArtBlobUrlRef.current) {
        URL.revokeObjectURL(albumArtBlobUrlRef.current);
        albumArtBlobUrlRef.current = null;
      }
      setAlbumArtUrl(null);
      setMetaTitle(null);
      setMetaArtist(null);
      setMetaAlbum(null);

      try {
        const meta = await extractId3MetaAndArt(src);
        if (cancelled) return;

        if (meta.title) setMetaTitle(meta.title);
        if (meta.artist) setMetaArtist(meta.artist);
        if (meta.album) setMetaAlbum(meta.album);

        if (meta.pictureUrl) {
          albumArtBlobUrlRef.current = meta.pictureUrl;
          setAlbumArtUrl(meta.pictureUrl);
        } else if (defaultAlbumArtUrl) {
          setAlbumArtUrl(defaultAlbumArtUrl);
        }
      } catch {
        if (!cancelled && defaultAlbumArtUrl) setAlbumArtUrl(defaultAlbumArtUrl);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [src, defaultAlbumArtUrl]);

  // Cleanup artwork blob URL on unmount
  useEffect(() => {
    return () => {
      if (albumArtBlobUrlRef.current) {
        URL.revokeObjectURL(albumArtBlobUrlRef.current);
        albumArtBlobUrlRef.current = null;
      }
    };
  }, []);

  const effectiveTitle = metaTitle || title;
  const effectiveSubtitle = metaArtist || subtitle || metaAlbum || '';

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
            {effectiveTitle}
          </div>
          {effectiveSubtitle ? (
            <div className="mt-1 truncate text-gray-400" style={{ fontSize: subSize, lineHeight: 1.15 }}>
              {effectiveSubtitle}
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
