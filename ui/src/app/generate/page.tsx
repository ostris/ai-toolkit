'use client';

import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { Button } from '@headlessui/react';
import { ChevronDown, ChevronLeft, ChevronRight, Loader2, OctagonX, Play, Plus, Square, Sparkles, Trash2, X } from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import { Checkbox, CreatableSelectInput, NumberInput, SelectInput, SliderInput, TextAreaInput, TextInput } from '@/components/formInputs';
import { apiClient } from '@/utils/api';
import { startJob, stopJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import useGPUInfo from '@/hooks/useGPUInfo';
import usePollLoop from '@/hooks/usePollLoop';
import { defaultInferenceJobConfig } from '@/helpers/inferenceJobConfig';
import { encodeFilePathForUrl } from '@/utils/basic';
import { latentToImage, payloadToFloat32, readEngineFrames, PreviewInfo } from '@/lib/engineStream';
import { isMac } from '@/helpers/basic';
import { modelArchs, getGenerateDefaults, GenerateDefaults } from '@/app/jobs/new/options';
import GenerateFooter from '@/components/generate/GenerateFooter';
import LoraBrowserModal, { LoraPick } from '@/components/generate/LoraBrowserModal';


interface EngineStatus {
  running: boolean;
  engine: { jobId: string; jobName: string; gpuIds: string; status: string; ready: boolean } | null;
  engines: { jobId: string; jobName: string; gpuIds: string; status: string; ready: boolean }[];
}

interface ResultItem {
  request_id: string;
  path: string;
  seconds?: number;
  steps?: number;
  kind: string;
  ext: string;
  seed: number;
  width: number;
  height: number;
  prompt: string;
  arch: string;
}

const qtypeOptions = [
  { value: 'convrot8', label: 'ConvRot 8-bit' },
  { value: 'float8', label: 'Float8' },
  { value: 'qfloat8', label: 'Quanto float8' },
  { value: 'nvfp4', label: 'NVFP4' },
];

// everything the user set on the page survives a reload
const STORAGE_KEY = 'aitk_generate_page';
const ACTIVE_REQUEST_KEY = 'aitk_generate_active_request';
interface PersistedState {
  arch?: string;
  model?: { [key: string]: any };
  sample?: { [key: string]: any };
  gpuId?: string;
  results?: ResultItem[];
  sidebarOpen?: boolean;
  cards?: { [key: string]: boolean };
}
const loadPersisted = (): PersistedState => {
  if (typeof window === 'undefined') return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PersistedState) : {};
  } catch {
    return {};
  }
};

const authHeaders = (): Record<string, string> => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
};

function Card({
  title,
  subtitle,
  open,
  onToggle,
  children,
}: {
  title: string;
  subtitle?: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800">
      <button type="button" onClick={onToggle} className="w-full flex items-center justify-between gap-2 px-3 py-1.5 text-xs text-gray-200 font-semibold">
        <span className="flex items-baseline gap-2 min-w-0">
          <span>{title}</span>
          {subtitle && <span className="text-[11px] font-normal text-gray-500 truncate">{subtitle}</span>}
        </span>
        {open ? <ChevronDown className="w-4 h-4 text-gray-400" /> : <ChevronLeft className="w-4 h-4 text-gray-400" />}
      </button>
      <div className={open ? 'px-3 pb-3' : 'hidden'}>{children}</div>
    </div>
  );
}

function GeneratePageInner() {
  const searchParams = useSearchParams();
  const preferredJob = searchParams.get('job');
  const { gpuList } = useGPUInfo();

  // ---- engine lifecycle ----
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);
  const [health, setHealth] = useState<any>(null);
  const persisted = useMemo(loadPersisted, []);
  const [gpuId, setGpuId] = useState<string>(persisted.gpuId || '0');
  const [engineBusy, setEngineBusy] = useState(false);
  // job ids we are waiting on: the button stays disabled from the click until
  // the engine is actually up (or the job row reports a failure)
  const [startingJobId, setStartingJobId] = useState<string | null>(null);
  const [stoppingJobId, setStoppingJobId] = useState<string | null>(null);
  const [stoppingSince, setStoppingSince] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const engineJobId = preferredJob || engineStatus?.engine?.jobId || null;
  const proxy = useCallback((path: string) => `/api/inference/${path}${engineJobId ? `${path.includes('?') ? '&' : '?'}job=${engineJobId}` : ''}`, [engineJobId]);

  usePollLoop(
    () =>
      apiClient
        .get(`/api/inference/status${preferredJob ? `?job=${preferredJob}` : ''}`)
        .then(res => res.data)
        .then(async (data: EngineStatus) => {
          setEngineStatus(data);
          if (startingJobId) {
            const mine = data.engines.find(e => e.jobId === startingJobId);
            if (mine?.ready) {
              setStartingJobId(null);
            } else if (!mine) {
              // no longer queued/running: the launch failed (or was stopped)
              const job = await apiClient.get('/api/jobs', { params: { id: startingJobId } }).then(r => r.data).catch(() => null);
              if (!job || ['error', 'stopped', 'completed'].includes(job.status)) {
                setStartingJobId(null);
                alert(`The engine did not start${job?.info ? `: ${job.info}` : ''}. Check the job log in the Queue.`);
              }
            }
          }
          if (stoppingJobId && !data.engines.some(e => e.jobId === stoppingJobId)) {
            setStoppingJobId(null);
            setStoppingSince(null);
          }
          if (data.running) {
            return apiClient
              .get(proxy('health'))
              .then(r => setHealth(r.data))
              .catch(() => setHealth(null));
          }
          setHealth(null);
        })
        .catch(() => {}),
    2000,
    [preferredJob, proxy, startingJobId, stoppingJobId],
  );

  useEffect(() => {
    if (gpuList.length && !gpuList.some(g => `${g.index}` === gpuId)) setGpuId(`${gpuList[0].index}`);
  }, [gpuList, gpuId]);

  const startEngine = async () => {
    setEngineBusy(true);
    try {
      const gpu = isMac() ? 'mps' : gpuId;
      const name = `inference_engine_gpu${gpu}`;
      const existing = await apiClient.get('/api/jobs', { params: { job_type: 'inference' } }).then(r => r.data.jobs || []);
      const match = existing.find((j: any) => j.name === name);
      const res = await apiClient.post('/api/jobs', {
        id: match ? match.id : null,
        name,
        gpu_ids: gpu,
        job_config: { ...defaultInferenceJobConfig, config: { ...defaultInferenceJobConfig.config, name } },
        job_type: 'inference',
      });
      await startJob(res.data.id);
      await startQueue(gpu);
      setStartingJobId(res.data.id);
    } catch (e: any) {
      alert(`Failed to start the engine: ${e?.response?.data?.error || e?.message || e}`);
    } finally {
      setEngineBusy(false);
    }
  };

  // Stop = graceful stop, then a hard kill if the process is still around a
  // few seconds later (a model load in progress can hold the shutdown up)
  const stopEngine = async () => {
    const eng = engineStatus?.engine;
    if (!eng) return;
    const id = eng.jobId;
    setEngineBusy(true);
    setStoppingJobId(id);
    setStoppingSince(Date.now());
    try {
      await stopJob(id).catch(e => console.warn('graceful stop failed, will kill', e?.message || e));
      const deadline = Date.now() + 4000;
      let alive = true;
      while (Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 750));
        const st = await apiClient.get(`/api/inference/status?job=${id}`).then(r => r.data).catch(() => null);
        alive = !!st?.engines?.some((e: any) => e.jobId === id && (e.ready || e.status === 'stopping'));
        if (!alive) break;
      }
      if (alive) {
        await apiClient.get(`/api/jobs/${id}/kill`);
      }
    } catch (e: any) {
      alert(`Failed to stop the engine: ${e?.response?.data?.error || e?.message || e}`);
    } finally {
      setEngineBusy(false);
    }
  };

  // a row that is stopped but whose process is still alive shows as 'stopping' from the status route
  const isStopping = !!stoppingJobId || engineStatus?.engine?.status === 'stopping';
  const isStarting = !isStopping && (engineBusy || !!startingJobId || (!!engineStatus?.engine && !engineStatus.running));
  const stoppingForMs = stoppingSince ? now - stoppingSince : isStopping ? 999_999 : 0;
  // cancel a start that is not getting anywhere: a queued job (no process)
  // is just marked stopped; a launched one gets the normal stop
  const [startingSince, setStartingSince] = useState<number | null>(null);
  useEffect(() => {
    if (isStarting && startingSince === null) setStartingSince(Date.now());
    if (!isStarting && startingSince !== null) setStartingSince(null);
  }, [isStarting, startingSince]);
  const cancelStart = async () => {
    const eng = engineStatus?.engine;
    const id = startingJobId || eng?.jobId;
    if (!id) return;
    try {
      if (!eng || eng.status === 'queued') {
        await apiClient.get(`/api/jobs/${id}/mark_stopped`);
      } else {
        await stopJob(id);
      }
    } catch (e: any) {
      alert(`Cancel failed: ${e?.response?.data?.error || e?.message || e}`);
    }
    setStartingJobId(null);
    setStartingSince(null);
  };
  const forceStop = async () => {
    const id = stoppingJobId || engineStatus?.engine?.jobId;
    if (!id) return;
    if (!confirm('Force stop kills the engine process immediately. Continue?')) return;
    try {
      await apiClient.get(`/api/jobs/${id}/kill`);
    } catch (e: any) {
      alert(`Force stop failed: ${e?.response?.data?.error || e?.message || e}`);
    }
  };

  // ---- models: the same arch list the training UI uses (jobs/new/options.tsx) ----
  const archs: GenerateDefaults[] = useMemo(() => modelArchs.map(getGenerateDefaults), []);
  const ready = !!engineStatus?.running;

  const [arch, setArch] = useState<string>(persisted.arch || 'zimage');
  const [model, setModel] = useState<{ [key: string]: any }>(persisted.model || {});
  const [sample, setSample] = useState<{ [key: string]: any }>(persisted.sample || { prompt: '', negative_prompt: '', seed: -1 });
  const entry = useMemo(() => archs.find(a => a.arch === arch) || null, [archs, arch]);

  const applyArch = (name: string) => {
    setArch(name);
    const e = archs.find(a => a.arch === name);
    if (!e) return;
    setModel({ ...e.model });
    setSample(s => ({ ...s, ...e.sample, prompt: s.prompt, negative_prompt: s.negative_prompt, seed: s.seed ?? -1, ctrl_img: undefined }));
  };
  useEffect(() => {
    if (archs.length && !Object.keys(model).length) applyArch(archs.some(a => a.arch === arch) ? arch : archs[0].arch);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [archs]);

  const archOptions = useMemo(() => {
    const groups: { [k: string]: { value: string; label: string }[] } = {};
    for (const a of archs) {
      (groups[a.group] ||= []).push({ value: a.arch, label: a.label });
    }
    return Object.entries(groups).map(([label, options]) => ({ label, options }));
  }, [archs]);

  // ---- generation ----
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<{ step: number; total: number | null; elapsed: number } | null>(null);
  const [statusLine, setStatusLine] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultItem[]>(persisted.results || []);
  const [selected, setSelected] = useState<ResultItem | null>(persisted.results?.[0] || null);
  // the stage shows the live latent canvas while generating; a finished (or
  // clicked) result replaces it
  const [showPreview, setShowPreview] = useState<boolean>(!persisted.results?.length);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(persisted.sidebarOpen ?? true);
  const [cards, setCards] = useState<{ [key: string]: boolean }>(persisted.cards || {});
  const [loraModalOpen, setLoraModalOpen] = useState(false);
  const loras: { path: string; name: string; strength: number }[] = model.loras || [];
  const setLoras = (next: { path: string; name: string; strength: number }[]) => setModel(m => ({ ...m, loras: next }));
  const addLora = (pick: LoraPick) => {
    if (loras.some(l => l.path === pick.path)) return;
    setLoras([...loras, { path: pick.path, name: pick.name, strength: 1.0 }]);
  };
  const cardOpen = (key: string) => cards[key] ?? true;
  const toggleCard = (key: string) => setCards(c => ({ ...c, [key]: !(c[key] ?? true) }));
  const resultUrl = (r: ResultItem) => `/api/files/${encodeFilePathForUrl(r.path)}`;

  const deleteResults = async (items: ResultItem[]) => {
    const paths = items.map(i => i.path);
    try {
      await apiClient.post('/api/inference/outputs/delete', { paths });
    } catch (e: any) {
      alert(`Failed to delete: ${e?.response?.data?.error || e?.message || e}`);
      return;
    }
    setResults(prev => {
      const next = prev.filter(r => !paths.includes(r.path));
      if (selected && paths.includes(selected.path)) {
        const replacement = next[0] || null;
        setSelected(replacement);
        if (!replacement) setShowPreview(true);
      }
      return next;
    });
  };

  const deleteAll = () => {
    if (!results.length) return;
    openConfirm({
      title: 'Delete all results',
      message: `Delete all ${results.length} generated files from disk? This cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete all',
      onConfirm: () => deleteResults(results),
    });
  };

  // persist on every change (small payload; results capped)
  useEffect(() => {
    try {
      const state: PersistedState = { arch, model, sample, gpuId, results: results.slice(0, 60), sidebarOpen, cards };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // storage full or unavailable: nothing to do
    }
  }, [arch, model, sample, gpuId, results, sidebarOpen, cards]);
  const [frameIdx, setFrameIdx] = useState(0);
  const [frameCount, setFrameCount] = useState(1);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const framesRef = useRef<ImageData[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef<string | null>(null);

  // the canvas is sized to the requested image dimensions (from the engine's
  // start frame) so the preview occupies exactly the space the image will
  const previewSizeRef = useRef<{ width: number; height: number } | null>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const drawFrame = (idx: number) => {
    const canvas = canvasRef.current;
    const frames = framesRef.current;
    if (!canvas || !frames.length) return;
    const img = frames[Math.min(idx, frames.length - 1)];
    const target = previewSizeRef.current;
    const w = target?.width || img.width;
    const h = target?.height || img.height;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    if (!scratchRef.current) scratchRef.current = document.createElement('canvas');
    const scratch = scratchRef.current;
    if (scratch.width !== img.width || scratch.height !== img.height) {
      scratch.width = img.width;
      scratch.height = img.height;
    }
    scratch.getContext('2d')?.putImageData(img, 0, 0);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(scratch, 0, 0, w, h);
    // the live tile in the history strip mirrors the preview
    const thumb = thumbRef.current;
    if (thumb) {
      const size = 64;
      const scaleT = Math.min(size / w, size / h);
      const tw = Math.max(1, Math.round(w * scaleT));
      const th = Math.max(1, Math.round(h * scaleT));
      if (thumb.width !== tw || thumb.height !== th) {
        thumb.width = tw;
        thumb.height = th;
      }
      const tctx = thumb.getContext('2d');
      if (tctx) {
        tctx.imageSmoothingEnabled = true;
        tctx.drawImage(scratch, 0, 0, tw, th);
      }
    }
  };
  const thumbRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => drawFrame(frameIdx), [frameIdx]);

  // temporal latents auto-play: T latent frames span num_frames / fps
  // seconds, so each latent frame shows for that fraction of the clip
  const clipRef = useRef<{ num_frames: number; fps: number } | null>(null);
  useEffect(() => {
    if (frameCount <= 1 || !showPreview) return;
    const clip = clipRef.current;
    const seconds = clip && clip.num_frames > 1 && clip.fps > 0 ? clip.num_frames / clip.fps : frameCount / 4;
    const msPerFrame = Math.max(30, (seconds * 1000) / frameCount);
    const t = setInterval(() => setFrameIdx(i => (i + 1) % frameCount), msPerFrame);
    return () => clearInterval(t);
  }, [frameCount, showPreview]);

  // one consumer for both a fresh /generate response and a /stream/{id}
  // reattach: everything the stage needs comes from the frames themselves
  const consumeStream = async (res: Response, abort: AbortController) => {
    let preview: PreviewInfo | null = null;
    let startInfo: { arch: string; prompt: string } = { arch, prompt: sample.prompt || '' };
    // only the first latent of a run pulls the stage to the preview; after
    // that the user may browse history and come back via the live tile
    let shownPreview = false;
    await readEngineFrames(
      res,
      ({ header, payload }) => {
        requestIdRef.current = header.request_id;
        switch (header.type) {
          case 'start':
            preview = header.preview;
            startInfo = { arch: header.model?.arch || arch, prompt: header.sample?.prompt ?? '' };
            if (header.sample?.width && header.sample?.height) {
              previewSizeRef.current = { width: header.sample.width, height: header.sample.height };
            }
            clipRef.current = { num_frames: header.sample?.num_frames ?? 1, fps: header.sample?.fps ?? 16 };
            setStatusLine(`Generating with ${startInfo.arch}`);
            break;
          case 'status':
            setStatusLine(header.message);
            break;
          case 'progress':
            setProgress({ step: header.step, total: header.total, elapsed: header.elapsed });
            break;
          case 'latent': {
            try {
              if (header.preview) preview = header.preview;
              const data = payloadToFloat32(header, payload);
              const img = latentToImage(header, data, preview);
              if (img) {
                framesRef.current = img.frameData.map(px => new ImageData(px, img.width, img.height));
                setFrameCount(img.frames);
                drawFrame(frameIdx);
                if (!shownPreview) {
                  shownPreview = true;
                  setShowPreview(true);
                }
              }
            } catch (e) {
              console.warn('latent preview failed', e);
            }
            break;
          }
          case 'result': {
            const item: ResultItem = {
              request_id: header.request_id,
              path: header.path,
              kind: header.kind,
              ext: header.ext,
              seed: header.seed,
              width: header.width,
              height: header.height,
              seconds: header.seconds,
              steps: header.steps,
              prompt: startInfo.prompt,
              arch: startInfo.arch,
            };
            setResults(r => (r.some(x => x.path === item.path) ? r : [item, ...r]));
            setSelected(item);
            setShowPreview(false);
            break;
          }
          case 'error':
            setError(header.cancelled ? 'Cancelled' : header.message);
            break;
          case 'end':
            setStatusLine(header.status === 'done' ? 'Done' : header.status);
            break;
        }
      },
      abort.signal,
    );
  };

  const beginRun = () => {
    setRunning(true);
    setError(null);
    setProgress(null);
    framesRef.current = [];
    setFrameCount(1);
    setFrameIdx(0);
    // keep whatever is on the stage until the first latent of this run
    // arrives; wipe the stale bitmap so an empty canvas never shows old data
    const cv = canvasRef.current;
    if (cv) cv.getContext('2d')?.clearRect(0, 0, cv.width, cv.height);
    const abort = new AbortController();
    abortRef.current = abort;
    return abort;
  };

  const endRun = () => {
    setRunning(false);
    abortRef.current = null;
    requestIdRef.current = null;
    try {
      localStorage.removeItem(ACTIVE_REQUEST_KEY);
    } catch {
      // storage unavailable
    }
  };

  const generate = async () => {
    if (!ready || running) return;
    const abort = beginRun();
    setStatusLine('Submitting');
    previewSizeRef.current = sample.width && sample.height ? { width: sample.width, height: sample.height } : null;
    const body: { model: any; sample: any; stream: any } = {
      model: { ...model, arch },
      sample: { ...sample, seed: sample.seed === '' ? -1 : sample.seed },
      stream: { latents: 'raw', every_n_steps: 1, max_frames: 0 },
    };
    try {
      const res = await fetch(proxy('generate'), {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abort.signal,
      });
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      // remembered so a reload can reattach to this run
      const id = res.headers.get('x-request-id');
      if (id) {
        try {
          localStorage.setItem(ACTIVE_REQUEST_KEY, id);
        } catch {
          // storage unavailable
        }
      }
      await consumeStream(res, abort);
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message || String(e));
    } finally {
      endRun();
    }
  };

  // attach to a request already running on the engine (after a reload, or a
  // run started from another tab): replays its frames, then follows live
  const attachedRef = useRef<Set<string>>(new Set());
  const attachTo = async (id: string) => {
    if (running || attachedRef.current.has(id)) return;
    attachedRef.current.add(id);
    const abort = beginRun();
    requestIdRef.current = id;
    setStatusLine('Reattaching…');
    try {
      const res = await fetch(proxy(`stream/${id}`), { headers: authHeaders(), signal: abort.signal });
      if (!res.ok) throw new Error(res.status === 404 ? 'previous generation is gone' : `HTTP ${res.status}`);
      await consumeStream(res, abort);
    } catch (e: any) {
      if (e?.name !== 'AbortError') setError(e?.message || String(e));
    } finally {
      endRun();
    }
  };
  const attachRef = useRef(attachTo);
  attachRef.current = attachTo;

  // 1) the id we stored when we started a run (covers a job still queued)
  useEffect(() => {
    if (!ready || running) return;
    let id: string | null = null;
    try {
      id = localStorage.getItem(ACTIVE_REQUEST_KEY);
    } catch {
      id = null;
    }
    if (id) attachRef.current(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // 2) whatever the engine says it is generating right now
  const busyRequestId: string | null = health?.current?.request_id || null;
  useEffect(() => {
    if (ready && !running && busyRequestId) attachRef.current(busyRequestId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, running, busyRequestId]);

  // engine busy with a run we could not attach to (e.g. an engine older than
  // /stream): still let the user cancel it
  const cancelBusy = async () => {
    if (!busyRequestId) return;
    try {
      await apiClient.post(proxy(`cancel/${busyRequestId}`));
    } catch (e: any) {
      alert(`Cancel failed: ${e?.response?.data?.error || e?.message || e}`);
    }
  };

  // ctrl/cmd+enter anywhere on the page generates (the prompt textarea included)
  const generateRef = useRef(generate);
  generateRef.current = generate;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && !e.repeat) {
        e.preventDefault();
        generateRef.current();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const cancel = async () => {
    const id = requestIdRef.current;
    if (id) {
      try {
        await apiClient.post(proxy(`cancel/${id}`));
      } catch {
        // engine gone; abort below still tears the stream down
      }
    }
    abortRef.current?.abort();
  };

  const modality = entry?.modality || 'image';
  const footerStatus = running
    ? statusLine || 'Generating'
    : error
      ? error
      : ready
        ? health?.current
          ? `Generating ${health.current.arch}`
          : health?.active?.arch
            ? `Idle - ${health.active.arch} (${health.active.model?.name_or_path || ''})`
            : 'Idle - no model loaded'
        : engineStatus?.engine
          ? `Engine ${engineStatus.engine.status}`
          : 'Engine off';
  const vram = health?.vram ? `${(health.vram.used / 1e9).toFixed(1)} / ${(health.vram.total / 1e9).toFixed(0)} GB` : null;
  const pool = health?.pool;

  return (
    <>
      <TopBar>
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-blue-400" />
          <h1 className="text-base sm:text-lg">Generate</h1>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2 text-xs sm:text-sm">
          {isStopping ? (
            <>
              <Button disabled className="px-2 py-1 rounded-md bg-red-900 text-white/70 flex items-center gap-1 cursor-wait">
                <Loader2 className="w-4 h-4 animate-spin" /> Stopping engine…
              </Button>
              {stoppingForMs > 8000 && (
                <Button onClick={forceStop} className="px-2 py-1 rounded-md bg-red-700 hover:bg-red-600 text-white flex items-center gap-1" title="Kill the engine process">
                  <OctagonX className="w-4 h-4" /> Force stop
                </Button>
              )}
            </>
          ) : ready ? (
            <>
              <span className="text-green-400">Engine running</span>
              {health?.active?.arch && <span className="text-gray-400 hidden sm:inline">· {health.active.arch}</span>}
              {vram && <span className="text-gray-400 hidden md:inline">· VRAM {vram}</span>}
              <Button onClick={stopEngine} disabled={engineBusy} className="ml-2 px-2 py-1 rounded-md bg-red-700 hover:bg-red-600 disabled:opacity-60 text-white flex items-center gap-1">
                {engineBusy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Square className="w-4 h-4" />} Stop
              </Button>
            </>
          ) : null}
        </div>
      </TopBar>
      <MainContent belowTopBar className="overflow-hidden">
        {!ready && (
          <div className="absolute inset-0 z-20 flex items-center justify-center p-4">
            <div className="bg-gray-900/95 backdrop-blur border border-gray-700 rounded-2xl shadow-2xl px-8 py-8 max-w-md w-full text-center">
              <div className="mx-auto w-16 h-16 rounded-full bg-blue-500/10 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-blue-400" />
              </div>
              <h2 className="text-lg text-gray-100 font-semibold mb-2">Inference Engine</h2>
              <p className="text-sm text-gray-400 mb-6">
                {isStarting
                  ? 'The engine is starting. Model options appear once its server is up.'
                  : 'Start the inference engine on a GPU to generate images, video, and audio with any model.'}
              </p>
              {isStarting ? (
                <div className="flex flex-col gap-2">
                  <Button disabled className="w-full px-3 py-2 rounded-md bg-blue-900 text-white/70 flex items-center justify-center gap-2 cursor-wait">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {engineStatus?.engine?.status === 'running' ? 'Engine loading…' : engineStatus?.engine ? `Engine ${engineStatus.engine.status}…` : 'Starting engine…'}
                  </Button>
                  {engineStatus?.engine?.status === 'queued' && (
                    <div className="text-[11px] text-gray-500">Waiting for the GPU queue to pick the job up. If the queue is stopped, start it from the Queue page or cancel here.</div>
                  )}
                  {startingSince !== null && now - startingSince > 5000 && (
                    <Button onClick={cancelStart} className="w-full px-3 py-1.5 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-200 text-sm">
                      Cancel
                    </Button>
                  )}
                </div>
              ) : (
                <div className="flex flex-col gap-3">
                  {!isMac() && (
                    <select value={gpuId} onChange={e => setGpuId(e.target.value)} className="w-full bg-gray-800 text-gray-200 rounded-md px-3 py-2 border border-gray-700">
                      {gpuList.map(g => (
                        <option key={g.index} value={`${g.index}`}>
                          GPU #{g.index} {g.name}
                        </option>
                      ))}
                    </select>
                  )}
                  <Button onClick={startEngine} className="w-full px-3 py-2 rounded-md bg-blue-700 hover:bg-blue-600 text-white flex items-center justify-center gap-2">
                    <Play className="w-4 h-4" /> Start engine
                  </Button>
                </div>
              )}
            </div>
          </div>
        )}
        <div className={`absolute inset-0 pb-9 flex ${ready ? '' : 'opacity-30 pointer-events-none select-none'}`}>
          {/* ---- center: stage + history strip ---- */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex-1 min-h-0 relative bg-black/40 m-2 mb-0 rounded-xl border border-gray-800 overflow-hidden">
              {/* status strip over the stage */}
              <div className="absolute top-0 left-0 right-0 z-10 px-3 py-1.5 flex items-center gap-2 text-xs text-gray-300 bg-gradient-to-b from-gray-950/80 to-transparent">
                {running && <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />}
                <span className="truncate">
                  {running
                    ? statusLine
                    : selected
                      ? `${selected.arch} · seed ${selected.seed}${selected.seconds ? ` · ${selected.seconds.toFixed(1)}s${selected.steps ? ` / ${selected.steps} steps` : ''}` : ''}`
                      : ''}
                </span>
                {progress ? (
                  <span className="text-gray-400 shrink-0">
                    step {progress.step}
                    {progress.total ? `/${progress.total}` : ''} · {progress.elapsed.toFixed(1)}s
                  </span>
                ) : null}
                <div className="flex-1" />
                {running && (
                  <Button onClick={cancel} className="px-2 py-0.5 rounded bg-gray-800/80 hover:bg-gray-700 text-gray-200 text-xs">
                    Cancel
                  </Button>
                )}
                {!running && busyRequestId && (
                  <>
                    <span className="text-amber-300 truncate">
                      Engine busy: {health.current.arch}
                      {health.current.total_steps ? ` ${health.current.step}/${health.current.total_steps}` : ''}
                    </span>
                    <Button onClick={cancelBusy} className="px-2 py-0.5 rounded bg-gray-800/80 hover:bg-gray-700 text-gray-200 text-xs">
                      Cancel
                    </Button>
                  </>
                )}
              </div>
              {progress?.total ? (
                <div className="absolute top-0 left-0 right-0 h-0.5 bg-gray-800 z-10">
                  <div className="h-0.5 bg-blue-500 transition-all" style={{ width: `${Math.min(100, (progress.step / progress.total) * 100)}%` }} />
                </div>
              ) : null}
              {error && <div className="absolute bottom-3 left-3 right-3 z-10 text-rose-300 text-sm bg-rose-950/70 rounded px-3 py-2 whitespace-pre-wrap">{error}</div>}

              {/* live latent preview: shown while generating (or when nothing is selected) */}
              <div className={`absolute inset-3 flex items-center justify-center ${showPreview ? '' : 'hidden'}`}>
                <canvas ref={canvasRef} className="max-w-full max-h-full rounded" style={{ imageRendering: 'auto' }} />
                {!running && framesRef.current.length === 0 && (
                  <div className="absolute text-gray-600 text-sm">Your generation will appear here</div>
                )}
              </div>

              {/* selected result replaces the preview */}
              {!showPreview && selected && (
                <div className="absolute inset-3 flex items-center justify-center">
                  {selected.kind === 'video' ? (
                    <video key={selected.path} src={resultUrl(selected)} controls autoPlay loop className="max-w-full max-h-full rounded" />
                  ) : selected.kind === 'audio' ? (
                    <div className="w-full max-w-xl bg-gray-900 rounded-xl p-6 text-center">
                      <div className="text-gray-300 text-sm mb-4 truncate" title={selected.prompt}>
                        {selected.prompt}
                      </div>
                      <audio key={selected.path} src={resultUrl(selected)} controls autoPlay className="w-full" />
                    </div>
                  ) : (
                    <a href={resultUrl(selected)} target="_blank" rel="noreferrer" className="w-full h-full flex items-center justify-center">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img key={selected.path} src={resultUrl(selected)} alt={selected.prompt} className="max-w-full max-h-full rounded object-contain" />
                    </a>
                  )}
                </div>
              )}
            </div>

            {/* history strip */}
            <div className="h-20 shrink-0 m-2 flex items-center gap-2">
              <div className="flex-1 min-w-0 h-full flex items-center gap-2 overflow-x-auto overflow-y-hidden">
                {running && (
                  <button
                    type="button"
                    onClick={() => setShowPreview(true)}
                    title="Back to the live preview"
                    className={`h-16 w-16 shrink-0 rounded-md overflow-hidden border-2 ${showPreview ? 'border-blue-500' : 'border-transparent hover:border-gray-600'} bg-gray-800 relative flex items-center justify-center`}
                  >
                    <canvas ref={thumbRef} className="max-w-full max-h-full" />
                    <span className="absolute bottom-0.5 right-0.5 rounded bg-gray-950/80 p-0.5">
                      <Loader2 className="w-3 h-3 animate-spin text-blue-400" />
                    </span>
                  </button>
                )}
                {results.length === 0 && !running ? (
                  <div className="text-[11px] text-gray-600 px-1">History</div>
                ) : (
                  results.map(r => {
                    const isSel = selected?.path === r.path && !showPreview;
                    return (
                      <div key={`${r.request_id}-${r.path}`} className="relative shrink-0 group">
                        <button
                          type="button"
                          onClick={() => {
                            setSelected(r);
                            setShowPreview(false);
                          }}
                          title={`${r.arch} · seed ${r.seed} · ${r.prompt}`}
                          className={`h-16 w-16 rounded-md overflow-hidden border-2 ${isSel ? 'border-blue-500' : 'border-transparent hover:border-gray-600'} bg-gray-800 block`}
                        >
                          {r.kind === 'audio' ? (
                            <div className="w-full h-full flex items-center justify-center text-gray-400 text-[10px]">audio</div>
                          ) : r.kind === 'video' ? (
                            <video src={resultUrl(r)} muted className="w-full h-full object-cover" />
                          ) : (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={resultUrl(r)} alt="" className="w-full h-full object-cover" />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={e => {
                            e.stopPropagation();
                            deleteResults([r]);
                          }}
                          title="Delete"
                          className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-gray-900 border border-gray-600 text-gray-300 hover:bg-red-700 hover:text-white items-center justify-center hidden group-hover:flex"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
              {results.length > 0 && (
                <button
                  type="button"
                  onClick={deleteAll}
                  title="Delete all results"
                  className="shrink-0 h-16 w-10 rounded-md border border-gray-800 text-gray-500 hover:text-red-400 hover:border-red-900 flex items-center justify-center"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* ---- right: collapsible model + prompt sidebar ---- */}
          <div className={`relative shrink-0 transition-[width] duration-200 ${sidebarOpen ? 'w-[340px]' : 'w-0'}`}>
            <button
              type="button"
              onClick={() => setSidebarOpen(o => !o)}
              className="absolute top-1/2 -translate-y-1/2 -left-4 z-20 w-4 h-16 rounded-l-md bg-gray-800 border border-r-0 border-gray-700 text-gray-300 hover:bg-gray-700 flex items-center justify-center"
              title={sidebarOpen ? 'Collapse settings' : 'Expand settings'}
            >
              {sidebarOpen ? <ChevronRight className="w-3.5 h-3.5" /> : <ChevronLeft className="w-3.5 h-3.5" />}
            </button>
            <div className={`h-full flex flex-col border-l border-gray-800 bg-gray-900/60 ${sidebarOpen ? '' : 'hidden'}`}>
              <div className="compact-form flex-1 min-h-0 overflow-y-auto p-2 space-y-2">

                <Card title="Model" subtitle={entry?.label || arch} open={cardOpen('model')} onToggle={() => toggleCard('model')}>
                  <SelectInput label="Architecture" value={arch} onChange={v => applyArch(v as string)} options={archOptions} disabled={!ready} />
                  <CreatableSelectInput
                    label="Name or Path"
                    value={model.name_or_path || ''}
                    onChange={(v: string | null) => setModel(m => ({ ...m, name_or_path: v || '' }))}
                    options={entry?.model?.name_or_path ? [{ value: entry.model.name_or_path, label: entry.model.name_or_path }] : []}
                    placeholder="hub repo, local folder, or .safetensors"
                  />
                  {'extras_name_or_path' in (entry?.model || {}) && (
                    <TextInput label="Extras Name or Path" value={model.extras_name_or_path || ''} onChange={v => setModel(m => ({ ...m, extras_name_or_path: v }))} />
                  )}
                  <div className="grid grid-cols-2 gap-2">
                    <SelectInput
                      label="Quantize Transformer"
                      value={model.quantize ? model.qtype || 'convrot8' : ''}
                      onChange={v => setModel(m => ({ ...m, quantize: !!v, qtype: v || undefined }))}
                      options={[{ value: '', label: 'None (bf16)' }, ...qtypeOptions]}
                    />
                    <SelectInput
                      label="Quantize Text Encoder"
                      value={model.quantize_te ? model.qtype_te || 'convrot8' : ''}
                      onChange={v => setModel(m => ({ ...m, quantize_te: !!v, qtype_te: v || undefined }))}
                      options={[{ value: '', label: 'None (bf16)' }, ...qtypeOptions]}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    <Checkbox label="Low VRAM" checked={!!model.low_vram} onChange={v => setModel(m => ({ ...m, low_vram: v }))} />
                    <Checkbox label="Layer offloading" checked={!!model.layer_offloading} onChange={v => setModel(m => ({ ...m, layer_offloading: v }))} />
                  </div>
                  {model.layer_offloading && (
                    <div className="mt-2 space-y-1">
                      <SliderInput
                        label="Transformer Offload %"
                        value={Math.round((model.layer_offloading_transformer_percent ?? 1) * 100)}
                        onChange={v => setModel(m => ({ ...m, layer_offloading_transformer_percent: v * 0.01 }))}
                        min={0}
                        max={100}
                        step={1}
                      />
                      <SliderInput
                        label="Text Encoder Offload %"
                        value={Math.round((model.layer_offloading_text_encoder_percent ?? 1) * 100)}
                        onChange={v => setModel(m => ({ ...m, layer_offloading_text_encoder_percent: v * 0.01 }))}
                        min={0}
                        max={100}
                        step={1}
                      />
                    </div>
                  )}
                </Card>
    
                <Card title="LoRAs" subtitle={loras.length ? `${loras.length} loaded` : 'none'} open={cardOpen('loras')} onToggle={() => toggleCard('loras')}>
                  <div className="space-y-2">
                    {loras.map((l, i) => (
                      <div key={l.path} className="bg-gray-950/60 border border-gray-800 rounded-md px-2 py-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-200 truncate flex-1" title={l.path}>
                            {l.name}
                          </span>
                          <input
                            type="number"
                            step={0.05}
                            min={-2}
                            max={3}
                            value={l.strength}
                            onChange={e => {
                              const v = parseFloat(e.target.value);
                              const next = [...loras];
                              next[i] = { ...l, strength: isNaN(v) ? 0 : v };
                              setLoras(next);
                            }}
                            className="w-16 text-xs px-1.5 py-0.5 bg-gray-950 border border-gray-700 rounded text-gray-100 text-right"
                          />
                          <button
                            type="button"
                            onClick={() => setLoras(loras.filter(x => x.path !== l.path))}
                            className="text-gray-500 hover:text-red-400"
                            title="Remove"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={() => setLoraModalOpen(true)}
                        className="flex-1 px-2 py-1 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs flex items-center justify-center gap-1"
                      >
                        <Plus className="w-3.5 h-3.5" /> Add LoRA
                      </Button>
                    </div>
                    <Checkbox
                      label="Merge into weights"
                      checked={model.lora_mode === 'merge'}
                      onChange={v => setModel(m => ({ ...m, lora_mode: v ? 'merge' : 'hook' }))}
                      doc={{
                        title: 'Merge LoRAs into weights',
                        description: (
                          <>
                            <p>
                              <strong>Off</strong> (default): each LoRA is applied dynamically as an extra term on the layers it targets. Strength
                              changes take effect on the next generation and adding or removing a LoRA never reloads the model.
                            </p>
                            <p className="mt-2">
                              <strong>On</strong>: the LoRA deltas are added into the model weights. Quantized weights are dequantized, merged, and
                              re-quantized with stochastic rounding so small deltas are not rounded away. Merged weights cannot be un-merged, so
                              changing the LoRA set or strengths reloads the affected components. Slightly faster per step than the dynamic path.
                            </p>
                          </>
                        ),
                      }}
                    />
                  </div>
                </Card>
                <Card title="Prompt" open={cardOpen('prompt')} onToggle={() => toggleCard('prompt')}>
                  <TextAreaInput label="Prompt" value={sample.prompt || ''} onChange={v => setSample(s => ({ ...s, prompt: v }))} placeholder={modality === 'audio' ? 'song description, lyrics, bpm…' : 'describe what to generate'} />
                  {(sample.guidance_scale ?? 4) > 1 && (
                    <TextAreaInput label="Negative prompt" value={sample.negative_prompt || ''} onChange={v => setSample(s => ({ ...s, negative_prompt: v }))} />
                  )}
                  <div className="grid grid-cols-3 gap-2">
                    <NumberInput label="Width" value={sample.width ?? 1024} onChange={v => setSample(s => ({ ...s, width: v }))} min={64} max={4096} />
                    <NumberInput label="Height" value={sample.height ?? 1024} onChange={v => setSample(s => ({ ...s, height: v }))} min={64} max={4096} />
                    <NumberInput label="Steps" value={sample.num_inference_steps ?? 25} onChange={v => setSample(s => ({ ...s, num_inference_steps: v }))} min={1} max={200} />
                    <NumberInput label="Guidance" value={sample.guidance_scale ?? 4} onChange={v => setSample(s => ({ ...s, guidance_scale: v }))} min={1} max={30} />
                    <NumberInput label="Seed (-1 random)" value={sample.seed ?? -1} onChange={v => setSample(s => ({ ...s, seed: v }))} min={-1} max={4294967295} />
                    {modality === 'video' && (
                      <>
                        <NumberInput label="Frames" value={sample.num_frames ?? 33} onChange={v => setSample(s => ({ ...s, num_frames: v }))} min={1} max={1000} />
                        <NumberInput label="FPS" value={sample.fps ?? 16} onChange={v => setSample(s => ({ ...s, fps: v }))} min={1} max={60} />
                      </>
                    )}
                  </div>
                </Card>
    
              </div>
              {/* always-visible action bar */}
              <div className="shrink-0 p-2 border-t border-gray-800 bg-gray-900 flex gap-2">
                <Button
                  onClick={generate}
                  disabled={!ready || running || !model.name_or_path}
                  className="flex-1 px-3 py-1.5 rounded-md bg-blue-700 hover:bg-blue-600 disabled:opacity-40 text-white text-sm flex items-center justify-center gap-2"
                  title="Ctrl/Cmd + Enter"
                >
                  {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />} Generate
                </Button>
                {running && (
                  <Button onClick={cancel} className="px-3 py-2 rounded-md bg-gray-700 hover:bg-gray-600 text-white">
                    Cancel
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </MainContent>
      <GenerateFooter jobId={engineJobId} status={footerStatus} busy={running || isStarting} progress={running ? progress : null} />
      <LoraBrowserModal isOpen={loraModalOpen} onClose={() => setLoraModalOpen(false)} onPick={addLora} />
    </>
  );
}

export default function GeneratePage() {
  return (
    <Suspense fallback={null}>
      <GeneratePageInner />
    </Suspense>
  );
}
