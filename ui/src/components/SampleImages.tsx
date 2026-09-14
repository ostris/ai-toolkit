import { useMemo, useState, useRef, useCallback, useEffect } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { Button, Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { FaDownload } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import { encodeFilePathForUrl } from '@/utils/basic';
import classNames from 'classnames';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';
import { openConfirm } from './ConfirmModal';

interface SampleImagesMenuProps {
  job?: Job | null;
}

export const SampleImagesMenu = ({ job }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = `/api/files/${encodeFilePathForUrl(zipPath)}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <Button
      onClick={downloadZip}
      className={classNames(
        `flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center`,
        {
          'opacity-50 cursor-not-allowed': isZipping,
        },
      )}
    >
      {isZipping ? (
        <LuLoader className="animate-spin inline-block sm:mr-2" />
      ) : (
        <FaDownload className="inline-block sm:mr-2" />
      )}
      <span className="hidden sm:inline">{isZipping ? 'Preparing' : 'Download'}</span>
    </Button>
  );
};

interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  const { sampleImages, status, refreshSampleImages } = useSampleImages(job.id, 5000);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  // multi-select for bulk delete: shift-click ranges from the anchor, ctrl/cmd-click toggles
  const [selectedSet, setSelectedSet] = useState<Set<string>>(() => new Set());
  const anchorIdxRef = useRef<number | null>(null);
  // selection as it was when the anchor was set; shift-click ranges are rebuilt on top of it, not accumulated
  const baseSetRef = useRef<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  const numSamples = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      const numPrompts = sampleConfig.prompts ? sampleConfig.prompts.length : 0;
      const numSamples = sampleConfig.samples.length;
      return Math.max(numPrompts, numSamples, 1);
    }
    return 10;
  }, [job]);

  // Group samples into rows of `numSamples` for the virtualized list — one row per sample iteration.
  const rows = useMemo(() => {
    const out: string[][] = [];
    for (let i = 0; i < sampleImages.length; i += numSamples) {
      out.push(sampleImages.slice(i, i + numSamples));
    }
    return out;
  }, [sampleImages, numSamples]);

  const handleCardClick = useCallback(
    (sample: string, e: React.MouseEvent) => {
      const isRange = e.shiftKey;
      const isToggle = e.ctrlKey || e.metaKey;
      if (!isRange && !isToggle) {
        setSelectedSet(new Set());
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
        setSelectedSamplePath(sample);
        return;
      }
      e.preventDefault();
      const idx = sampleImages.indexOf(sample);
      if (idx === -1) return;
      setSelectedSet(prev => {
        const anchor = anchorIdxRef.current;
        if (isRange && anchor !== null && anchor < sampleImages.length) {
          const next = new Set(baseSetRef.current);
          const [lo, hi] = anchor < idx ? [anchor, idx] : [idx, anchor];
          for (let i = lo; i <= hi; i++) next.add(sampleImages[i]);
          return next;
        }
        const next = new Set(prev);
        if (isToggle && next.has(sample)) {
          next.delete(sample);
        } else {
          next.add(sample);
        }
        anchorIdxRef.current = idx;
        baseSetRef.current = new Set(next);
        return next;
      });
    },
    [sampleImages],
  );

  const deleteSelected = useCallback(() => {
    const paths = Array.from(selectedSet);
    if (paths.length === 0) return;
    openConfirm({
      title: 'Delete Samples',
      message: `Are you sure you want to delete ${paths.length} sample${paths.length === 1 ? '' : 's'}? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeleteProgress({ done: 0, total: paths.length });
        // bounded concurrency so the progress counter advances and the server isn't flooded
        const CONCURRENCY = 4;
        let cursor = 0;
        let done = 0;
        const worker = async () => {
          while (cursor < paths.length) {
            const imgPath = paths[cursor++];
            try {
              await apiClient.post('/api/img/delete', { imgPath });
            } catch (error) {
              console.error('Error deleting sample:', imgPath, error);
            }
            done++;
            setDeleteProgress({ done, total: paths.length });
          }
        };
        await Promise.all(Array.from({ length: Math.min(CONCURRENCY, paths.length) }, worker));
        setDeleteProgress(null);
        setSelectedSet(new Set());
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
        refreshSampleImages();
      },
    });
  }, [selectedSet, refreshSampleImages]);

  // Delete/Backspace deletes the selection, Escape clears it; ignored while the viewer is open or typing in a field
  useEffect(() => {
    if (selectedSet.size === 0) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (selectedSamplePath) return;
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
      } else if (e.key === 'Escape') {
        setSelectedSet(new Set());
        anchorIdxRef.current = null;
        baseSetRef.current = new Set();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [selectedSet.size, selectedSamplePath, deleteSelected]);

  const scrollToBottom = () => {
    virtuosoRef.current?.scrollToIndex({ index: 'LAST', align: 'end' });
  };

  const scrollToTop = () => {
    virtuosoRef.current?.scrollToIndex({ index: 0, align: 'start' });
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

  // Inline style instead of Tailwind grid-cols-N classes — Tailwind only ships grid-cols-1..12,
  // so class-based columns silently break for larger sample counts.
  const gridCols = Math.max(numSamples, 3);

  const sampleConfig = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      return jobConfig.config.process[0].sample;
    }
    return null;
  }, [job]);

  return (
    <div ref={scrollParentCallback} className="absolute top-[80px] left-0 right-0 bottom-0 overflow-y-auto">
      <div className="pb-4">
        {PageInfoContent}
        {sampleImages && rows.length > 0 && scrollParent && (
          <Virtuoso
            ref={virtuosoRef}
            customScrollParent={scrollParent}
            totalCount={rows.length}
            initialTopMostItemIndex={rows.length - 1}
            followOutput="auto"
            increaseViewportBy={400}
            computeItemKey={index => rows[index]?.[0] ?? index}
            itemContent={index => {
              const row = rows[index];
              if (!row) return null;

              // Only pad the final row when numSamples < MIN_COLS and the row is short.
              const MIN_COLS = 3;
              const shouldPad = numSamples < MIN_COLS && row.length < MIN_COLS;
              const padsNeeded = shouldPad ? MIN_COLS - row.length : 0;

              return (
                // pb-1 recreates the vertical gap between rows that the original single CSS grid provided via `gap-1`.
                <div className="grid gap-1 pb-1" style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}>
                  {row.map(sample => (
                    <SampleImageCard
                      key={sample}
                      imageUrl={sample}
                      numSamples={numSamples}
                      sampleImages={sampleImages}
                      alt="Sample Image"
                      onClick={e => handleCardClick(sample, e)}
                      selected={selectedSet.has(sample) || selectedSamplePath === sample}
                      observerRoot={scrollParent}
                    />
                  ))}
                  {Array.from({ length: padsNeeded }).map((_, i) => (
                    <div key={`pad-${index}-${i}`} className="invisible" />
                  ))}
                </div>
              );
            }}
          />
        )}
      </div>
      <Dialog open={deleteProgress !== null} onClose={() => {}} className="relative z-20">
        <DialogBackdrop className="fixed inset-0 bg-gray-900/75" />
        <div className="fixed inset-0 z-10 flex items-center justify-center p-4">
          <DialogPanel className="w-full max-w-sm rounded-lg bg-gray-800 p-6 shadow-xl text-gray-200">
            <DialogTitle as="h3" className="text-base font-semibold flex items-center gap-2">
              <LuLoader className="animate-spin" />
              Deleting Samples
            </DialogTitle>
            <p className="mt-2 text-sm text-gray-400">
              {deleteProgress?.done ?? 0} / {deleteProgress?.total ?? 0}
            </p>
            <div className="mt-3 h-2 w-full rounded bg-gray-700 overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-150"
                style={{
                  width: `${deleteProgress && deleteProgress.total > 0 ? (deleteProgress.done / deleteProgress.total) * 100 : 0}%`,
                }}
              />
            </div>
          </DialogPanel>
        </div>
      </Dialog>
      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleImages}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
      <div
        className="hidden md:flex fixed top-20 mt-4 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToTop}
        title="Scroll to Top"
      >
        <FaCaretUp className="text-gray-500 dark:text-gray-400" />
      </div>
      <div
        className="hidden md:flex fixed bottom-5 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToBottom}
        title="Scroll to Bottom"
      >
        <FaCaretDown className="text-gray-500 dark:text-gray-400" />
      </div>
    </div>
  );
}
