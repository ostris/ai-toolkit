import Link from 'next/link';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Copy, AlertTriangle } from 'lucide-react';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import { startJob, stopJob, deleteJob, getAvaliableJobActions, markJobAsStopped } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import OverflowMenu from './OverflowMenu';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
  variant?: 'inline' | 'menu';
  stopPropagation?: boolean;
}

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
  variant = 'inline',
  stopPropagation = false,
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canRemoveFromQueue } = getAvaliableJobActions(job);

  if (!afterDelete) afterDelete = onRefresh;

  const wrap = (fn: (e: React.MouseEvent) => void) => (e: React.MouseEvent) => {
    if (stopPropagation) e.stopPropagation();
    fn(e);
  };

  const handleStart = wrap(async () => {
    await startJob(job.id);
    if (autoStartQueue) await startQueue(job.gpu_ids);
    onRefresh?.();
  });

  const handleStop = wrap(() => {
    openConfirm({
      title: 'Stop Job',
      message: `Are you sure you want to stop the job "${job.name}"? You CAN resume later.`,
      type: 'info',
      confirmText: 'Stop',
      onConfirm: async () => { await stopJob(job.id); onRefresh?.(); },
    });
  });

  const handleRemoveFromQueue = wrap(async () => {
    await markJobAsStopped(job.id);
    onRefresh?.();
  });

  const handleEdit = wrap(() => {
    if (job.job_type === 'caption') {
      openCaptionDatasetModal(job.job_ref || '', () => onRefresh?.(), { jobId: job.id });
    }
  });

  const handleDelete = wrap(() => {
    let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
    if (job.status === 'running') {
      message += ' WARNING: The job is currently running. You should stop it first if you can.';
    }
    openConfirm({
      title: 'Delete Job',
      message,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        if (job.status === 'running') {
          try { await stopJob(job.id); } catch (e) { console.error('Error stopping job before deleting:', e); }
        }
        await deleteJob(job.id);
        afterDelete?.();
      },
    });
  });

  const handleMarkStopped = wrap(() => {
    openConfirm({
      title: 'Mark Job as Stopped',
      message: `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`,
      type: 'warning',
      confirmText: 'Mark as Stopped',
      onConfirm: async () => { await markJobAsStopped(job.id); onRefresh?.(); },
    });
  });

  if (variant === 'menu') {
    const menuItemClass = 'w-full text-left cursor-pointer px-4 py-1.5 hover:bg-gray-800 rounded flex items-center gap-2 text-sm';
    return (
      <div className={className}>
        <OverflowMenu onClick={stopPropagation ? (e) => e.stopPropagation() : undefined}>
          {!hideView && (
            <MenuItem>
              <Link href={`/jobs/${job.id}`} className={menuItemClass} onClick={stopPropagation ? (e: React.MouseEvent) => e.stopPropagation() : undefined}>
                <Eye className="w-4 h-4" /> View
              </Link>
            </MenuItem>
          )}
          {canStart && (
            <MenuItem>
              <button className={menuItemClass} onClick={handleStart}><Play className="w-4 h-4" /> Start</button>
            </MenuItem>
          )}
          {canStop && (
            <MenuItem>
              <button className={menuItemClass} onClick={handleStop}><Pause className="w-4 h-4" /> Stop</button>
            </MenuItem>
          )}
          {canRemoveFromQueue && (
            <MenuItem>
              <button className={menuItemClass} onClick={handleRemoveFromQueue}><X className="w-4 h-4" /> Remove from Queue</button>
            </MenuItem>
          )}
          {job.job_type === 'train' && canEdit && (
            <MenuItem>
              <Link href={`/jobs/new?id=${job.id}`} className={menuItemClass} onClick={stopPropagation ? (e: React.MouseEvent) => e.stopPropagation() : undefined}>
                <Pen className="w-4 h-4" /> Edit
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'caption' && canEdit && (
            <MenuItem>
              <button className={menuItemClass} onClick={handleEdit}><Pen className="w-4 h-4" /> Edit</button>
            </MenuItem>
          )}
          <MenuItem>
            <button className={menuItemClass} onClick={handleDelete}><Trash2 className="w-4 h-4" /> Delete</button>
          </MenuItem>
          <div className="border-t border-gray-700 my-1"></div>
          {job.job_type === 'train' && (
            <MenuItem>
              <Link href={`/jobs/new?cloneId=${job.id}`} className={menuItemClass} onClick={stopPropagation ? (e: React.MouseEvent) => e.stopPropagation() : undefined}>
                <Copy className="w-4 h-4" /> Clone Job
              </Link>
            </MenuItem>
          )}
          <MenuItem>
            <button className={menuItemClass} onClick={handleMarkStopped}><AlertTriangle className="w-4 h-4" /> Mark as Stopped</button>
          </MenuItem>
        </OverflowMenu>
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      {canStart && (
        <Button onClick={handleStart} className="ml-2 opacity-100"><Play /></Button>
      )}
      {canRemoveFromQueue && (
        <Button onClick={handleRemoveFromQueue} className="ml-2 opacity-100"><X /></Button>
      )}
      {canStop && (
        <Button onClick={handleStop} className="ml-2 opacity-100"><Pause /></Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-2 text-gray-200 hover:text-gray-100 inline-block"><Eye /></Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div className="ml-2 hover:text-gray-100 inline-block cursor-pointer" onClick={handleEdit}><Pen /></div>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-2 hover:text-gray-100 inline-block"><Pen /></Link>
      )}
      <Button onClick={handleDelete} className="ml-2 opacity-100"><Trash2 /></Button>
      <div className="border-r border-1 border-gray-700 ml-2 inline"></div>
      <Menu>
        <MenuButton className="ml-2"><Cog /></MenuButton>
        <MenuItems anchor="bottom" className="bg-gray-900 border border-gray-700 rounded shadow-lg w-56 px-2 py-2 mt-4">
          {job.job_type === 'train' && (
            <MenuItem>
              <Link href={`/jobs/new?cloneId=${job.id}`} className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded block">Clone Job</Link>
            </MenuItem>
          )}
          <MenuItem>
            <div className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded" onClick={handleMarkStopped}>Mark as Stopped</div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
