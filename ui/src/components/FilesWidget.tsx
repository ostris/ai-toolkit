import React, { useMemo, useState } from 'react';
import useFilesList from '@/hooks/useFilesList';
import Link from 'next/link';
import { Loader2, Download, Box, Brain, Upload, X } from 'lucide-react';
import { openCheckpointUploadModal } from '@/components/CheckpointUploadModal';
import { apiClient } from '@/utils/api';
import useJob from '@/hooks/useJob';

export default function FilesWidget({ jobID }: { jobID: string }) {
  const { files, status, refreshFiles, baseCheckpoint } = useFilesList(jobID, 5000);
  const [removing, setRemoving] = useState<boolean>(false);
  const { job } = useJob(jobID, 3000); // Poll every 3 seconds for job status updates
  const jobStatus = useMemo(() => job?.status?.toLowerCase() || 'unknown', [job?.status]);
  const isActive = jobStatus === 'running' || jobStatus === 'stopping';

  const cleanSize = (size: number) => {
    if (size < 1024) {
      return `${size} B`;
    } else if (size < 1024 * 1024) {
      return `${(size / 1024).toFixed(1)} KB`;
    } else if (size < 1024 * 1024 * 1024) {
      return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    } else {
      return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    }
  };

  const openCheckpointUpload = () => {
    openCheckpointUploadModal(jobID, () => {
      refreshFiles();
    });
  };

  const removeBaseCheckpoint = async () => {
    try {
      setRemoving(true);
      const resp = await apiClient.delete(`/api/jobs/${jobID}/checkpoints/base`);
      await refreshFiles();
    } catch (e: any) {
      console.error('Failed to remove base checkpoint', e);
    } finally {
      setRemoving(false);
    }
  };

  return (
    <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h2 className="font-semibold text-gray-100">Checkpoints</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">{files.length}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            disabled={!!baseCheckpoint}
            onClick={openCheckpointUpload}
            className={`inline-flex items-center gap-2 rounded-md bg-gray-700 text-gray-100 px-3 py-1.5 text-sm transition-colors ${
              baseCheckpoint ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-600'
            }`}
            title="Import checkpoint (.safetensors)"
          >
            <Upload className="w-4 h-4" />
            Import
          </button>
        </div>
      </div>

      <div className="p-2">
        {/* Upload progress is handled inside the modal */}
        {status === 'loading' && (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="w-5 h-5 text-gray-400 animate-spin" />
          </div>
        )}

        {status === 'error' && (
          <div className="flex items-center justify-center py-4 text-rose-400 space-x-2 text-sm">Error loading checkpoints</div>
        )}

        {['success', 'refreshing'].includes(status) && (
          <div className="space-y-1">
            {files.map((file, index) => {
              const fileName = file.path.split('/').pop() || '';
              const nameWithoutExt = fileName.replace('.safetensors', '');
              return (
                <a
                  key={index}
                  target="_blank"
                  href={`/api/files/${encodeURIComponent(file.path)}`}
                  className="group flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-gray-800 transition-all duration-200"
                >
                  <div className="flex items-center space-x-2 min-w-0">
                    <Box className="w-4 h-4 text-purple-400 flex-shrink-0" />
                    <div className="flex flex-col min-w-0">
                      <div className="flex text-sm text-gray-200">
                        <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">
                          {nameWithoutExt}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">.safetensors</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-xs text-gray-400">{cleanSize(file.size)}</span>
                    <div className="bg-purple-500 bg-opacity-0 group-hover:bg-opacity-10 rounded-full p-1 transition-all">
                      <Download className="w-3 h-3 text-purple-400" />
                    </div>
                    {baseCheckpoint === fileName && !isActive && (
                      <button
                        className="p-1 rounded hover:bg-gray-700"
                        title={'Remove base'}
                        disabled={removing}
                        onClick={e => {
                          e.preventDefault();
                          e.stopPropagation();
                          removeBaseCheckpoint();
                        }}
                      >
                        <X className="w-3 h-3 text-rose-400" />
                      </button>
                    )}
                  </div>
                </a>
              );
            })}
          </div>
        )}

        {['success', 'refreshing'].includes(status) && files.length === 0 && (
          <div className="text-center py-4 text-gray-400 text-sm">No checkpoints available</div>
        )}
      </div>
    </div>
  );
}
