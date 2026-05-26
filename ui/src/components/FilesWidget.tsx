import React from 'react';
import useFilesList from '@/hooks/useFilesList';
import { Loader2, AlertCircle, Download, Box, Brain, Trash2 } from 'lucide-react';
import { openMergeLoRAsModal } from './MergeLoRAsModal';
import { getFilename, getFoldername } from '@/utils/basic';
import { openConfirm } from './ConfirmModal';
import { apiClient } from '@/utils/api';

export default function FilesWidget({ jobID, jobName }: { jobID: string; jobName: string }) {
  const { files, status, refreshFiles } = useFilesList(jobID, 5000);

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

  const handleDeleteFile = (filePath: string) => {
    const fileName = getFilename(filePath);
    openConfirm({
      title: 'Delete Checkpoint',
      message: `Are you sure you want to delete "${fileName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/files/delete', { filePath })
          .then(() => {
            refreshFiles();
          })
          .catch(error => {
            console.error('Error deleting checkpoint:', error);
          });
      },
    });
  };

  return (
    <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
          <h2 className="font-semibold text-gray-100">Checkpoints</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">{files.length}</span>
        </div>
        {files.length > 0 && (
          <span
            className="px-3 py-1 rounded-full text-sm bg-purple-500/10 text-purple-500 uppercase cursor-pointer hover:bg-purple-500/20"
            onClick={() => {
              const outputName = `${jobName}_merged`;
              openMergeLoRAsModal(
                getFoldername(files[0].path),
                outputName,
                files.map(f => ({ path: f.path })),
                () => {
                  refreshFiles();
                },
              );
            }}
          >
            merge
          </span>
        )}
      </div>

      <div className="p-2">
        {status === 'loading' && (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="w-5 h-5 text-gray-400 animate-spin" />
          </div>
        )}

        {status === 'error' && (
          <div className="flex items-center justify-center py-4 text-rose-400 space-x-2">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">Error loading checkpoints</span>
          </div>
        )}

        {['success', 'refreshing'].includes(status) && (
          <div className="space-y-1">
            {files.map((file, index) => {
              const fileName = getFilename(file.path);
              const nameWithoutExt = fileName.replace('.safetensors', '');
              return (
                <div
                  key={index}
                  className="group flex items-center justify-between px-2 py-1.5 rounded-lg hover:bg-gray-800 transition-all duration-200"
                >
                  <a
                    target="_blank"
                    href={`/api/files/${encodeURIComponent(file.path)}`}
                    className="flex items-center space-x-2 min-w-0 flex-1"
                  >
                    <Box className="w-4 h-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                    <div className="flex flex-col min-w-0">
                      <div className="flex text-sm text-gray-200">
                        <span className="overflow-hidden text-ellipsis direction-rtl whitespace-nowrap">
                          {nameWithoutExt}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">.safetensors</span>
                    </div>
                  </a>
                  <div className="flex items-center space-x-3 flex-shrink-0">
                    <span className="text-xs text-gray-400">{cleanSize(file.size)}</span>
                    <a
                      target="_blank"
                      href={`/api/files/${encodeURIComponent(file.path)}`}
                      className="bg-purple-500 bg-opacity-0 group-hover:bg-opacity-10 rounded-full p-1 transition-all"
                    >
                      <Download className="w-3 h-3 text-purple-600 dark:text-purple-400" />
                    </a>
                    <button
                      type="button"
                      onClick={() => handleDeleteFile(file.path)}
                      className="bg-red-500 bg-opacity-0 group-hover:bg-opacity-10 hover:!bg-opacity-30 rounded-full p-1 transition-all"
                      title="Delete checkpoint"
                    >
                      <Trash2 className="w-3 h-3 text-red-500" />
                    </button>
                  </div>
                </div>
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
