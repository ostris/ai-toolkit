import React, { useEffect, useState } from 'react';
import useFilesList from '@/hooks/useFilesList';
import { Loader2, AlertCircle, Download, Box, Brain, FolderInput, Check } from 'lucide-react';
import { apiClient } from '@/utils/api';

export default function FilesWidget({ jobID }: { jobID: string }) {
  const { files, status, refreshFiles } = useFilesList(jobID, 5000);
  const [loraInstallPath, setLoraInstallPath] = useState('');
  const [installingFiles, setInstallingFiles] = useState<Record<string, 'idle' | 'installing' | 'success' | 'error'>>(
    {},
  );

  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        setLoraInstallPath(data.LORA_INSTALL_PATH || '');
      })
      .catch(() => {});
  }, []);

  const handleInstall = async (filePath: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setInstallingFiles(prev => ({ ...prev, [filePath]: 'installing' }));
    try {
      await apiClient.post('/api/files/install', { filePath });
      setInstallingFiles(prev => ({ ...prev, [filePath]: 'success' }));
      setTimeout(() => {
        setInstallingFiles(prev => ({ ...prev, [filePath]: 'idle' }));
      }, 2000);
    } catch {
      setInstallingFiles(prev => ({ ...prev, [filePath]: 'error' }));
      setTimeout(() => {
        setInstallingFiles(prev => ({ ...prev, [filePath]: 'idle' }));
      }, 2000);
    }
  };

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

  return (
    <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden hover:shadow-2xl transition-all duration-300 border border-gray-800">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h2 className="font-semibold text-gray-100">Checkpoints</h2>
          <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300">{files.length}</span>
        </div>
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
              const fileName = file.path.split('/').pop() || '';
              const nameWithoutExt = fileName.replace('.safetensors', '');
              const installStatus = installingFiles[file.path] || 'idle';
              return (
                <div
                  key={index}
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
                  <div className="flex items-center space-x-1 flex-shrink-0">
                    <span className="text-xs text-gray-400 mr-2">{cleanSize(file.size)}</span>
                    {loraInstallPath && (
                      <button
                        onClick={e => handleInstall(file.path, e)}
                        disabled={installStatus === 'installing'}
                        title={`Install to ${loraInstallPath}`}
                        className="bg-green-500 bg-opacity-0 hover:bg-opacity-10 rounded-full p-1 transition-all disabled:opacity-50"
                      >
                        {installStatus === 'installing' && (
                          <Loader2 className="w-3 h-3 text-green-400 animate-spin" />
                        )}
                        {installStatus === 'success' && <Check className="w-3 h-3 text-green-400" />}
                        {installStatus === 'error' && <AlertCircle className="w-3 h-3 text-red-400" />}
                        {installStatus === 'idle' && <FolderInput className="w-3 h-3 text-green-400" />}
                      </button>
                    )}
                    <a
                      href={`/api/files/${encodeURIComponent(file.path)}`}
                      target="_blank"
                      className="bg-purple-500 bg-opacity-0 hover:bg-opacity-10 rounded-full p-1 transition-all"
                    >
                      <Download className="w-3 h-3 text-purple-400" />
                    </a>
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
