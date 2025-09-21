'use client';

import { useState } from 'react';
import { LuEdit, LuChevronDown, LuChevronRight, LuCheck, LuX } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { apiClient } from '@/utils/api';

interface RenamingSectionProps {
  datasetName: string;
  isExpanded: boolean;
  onToggleExpanded: () => void;
  onRenameComplete: () => void;
}

export default function RenamingSection({
  datasetName,
  isExpanded,
  onToggleExpanded,
  onRenameComplete,
}: RenamingSectionProps) {
  const [renameText, setRenameText] = useState(datasetName);
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameStatus, setRenameStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  const handleRename = async () => {
    if (!renameText.trim()) {
      alert('Please enter a valid name');
      return;
    }

    setIsRenaming(true);
    setRenameStatus(null); // Clear previous status
    
    try {
      const response = await apiClient.post('/api/datasets/rename', {
        datasetName: datasetName,
        newBaseName: renameText.trim()
      });

      if (response.data.success) {
        console.log('Rename successful:', response.data);
        setRenameStatus({
          type: 'success',
          message: `Successfully renamed ${response.data.totalRenamed} files!`
        });
        // Notify parent component to refresh
        onRenameComplete();
      } else {
        setRenameStatus({
          type: 'error',
          message: 'Rename failed: ' + (response.data.error || 'Unknown error')
        });
      }
    } catch (error: any) {
      console.error('Rename error:', error);
      setRenameStatus({
        type: 'error',
        message: 'Failed to rename files: ' + (error.response?.data?.error || error.message)
      });
    } finally {
      setIsRenaming(false);
    }
  };

  return (
    <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
      <button
        onClick={onToggleExpanded}
        className="flex items-center gap-2 w-full text-left focus:outline-none"
      >
        {isExpanded ? (
          <LuChevronDown className="w-5 h-5 text-gray-500" />
        ) : (
          <LuChevronRight className="w-5 h-5 text-gray-500" />
        )}
        <LuEdit className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Renaming</h2>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={renameText}
              onChange={(e) => setRenameText(e.target.value)}
              disabled={isRenaming}
              className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
              placeholder="Enter new base name for files"
            />
            <Button
              onClick={handleRename}
              disabled={isRenaming || !renameText.trim()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
            >
              {isRenaming ? 'Renaming...' : 'Rename'}
            </Button>
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            All media files (images and videos) will be renamed to "{renameText}001.ext", "{renameText}002.ext", etc. (keeping original extensions)
          </p>

          {/* Rename Status Message */}
          {renameStatus && (
            <div className={`mt-4 p-3 rounded-md flex items-center gap-2 ${
              renameStatus.type === 'success' 
                ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800' 
                : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800'
            }`}>
              {renameStatus.type === 'success' ? (
                <LuCheck className="w-5 h-5 flex-shrink-0" />
              ) : (
                <LuX className="w-5 h-5 flex-shrink-0" />
              )}
              <span className="text-sm font-medium flex-1">{renameStatus.message}</span>
              <button
                onClick={() => setRenameStatus(null)}
                className="text-current hover:opacity-70 transition-opacity"
              >
                <LuX className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
