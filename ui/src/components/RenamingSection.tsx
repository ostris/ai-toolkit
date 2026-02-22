'use client';

import { useState, useEffect } from 'react';
import { LuPencil, LuChevronDown, LuChevronRight, LuCheck, LuX, LuInfo } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { apiClient } from '@/utils/api';

// Common naming patterns
const COMMON_PATTERNS = [
  { name: 'Dataset + Index', pattern: '{dataset}-{###}', description: 'datasetname-001.jpg' },
  { name: 'Simple Index', pattern: '{####}', description: '0001.jpg' },
  { name: 'Date + Index', pattern: '{dataset}_{YYYYMMDD}_{###}', description: 'datasetname_20241221_001.jpg' },
  { name: 'IMG + Index', pattern: 'IMG_{#####}', description: 'IMG_00001.jpg' },
  { name: 'Date Folder Style', pattern: '{YYYY}-{MM}-{DD}_{dataset}_{###}', description: '2024-12-21_datasetname_001.jpg' },
  { name: 'Timestamp + Index', pattern: 'batch_{timestamp}_{###}', description: 'batch_1703174632_001.jpg' },
];

interface RenamingSectionProps {
  datasetName: string;
  isExpanded: boolean;
  onToggleExpanded: () => void;
  onRenameComplete: () => void;
}

interface PreviewItem {
  oldName: string;
  newName: string;
}

export default function RenamingSection({
  datasetName,
  isExpanded,
  onToggleExpanded,
  onRenameComplete,
}: RenamingSectionProps) {
  const [pattern, setPattern] = useState(`{dataset}-{###}`);
  const [isRenaming, setIsRenaming] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [preview, setPreview] = useState<PreviewItem[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [renameStatus, setRenameStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  // Load preview when pattern changes
  useEffect(() => {
    if (pattern.trim() && isExpanded) {
      loadPreview();
    }
  }, [pattern, datasetName, isExpanded]);

  const loadPreview = async () => {
    if (!pattern.trim()) return;

    setPreviewLoading(true);
    try {
      const response = await apiClient.get('/api/datasets/rename', {
        params: {
          datasetName: datasetName,
          pattern: pattern
        }
      });

      if (response.data.success) {
        setPreview(response.data.previews);
      } else {
        setPreview([]);
      }
    } catch (error: any) {
      console.error('Preview error:', error);
      setPreview([]);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleRename = async () => {
    if (!pattern.trim()) {
      alert('Please enter a valid pattern');
      return;
    }

    setIsRenaming(true);
    setRenameStatus(null); // Clear previous status

    try {
      const response = await apiClient.post('/api/datasets/rename', {
        datasetName: datasetName,
        pattern: pattern.trim()
      });

      if (response.data.success) {
        console.log('Rename successful:', response.data);
        setRenameStatus({
          type: 'success',
          message: `Successfully renamed ${response.data.totalRenamed} files!`
        });
        // Delay the refresh to allow the success message to be visible
        setTimeout(() => {
          onRenameComplete();
        }, 2000); // Show success message for 2 seconds before refreshing
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
        <LuPencil className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Renaming</h2>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Pattern Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Naming Pattern
            </label>
            <div className="space-y-2">
              <select
                value={COMMON_PATTERNS.find(p => p.pattern === pattern)?.name || 'Custom'}
                onChange={(e) => {
                  const selectedPattern = COMMON_PATTERNS.find(p => p.name === e.target.value);
                  if (selectedPattern) {
                    setPattern(selectedPattern.pattern);
                  }
                }}
                disabled={isRenaming}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
              >
                {COMMON_PATTERNS.map(p => (
                  <option key={p.name} value={p.name}>{p.name} - {p.description}</option>
                ))}
                <option value="Custom">Custom Pattern</option>
              </select>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={pattern}
                  onChange={(e) => setPattern(e.target.value)}
                  disabled={isRenaming}
                  className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                  placeholder="Enter custom pattern like {dataset}-{###}"
                />
                <Button
                  onClick={() => setShowHelp(!showHelp)}
                  className="px-3 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-md transition-colors"
                >
                  <LuInfo className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>

          {/* Help Section */}
          {showHelp && (
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
              <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Pattern Reference:</h4>
              <div className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                <div><strong>Index:</strong> {`{###}`} = 001, 002, 003 | {`{####}`} = 0001, 0002, 0003</div>
                <div><strong>Date:</strong> {`{YYYY}`} = 2024 | {`{MM}`} = 12 | {`{DD}`} = 21 | {`{YYYYMMDD}`} = 20241221</div>
                <div><strong>Time:</strong> {`{HHMMSS}`} = 143052 | {`{timestamp}`} = 1703174632</div>
                <div><strong>File:</strong> {`{dataset}`} = dataset name | {`{original}`} = original filename | {`{ext}`} = .jpg</div>
                <div><strong>Random:</strong> {`{random}`} = abc123 | {`{uuid}`} = unique identifier</div>
              </div>
            </div>
          )}

          {/* Preview */}
          {preview.length > 0 && (
            <div className="p-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md">
              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Preview (first 5 files):</h4>
              <div className="space-y-1 text-sm">
                {preview.map((item, index) => (
                  <div key={index} className="flex justify-between text-gray-700 dark:text-gray-300">
                    <span className="truncate">{item.oldName}</span>
                    <span className="mx-2">→</span>
                    <span className="truncate font-medium">{item.newName}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action Button */}
          <div className="flex gap-2">
            <Button
              onClick={handleRename}
              disabled={isRenaming || !pattern.trim() || previewLoading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
            >
              {isRenaming ? 'Renaming...' : 'Rename Files'}
            </Button>
            {previewLoading && (
              <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
                Loading preview...
              </div>
            )}
          </div>

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
