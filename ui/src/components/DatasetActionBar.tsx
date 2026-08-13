'use client';

import { useState } from 'react';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { Cog, Download, Captions } from 'lucide-react';
import { LuLoader } from 'react-icons/lu';
import { apiClient } from '@/utils/api';

type DatasetZipTarget = 'dataset' | 'dataset_captions';

interface DatasetActionBarProps {
  datasetName: string;
  className?: string;
}

export default function DatasetActionBar({ datasetName, className }: DatasetActionBarProps) {
  const [zipping, setZipping] = useState<DatasetZipTarget | null>(null);

  const downloadZip = async (zipTarget: DatasetZipTarget) => {
    if (zipping) return;
    setZipping(zipTarget);
    try {
      const res = await apiClient.post('/api/zip', { zipTarget, datasetName });
      const zipPath = res.data.zipPath;
      if (!zipPath) throw new Error('No zipPath in response');

      // Cache-buster: /api/files serves with a long max-age, and the zip is rebuilt
      // at the same path every time, so a bare URL could hand back a stale download.
      const downloadPath = `/api/files/${encodeURIComponent(zipPath)}?v=${Date.now()}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      a.download = res.data.fileName || `${datasetName}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setZipping(null);
    }
  };

  const iconSizeClass = 'w-5 h-5 sm:w-6 sm:h-6';
  return (
    <div className={`flex items-center flex-shrink-0 ${className ?? ''}`}>
      <Menu>
        <MenuButton className={'ml-1 sm:ml-2'}>
          {zipping ? <LuLoader className={`${iconSizeClass} animate-spin`} /> : <Cog className={iconSizeClass} />}
        </MenuButton>
        <MenuItems
          anchor={{ to: 'bottom end', gap: 16 }}
          className="bg-gray-900 border border-gray-700 rounded shadow-lg w-64 px-2 py-2 z-50"
        >
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              onClick={() => downloadZip('dataset')}
            >
              <Download className="w-4 h-4" />
              Download Full Dataset
            </div>
          </MenuItem>
          <MenuItem>
            <div
              className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              onClick={() => downloadZip('dataset_captions')}
            >
              <Captions className="w-4 h-4" />
              Download Captions
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
    </div>
  );
}
