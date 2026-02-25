'use client';

import { useEffect, useState, useCallback } from 'react';
import { LuImageOff, LuLoader } from 'react-icons/lu';
import { FaTrashAlt, FaTimes, FaUndoAlt } from 'react-icons/fa';
import TrashImageCard from '@/components/TrashImageCard';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { openConfirm } from '@/components/ConfirmModal';

export default function TrashPage() {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [isSelectMode, setIsSelectMode] = useState<boolean>(false);
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set());

  const refreshTrashList = useCallback(() => {
    setStatus('loading');
    apiClient
      .get('/api/trash/list')
      .then((res: any) => {
        const data = res.data;
        data.images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
        setImgList(data.images);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching trash:', error);
        setStatus('error');
      });
  }, []);

  useEffect(() => {
    refreshTrashList();
  }, [refreshTrashList]);

  const removeImageFromList = useCallback((imgPath: string) => {
    setImgList(prev => prev.filter(x => x.img_path !== imgPath));
  }, []);

  const handleRestore = useCallback(
    (imgPath: string) => {
      apiClient
        .post('/api/img/restore', { imgPath })
        .then(() => {
          removeImageFromList(imgPath);
        })
        .catch(error => console.error('Error restoring image:', error));
    },
    [removeImageFromList],
  );

  const handlePermanentDelete = useCallback(
    (imgPath: string) => {
      openConfirm({
        title: 'Permanently Delete',
        message: 'Are you sure you want to permanently delete this file? This action cannot be undone.',
        type: 'warning',
        confirmText: 'Delete',
        onConfirm: () => {
          apiClient
            .post('/api/img/delete', { imgPath })
            .then(() => removeImageFromList(imgPath))
            .catch(error => console.error('Error deleting image:', error));
        },
      });
    },
    [removeImageFromList],
  );

  const handleLongPress = useCallback((imgPath: string) => {
    setIsSelectMode(true);
    setSelectedImages(new Set([imgPath]));
  }, []);

  const handleSelect = useCallback((imgPath: string) => {
    setSelectedImages(prev => {
      const next = new Set(prev);
      if (next.has(imgPath)) {
        next.delete(imgPath);
      } else {
        next.add(imgPath);
      }
      return next;
    });
  }, []);

  const handleCancelSelect = useCallback(() => {
    setIsSelectMode(false);
    setSelectedImages(new Set());
  }, []);

  useEffect(() => {
    if (isSelectMode && selectedImages.size === 0) {
      setIsSelectMode(false);
    }
  }, [isSelectMode, selectedImages.size]);

  const handleBulkRestore = useCallback(async () => {
    const paths = Array.from(selectedImages);
    await Promise.all(
      paths.map(imgPath =>
        apiClient
          .post('/api/img/restore', { imgPath })
          .then(() => removeImageFromList(imgPath))
          .catch(error => console.error('Error restoring image:', error)),
      ),
    );
    setIsSelectMode(false);
    setSelectedImages(new Set());
  }, [selectedImages, removeImageFromList]);

  const handleEmptyTrash = useCallback(() => {
    openConfirm({
      title: 'Empty Trash',
      message: `Are you sure you want to permanently delete all ${imgList.length} file${imgList.length !== 1 ? 's' : ''} in the trash? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Empty Trash',
      onConfirm: () => {
        apiClient
          .post('/api/trash/empty')
          .then(() => {
            setImgList([]);
          })
          .catch(error => console.error('Error emptying trash:', error));
      },
    });
  }, [imgList.length]);

  return (
    <>
      <TopBar>
        {isSelectMode ? (
          <>
            <div>
              <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={handleCancelSelect}>
                <FaTimes />
              </Button>
            </div>
            <div>
              <h1 className="text-lg">
                {selectedImages.size} image{selectedImages.size !== 1 ? 's' : ''} selected
              </h1>
            </div>
            <div className="flex-1"></div>
            <div>
              <Button
                className="text-gray-200 bg-blue-700 px-3 py-1 rounded-md flex items-center gap-2 disabled:opacity-50"
                onClick={handleBulkRestore}
                disabled={selectedImages.size === 0}
              >
                <FaUndoAlt />
                Restore Selected
              </Button>
            </div>
          </>
        ) : (
          <>
            <div>
              <h1 className="text-2xl font-semibold text-gray-100">
                Trash{status === 'success' ? `, Files: ${imgList.length}` : ''}
              </h1>
            </div>
            <div className="flex-1"></div>
            {imgList.length > 0 && (
              <div>
                <Button
                  className="text-gray-200 bg-red-700 px-3 py-1 rounded-md flex items-center gap-2"
                  onClick={handleEmptyTrash}
                >
                  <FaTrashAlt />
                  Empty Trash
                </Button>
              </div>
            )}
          </>
        )}
      </TopBar>

      <MainContent>
        {isSelectMode && (
          <p className="text-xs text-gray-400 mb-3">
            Click images to select or deselect. Press Cancel to exit select mode.
          </p>
        )}

        {status === 'loading' && (
          <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
            <LuLoader className="animate-spin w-8 h-8 text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Loading Trash</h3>
            <p className="text-sm opacity-75 leading-relaxed">Please wait...</p>
          </div>
        )}

        {status === 'success' && imgList.length === 0 && (
          <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
            <LuImageOff className="w-8 h-8 text-gray-400 mb-4" />
            <h3 className="text-lg font-semibold mb-2">Trash is Empty</h3>
            <p className="text-sm opacity-75 leading-relaxed">
              No files in trash. Deleted dataset images will appear here.
            </p>
          </div>
        )}

        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {imgList.map(img => (
              <TrashImageCard
                key={img.img_path}
                alt="trashed image"
                imageUrl={img.img_path}
                onRestore={() => handleRestore(img.img_path)}
                onDelete={() => handlePermanentDelete(img.img_path)}
                isSelectMode={isSelectMode}
                selected={selectedImages.has(img.img_path)}
                onLongPress={() => handleLongPress(img.img_path)}
                onSelect={() => handleSelect(img.img_path)}
              />
            ))}
          </div>
        )}
      </MainContent>
    </>
  );
}
