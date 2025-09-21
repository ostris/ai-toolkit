'use client';

import { useEffect, useState, use, useMemo } from 'react';
import { LuImageOff, LuLoader, LuBan, LuWand, LuCheck, LuX, LuClock, LuChevronDown, LuChevronRight } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import FullscreenDropOverlay from '@/components/FullscreenDropOverlay';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [renameText, setRenameText] = useState(datasetName);
  const [isRenaming, setIsRenaming] = useState(false);

  // Captioning state
  const [captionStyle, setCaptionStyle] = useState('descriptive');
  const [customPrompt, setCustomPrompt] = useState('');
  const [isCaptioning, setIsCaptioning] = useState(false);
  const [captionProgress, setCaptionProgress] = useState<{
    total: number;
    completed: number;
    current?: string;
  } | null>(null);
  const [captionServiceAvailable, setCaptionServiceAvailable] = useState<boolean | null>(null);
  const [availableStyles, setAvailableStyles] = useState<string[]>([]);
  const [overwriteExisting, setOverwriteExisting] = useState(false);
  const [isStartingService, setIsStartingService] = useState(false);
  const [isServiceLoading, setIsServiceLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState<{
    status: string;
    message: string;
    progress: number;
    current_file?: string;
  } | null>(null);

  // Collapsible sections state
  const [isRenamingExpanded, setIsRenamingExpanded] = useState(false);
  const [isCaptioningExpanded, setIsCaptioningExpanded] = useState(false);

  // Check caption service availability
  const checkCaptionService = async () => {
    try {
      const response = await apiClient.get('/api/datasets/caption/health');
      setCaptionServiceAvailable(response.data.available);

      // Check if service is running but model is still loading
      const serviceStatus = response.data.serviceStatus;
      const isServiceHealthy = serviceStatus?.status === 'healthy';
      const isModelLoaded = response.data.modelLoaded;
      setIsServiceLoading(isServiceHealthy && !isModelLoaded);

      // Get loading progress if available
      if (response.data.serviceStatus?.loading_progress) {
        setLoadingProgress(response.data.serviceStatus.loading_progress);
      }

      if (response.data.available) {
        // Get available styles
        const stylesResponse = await apiClient.get('/api/datasets/caption');
        setAvailableStyles(stylesResponse.data.styles || []);
        setLoadingProgress(null); // Clear progress when ready
      } else {
        // Try to get detailed progress
        try {
          const progressResponse = await apiClient.get('/api/datasets/caption/progress');
          if (progressResponse.data.available) {
            setLoadingProgress(progressResponse.data.progress);
          }
        } catch (progressError) {
          // Progress endpoint not available, service likely not running
          setLoadingProgress(null);
        }
      }
    } catch (error) {
      console.error('Failed to check caption service:', error);
      setCaptionServiceAvailable(false);
      setIsServiceLoading(false);
      setLoadingProgress(null);
    }
  };

  // Poll for progress updates
  const pollProgress = async () => {
    try {
      const response = await apiClient.get('/api/datasets/caption/progress');
      if (response.data.available) {
        setLoadingProgress(response.data.progress);

        // If ready, stop polling and check service
        if (response.data.progress.status === 'ready') {
          setIsStartingService(false);
          checkCaptionService();
          return false; // Stop polling
        }
        return true; // Continue polling
      }
    } catch (error) {
      console.error('Failed to get progress:', error);
    }
    return true; // Continue polling
  };

  // Start caption service in background
  const startCaptionService = async () => {
    setIsStartingService(true);
    setLoadingProgress({
      status: 'starting',
      message: 'Starting caption service...',
      progress: 0.0
    });

    try {
      const response = await apiClient.post('/api/datasets/caption/start');
      if (response.data.success) {
        // Start polling for progress
        const pollInterval = setInterval(async () => {
          const shouldContinue = await pollProgress();
          if (!shouldContinue) {
            clearInterval(pollInterval);
          }
        }, 2000); // Poll every 2 seconds

        // Fallback timeout to stop polling after 10 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          setIsStartingService(false);
        }, 600000);
      } else {
        alert('Failed to start caption service: ' + (response.data.error || 'Unknown error'));
        setIsStartingService(false);
        setLoadingProgress(null);
      }
    } catch (error: any) {
      console.error('Failed to start caption service:', error);
      alert('Failed to start caption service: ' + (error.response?.data?.error || error.message));
      setIsStartingService(false);
      setLoadingProgress(null);
    }
  };

  const handleCaption = async () => {
    if (!captionServiceAvailable) {
      alert('Caption service is not available. Please ensure the captioning service is running.');
      return;
    }

    setIsCaptioning(true);
    setCaptionProgress({ total: imgList.length, completed: 0 });

    try {
      const response = await apiClient.post('/api/datasets/caption', {
        datasetName: datasetName,
        style: captionStyle,
        prompt: customPrompt.trim() || undefined,
        overwriteExisting: overwriteExisting,
        saveToFile: true,
      });

      if (response.data.success) {
        const stats = response.data.statistics;
        alert(`Captioning completed!\nSuccessful: ${stats.successful}\nFailed: ${stats.failed}\nTotal time: ${stats.totalTime.toFixed(1)}s`);
        // Refresh the image list to show any changes
        refreshImageList(datasetName);
      } else {
        alert('Captioning failed: ' + (response.data.error || 'Unknown error'));
      }
    } catch (error: any) {
      console.error('Caption error:', error);
      alert('Failed to caption images: ' + (error.response?.data?.error || error.message));
    } finally {
      setIsCaptioning(false);
      setCaptionProgress(null);
    }
  };

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    console.log('Fetching images for dataset:', dbName);
    apiClient
      .post('/api/datasets/listImages', { datasetName: dbName })
      .then((res: any) => {
        const data = res.data;
        console.log('Images:', data.images);
        // sort
        data.images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
        setImgList(data.images);
        setStatus('success');
      })
      .catch((error: any) => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };

  const handleRename = async () => {
    if (!renameText.trim()) {
      alert('Please enter a valid name');
      return;
    }

    setIsRenaming(true);
    try {
      const response = await apiClient.post('/api/datasets/rename', {
        datasetName: datasetName,
        newBaseName: renameText.trim()
      });

      if (response.data.success) {
        console.log('Rename successful:', response.data);
        // Refresh the image list to show the renamed files
        refreshImageList(datasetName);
        alert(`Successfully renamed ${response.data.totalRenamed} files`);
      } else {
        alert('Rename failed: ' + (response.data.error || 'Unknown error'));
      }
    } catch (error: any) {
      console.error('Rename error:', error);
      alert('Failed to rename files: ' + (error.response?.data?.error || error.message));
    } finally {
      setIsRenaming(false);
    }
  };
  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
      setRenameText(datasetName);
      checkCaptionService();
    }
  }, [datasetName]);

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Please wait while we fetch your dataset images...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'There was a problem fetching the images. Please try refreshing the page.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Images Found';
      subtitle = 'This dataset is empty. Click "Add Images" to get started.';
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
  }, [status, imgList.length]);

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Dataset: {datasetName}</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-gray-200 bg-slate-600 px-3 py-1 rounded-md"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            Add Images
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {PageInfoContent}

        {/* Renaming Section - Only visible when files are successfully uploaded */}
        {status === 'success' && imgList.length > 0 && (
          <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setIsRenamingExpanded(!isRenamingExpanded)}
              className="flex items-center gap-2 w-full text-left focus:outline-none"
            >
              {isRenamingExpanded ? (
                <LuChevronDown className="w-5 h-5 text-gray-500" />
              ) : (
                <LuChevronRight className="w-5 h-5 text-gray-500" />
              )}
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Renaming</h2>
            </button>

            {isRenamingExpanded && (
              <div className="mt-4 space-y-4">
                <div className="flex gap-3 items-center">
                  <input
                    type="text"
                    value={renameText}
                    onChange={(e) => setRenameText(e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Enter base name for files"
                    disabled={isRenaming}
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
              </div>
            )}
          </div>
        )}

        {/* Auto-Captioning Section - Only visible when files are successfully uploaded */}
        {status === 'success' && imgList.length > 0 && (
          <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setIsCaptioningExpanded(!isCaptioningExpanded)}
              className="flex items-center gap-2 w-full text-left focus:outline-none"
            >
              {isCaptioningExpanded ? (
                <LuChevronDown className="w-5 h-5 text-gray-500" />
              ) : (
                <LuChevronRight className="w-5 h-5 text-gray-500" />
              )}
              <LuWand className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Auto-Captioning</h2>
              {captionServiceAvailable === false && !isStartingService && !isServiceLoading && (
                <span className="text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-2 py-1 rounded-full">
                  Service Unavailable
                </span>
              )}
              {captionServiceAvailable === true && (
                <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 px-2 py-1 rounded-full">
                  Service Ready
                </span>
              )}
              {(captionServiceAvailable === null || isStartingService || isServiceLoading) && (
                <LuLoader className="w-4 h-4 animate-spin text-gray-500" />
              )}
              {isStartingService && (
                <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-2 py-1 rounded-full">
                  Starting Service...
                </span>
              )}
              {isServiceLoading && !isStartingService && (
                <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-2 py-1 rounded-full">
                  Loading Model...
                </span>
              )}
            </button>

            {isCaptioningExpanded && (
              <div className="mt-4 space-y-4">
                {captionServiceAvailable === false && !isStartingService && !isServiceLoading && (
                  <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
                    <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-3">
                      The captioning service is not available. Would you like to start it automatically?
                    </p>
                    <Button
                      onClick={startCaptionService}
                      disabled={isStartingService || isServiceLoading}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors text-sm"
                    >
                      Start Captioning Service
                    </Button>
                  </div>
                )}

                {(isStartingService || isServiceLoading || loadingProgress) && (
                  <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <LuLoader className="w-4 h-4 animate-spin text-blue-600" />
                        <p className="text-sm text-blue-800 dark:text-blue-200">
                          {loadingProgress?.message ||
                           (isServiceLoading ? 'Loading model into memory...' :
                            'Starting captioning service...')}
                        </p>
                      </div>

                      {loadingProgress && loadingProgress.progress > 0 && (
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs text-blue-700 dark:text-blue-300">
                            <span>Progress</span>
                            <span>{Math.round(loadingProgress.progress * 100)}%</span>
                          </div>
                          <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${loadingProgress.progress * 100}%` }}
                            />
                          </div>
                          {loadingProgress.current_file && (
                            <p className="text-xs text-blue-600 dark:text-blue-400">
                              {loadingProgress.current_file}
                            </p>
                          )}
                        </div>
                      )}

                      {loadingProgress?.status === 'downloading' && (
                        <p className="text-xs text-blue-600 dark:text-blue-400">
                          Downloading model files... This may take several minutes on first use.
                        </p>
                      )}

                      {loadingProgress?.status === 'loading' && (
                        <p className="text-xs text-blue-600 dark:text-blue-400">
                          Loading model into memory... Almost ready!
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {captionServiceAvailable === true && (
                  <>
                    {/* Caption Style Selection */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Caption Style
                      </label>
                      <select
                        value={captionStyle}
                        onChange={(e) => setCaptionStyle(e.target.value)}
                        disabled={isCaptioning || !captionServiceAvailable}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                      >
                        {availableStyles.length > 0 ? (
                          availableStyles.map(style => (
                            <option key={style} value={style}>
                              {style.charAt(0).toUpperCase() + style.slice(1).replace('_', ' ')}
                            </option>
                          ))
                        ) : (
                          <>
                            <option value="descriptive">Descriptive</option>
                            <option value="casual">Casual</option>
                            <option value="detailed">Detailed</option>
                            <option value="straightforward">Straightforward</option>
                            <option value="stable_diffusion">Stable Diffusion</option>
                            <option value="booru">Booru Tags</option>
                          </>
                        )}
                      </select>
                    </div>

                    {/* Custom Prompt */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Custom Prompt (Optional)
                      </label>
                      <textarea
                        value={customPrompt}
                        onChange={(e) => setCustomPrompt(e.target.value)}
                        disabled={isCaptioning || !captionServiceAvailable}
                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                        rows={3}
                        placeholder="Enter a custom prompt to override the selected style..."
                      />
                    </div>

                    {/* Options */}
                    <div className="flex items-center gap-4">
                      <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                        <input
                          type="checkbox"
                          checked={overwriteExisting}
                          onChange={(e) => setOverwriteExisting(e.target.checked)}
                          disabled={isCaptioning || !captionServiceAvailable}
                          className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500 disabled:cursor-not-allowed"
                        />
                        Overwrite existing captions
                      </label>
                    </div>

                    {/* Progress */}
                    {captionProgress && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
                          <span>Progress: {captionProgress.completed} / {captionProgress.total}</span>
                          <span>{Math.round((captionProgress.completed / captionProgress.total) * 100)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${(captionProgress.completed / captionProgress.total) * 100}%` }}
                          />
                        </div>
                        {captionProgress.current && (
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Currently processing: {captionProgress.current}
                          </p>
                        )}
                      </div>
                    )}

                    {/* Action Button */}
                    <div className="flex gap-3 items-center">
                      <Button
                        onClick={handleCaption}
                        disabled={isCaptioning || !captionServiceAvailable}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors flex items-center gap-2"
                      >
                        {isCaptioning ? (
                          <>
                            <LuLoader className="w-4 h-4 animate-spin" />
                            Captioning...
                          </>
                        ) : (
                          <>
                            <LuWand className="w-4 h-4" />
                            Generate Captions
                          </>
                        )}
                      </Button>

                      {captionServiceAvailable === false && !isStartingService && (
                        <Button
                          onClick={checkCaptionService}
                          className="px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-md font-medium transition-colors text-sm"
                        >
                          Retry Connection
                        </Button>
                      )}
                    </div>
                  </>
                )}

                <p className="text-sm text-gray-600 dark:text-gray-400 mt-4">
                  Auto-captioning uses JoyCaption to generate descriptive captions for your images.
                  Captions will be saved as .txt files alongside each image.
                </p>
              </div>
            )}
          </div>
        )}

        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {imgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                imageUrl={img.img_path}
                onDelete={() => refreshImageList(datasetName)}
              />
            ))}
          </div>
        )}
      </MainContent>
      <AddImagesModal />
      <FullscreenDropOverlay
        datasetName={datasetName}
        onComplete={() => refreshImageList(datasetName)}
      />
    </>
  );
}
