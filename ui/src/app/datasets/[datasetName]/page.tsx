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

// JoyCaption constants
const CAPTION_TYPES = [
  'Descriptive',
  'Descriptive (Casual)',
  'Straightforward',
  'Stable Diffusion Prompt',
  'MidJourney',
  'Danbooru tag list',
  'e621 tag list',
  'Rule34 tag list',
  'Booru-like tag list',
  'Art Critic',
  'Product Listing',
  'Social Media Post'
];

const CAPTION_LENGTHS = [
  'any',
  'very short',
  'short',
  'medium-length',
  'long',
  'very long',
  ...Array.from({ length: 25 }, (_, i) => String((i + 2) * 10)) // 20, 30, 40, ..., 260
];

const NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}.";

const EXTRA_OPTIONS = [
  NAME_OPTION,
  "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
  "Include information about lighting.",
  "Include information about camera angle.",
  "Include information about whether there is a watermark or not.",
  "Include information about whether there are JPEG artifacts or not.",
  "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
  "Do NOT include anything sexual; keep it PG.",
  "Do NOT mention the image's resolution.",
  "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
  "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
  "Do NOT mention any text that is in the image.",
  "Specify the depth of field and whether the background is in focus or blurred.",
  "If applicable, mention the likely use of artificial or natural lighting sources.",
  "Do NOT use any ambiguous language.",
  "Include whether the image is sfw, suggestive, or nsfw.",
  "ONLY describe the most important elements of the image.",
  "If it is a work of art, do not include the artist's name or the title of the work.",
  "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
  "Use vulgar slang and profanity, such as (but not limited to) \"fucking,\" \"slut,\" \"cock,\" etc.",
  "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
  "Include information about the ages of any people/characters when applicable.",
  "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
  "Do not mention the mood/feeling/etc of the image.",
  "Explicitly specify the vantage height (eye-level, low-angle worm's-eye, bird's-eye, drone, rooftop, etc.).",
  "If there is a watermark, you must mention it.",
  "Your response will be used by a text-to-image model, so avoid useless meta phrases like \"This image shows…\", \"You are looking at...\", etc."
];

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [renameText, setRenameText] = useState(datasetName);
  const [isRenaming, setIsRenaming] = useState(false);

  // Captioning state
  const [captionStyle, setCaptionStyle] = useState('Descriptive');
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

  // Status messages
  const [renameStatus, setRenameStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const [captionStatus, setCaptionStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);

  // Saved captioning settings
  const [savedSettings, setSavedSettings] = useState<{
    [key: string]: {
      captionType: string;
      captionLength: string;
      extraOptions: string[];
      temperature: number;
      topP: number;
      maxTokens: number;
    };
  }>({});
  const [currentSettingName, setCurrentSettingName] = useState('');

  // Load saved settings from localStorage on component mount
  useEffect(() => {
    const saved = localStorage.getItem('captionSettings');
    if (saved) {
      try {
        setSavedSettings(JSON.parse(saved));
      } catch (error) {
        console.error('Failed to load saved caption settings:', error);
      }
    } else {
      // Add some default presets if none exist
      const defaultPresets = {
        'Quick Descriptive': {
          captionType: 'Descriptive',
          captionLength: 'short',
          extraOptions: [],
          temperature: 0.6,
          topP: 0.9,
          maxTokens: 256,
        },
        'Detailed Analysis': {
          captionType: 'Descriptive',
          captionLength: 'long',
          extraOptions: ['Include information about lighting.', 'Include information about camera angle.'],
          temperature: 0.7,
          topP: 0.9,
          maxTokens: 512,
        },
        'Art Critic': {
          captionType: 'Art Critic',
          captionLength: 'medium-length',
          extraOptions: ['Include information about lighting.', 'Include information about camera angle.'],
          temperature: 0.8,
          topP: 0.95,
          maxTokens: 400,
        },
        'Stable Diffusion': {
          captionType: 'Stable Diffusion Prompt',
          captionLength: 'short',
          extraOptions: ['Your response will be used by a text-to-image model, so avoid useless meta phrases like "This image shows…", "You are looking at...", etc.'],
          temperature: 0.7,
          topP: 0.9,
          maxTokens: 200,
        },
      };
      setSavedSettings(defaultPresets);
      localStorage.setItem('captionSettings', JSON.stringify(defaultPresets));
    }
  }, []);

  // Save current settings
  const saveCurrentSettings = (name: string) => {
    if (!name.trim()) return;

    const settings = {
      captionType: captionStyle,
      captionLength: captionLength,
      extraOptions: extraOptions,
      temperature: temperature,
      topP: topP,
      maxTokens: maxTokens,
    };

    const newSavedSettings = { ...savedSettings, [name]: settings };
    setSavedSettings(newSavedSettings);
    localStorage.setItem('captionSettings', JSON.stringify(newSavedSettings));
    setCurrentSettingName(name);
  };

  // Load settings by name
  const loadSettings = (name: string) => {
    const settings = savedSettings[name];
    if (settings) {
      setCaptionStyle(settings.captionType);
      setCaptionLength(settings.captionLength);
      setExtraOptions(settings.extraOptions);
      setTemperature(settings.temperature);
      setTopP(settings.topP);
      setMaxTokens(settings.maxTokens);
      setCurrentSettingName(name);
    }
  };

  // Handle settings dropdown/input change
  const handleSettingsChange = (value: string) => {
    setCurrentSettingName(value);
    if (savedSettings[value]) {
      loadSettings(value);
    }
  };

  // Delete a saved setting
  const deleteSetting = (name: string) => {
    const newSavedSettings = { ...savedSettings };
    delete newSavedSettings[name];
    setSavedSettings(newSavedSettings);
    localStorage.setItem('captionSettings', JSON.stringify(newSavedSettings));
    if (currentSettingName === name) {
      setCurrentSettingName('');
    }
  };

  // Clear current setting name when manually changing options
  const clearCurrentSetting = () => {
    if (currentSettingName && savedSettings[currentSettingName]) {
      setCurrentSettingName('');
    }
  };

  // Advanced captioning options
  const [captionLength, setCaptionLength] = useState('long');
  const [personName, setPersonName] = useState('');
  const [extraOptions, setExtraOptions] = useState<string[]>([]);
  const [temperature, setTemperature] = useState(0.6);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(512);

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
    setCaptionStatus(null); // Clear previous status
    setCaptionProgress({ total: imgList.length, completed: 0 });

    // Save current settings if a name is provided and it's not already saved
    if (currentSettingName.trim() && !savedSettings[currentSettingName]) {
      saveCurrentSettings(currentSettingName);
    }

    try {
      // Process images one by one for real-time progress updates
      const results = [];
      let successful = 0;
      let failed = 0;
      let totalTime = 0;

      for (let i = 0; i < imgList.length; i++) {
        const img = imgList[i];
        setCaptionProgress({
          total: imgList.length,
          completed: i,
          current: `Processing ${img.img_path.split('/').pop()}...` // Show just the filename
        });

        try {
          const response = await apiClient.post('/api/datasets/caption', {
            datasetName: datasetName,
            imagePaths: [img.img_path], // Process single image
            // Legacy style parameter for backward compatibility
            style: captionStyle,
            prompt: customPrompt.trim() || undefined,
            // New JoyCaption parameters
            captionType: captionStyle,
            captionLength: captionLength,
            extraOptions: extraOptions,
            nameInput: personName.trim() || undefined,
            // Generation parameters
            temperature: temperature,
            topP: topP,
            maxNewTokens: maxTokens,
            // Options
            overwriteExisting: overwriteExisting,
            saveToFile: true,
          });

          if (response.data.success && response.data.results && response.data.results.length > 0) {
            const result = response.data.results[0];
            results.push(result);
            if (result.success) {
              successful++;
              totalTime += result.generationTime || 0;
            } else {
              failed++;
            }
          } else {
            failed++;
            results.push({
              imagePath: img.img_path,
              success: false,
              error: response.data.error || 'Unknown error'
            });
          }
        } catch (imageError: any) {
          console.error(`Error processing ${img.img_path}:`, imageError);
          failed++;
          results.push({
            imagePath: img.img_path,
            success: false,
            error: imageError.response?.data?.error || imageError.message || 'Unknown error'
          });
        }
      }

      // Update final progress
      setCaptionProgress({
        total: imgList.length,
        completed: imgList.length,
        current: 'Captioning completed!'
      });

      // Show final status
      if (successful > 0 || failed === 0) {
        setCaptionStatus({
          type: 'success',
          message: `Captioning completed! Successful: ${successful}, Failed: ${failed}, Total time: ${totalTime.toFixed(1)}s`
        });
      } else {
        setCaptionStatus({
          type: 'error',
          message: `Captioning failed! Successful: ${successful}, Failed: ${failed}`
        });
      }

      // Refresh the image list to show any changes
      refreshImageList(datasetName);
    } catch (error: any) {
      console.error('Caption error:', error);
      setCaptionStatus({
        type: 'error',
        message: 'Failed to caption images: ' + (error.message || 'Unknown error')
      });
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
        // Refresh the image list to show the renamed files
        refreshImageList(datasetName);
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
                    {/* Saved Settings */}
                    <div className="mb-4">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Caption Settings Preset
                      </label>
                      <div className="flex gap-2">
                        <div className="flex-1">
                          <input
                            list="caption-settings"
                            value={currentSettingName}
                            onChange={(e) => handleSettingsChange(e.target.value)}
                            placeholder="Select existing preset or type new name to save..."
                            disabled={isCaptioning || !captionServiceAvailable}
                            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                          />
                          <datalist id="caption-settings">
                            {Object.keys(savedSettings).map(name => (
                              <option key={name} value={name} />
                            ))}
                          </datalist>
                        </div>
                        {currentSettingName.trim() && !savedSettings[currentSettingName] && (
                          <Button
                            onClick={() => saveCurrentSettings(currentSettingName)}
                            disabled={isCaptioning || !captionServiceAvailable}
                            className="px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors text-sm whitespace-nowrap"
                          >
                            Save Preset
                          </Button>
                        )}
                      </div>
                      {Object.keys(savedSettings).length > 0 && (
                        <div className="mt-2">
                          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                            Saved presets:
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {Object.keys(savedSettings).map(name => (
                              <div key={name} className="flex items-center gap-1 bg-gray-100 dark:bg-gray-700 rounded-md px-2 py-1">
                                <button
                                  onClick={() => handleSettingsChange(name)}
                                  disabled={isCaptioning || !captionServiceAvailable}
                                  className="text-xs text-gray-700 dark:text-gray-300 hover:text-purple-600 dark:hover:text-purple-400 disabled:cursor-not-allowed"
                                >
                                  {name}
                                </button>
                                <button
                                  onClick={() => deleteSetting(name)}
                                  disabled={isCaptioning || !captionServiceAvailable}
                                  className="text-xs text-red-500 hover:text-red-700 disabled:cursor-not-allowed ml-1"
                                  title="Delete preset"
                                >
                                  <LuX className="w-3 h-3" />
                                </button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Caption Type Selection */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Caption Type
                        </label>
                        <select
                          value={captionStyle}
                          onChange={(e) => {
                            setCaptionStyle(e.target.value);
                            clearCurrentSetting();
                          }}
                          disabled={isCaptioning || !captionServiceAvailable}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                        >
                          {CAPTION_TYPES.map(type => (
                            <option key={type} value={type}>
                              {type}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Caption Length
                        </label>
                        <select
                          value={captionLength}
                          onChange={(e) => {
                            setCaptionLength(e.target.value);
                            clearCurrentSetting();
                          }}
                          disabled={isCaptioning || !captionServiceAvailable}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                        >
                          {CAPTION_LENGTHS.map(length => (
                            <option key={length} value={length}>
                              {length}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Person Name Input - Show only when name option is selected */}
                    {extraOptions.includes(NAME_OPTION) && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                          Person / Character Name
                        </label>
                        <input
                          type="text"
                          value={personName}
                          onChange={(e) => setPersonName(e.target.value)}
                          disabled={isCaptioning || !captionServiceAvailable}
                          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                          placeholder="Enter the name to use for people/characters in the image"
                        />
                      </div>
                    )}

                    {/* Extra Options */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                        Extra Options
                      </label>
                      <div className="max-h-48 overflow-y-auto border border-gray-300 dark:border-gray-600 rounded-md p-3 bg-white dark:bg-gray-700">
                        <div className="space-y-2">
                          {EXTRA_OPTIONS.map((option, index) => (
                            <label key={index} className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={extraOptions.includes(option)}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setExtraOptions([...extraOptions, option]);
                                  } else {
                                    setExtraOptions(extraOptions.filter(o => o !== option));
                                  }
                                }}
                                disabled={isCaptioning || !captionServiceAvailable}
                                className="mt-0.5 rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500 disabled:cursor-not-allowed"
                              />
                              <span className="leading-tight">{option}</span>
                            </label>
                          ))}
                        </div>
                      </div>
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

                    {/* Generation Settings */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                        Generation Settings
                      </label>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                            Temperature ({temperature})
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.05"
                            value={temperature}
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            disabled={isCaptioning || !captionServiceAvailable}
                            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                          />
                          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            Higher = more random
                          </div>
                        </div>

                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                            Top-p ({topP})
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={topP}
                            onChange={(e) => setTopP(parseFloat(e.target.value))}
                            disabled={isCaptioning || !captionServiceAvailable}
                            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                          />
                          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            Nucleus sampling
                          </div>
                        </div>

                        <div>
                          <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                            Max Tokens ({maxTokens})
                          </label>
                          <input
                            type="range"
                            min="1"
                            max="2048"
                            step="1"
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                            disabled={isCaptioning || !captionServiceAvailable}
                            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                          />
                          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            Maximum length
                          </div>
                        </div>
                      </div>
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

            {/* Caption Status Message */}
            {captionStatus && (
              <div className={`mt-4 p-3 rounded-md flex items-center gap-2 ${
                captionStatus.type === 'success'
                  ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800'
                  : 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-800'
              }`}>
                {captionStatus.type === 'success' ? (
                  <LuCheck className="w-5 h-5 flex-shrink-0" />
                ) : (
                  <LuX className="w-5 h-5 flex-shrink-0" />
                )}
                <span className="text-sm font-medium flex-1">{captionStatus.message}</span>
                <button
                  onClick={() => setCaptionStatus(null)}
                  className="text-current hover:opacity-70 transition-opacity"
                >
                  <LuX className="w-4 h-4" />
                </button>
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
