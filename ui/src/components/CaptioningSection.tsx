'use client';

import { useState, useEffect } from 'react';
import { LuWand, LuChevronDown, LuChevronRight, LuLoader, LuCheck, LuX } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { apiClient } from '@/utils/api';

// JoyCaption constants
const CAPTION_TYPES = [
  'Descriptive',
  'Descriptive (Casual)',
  'Straightforward',
  'Stable Diffusion Prompt',
  'MidJourney',
  'Booru tag list',
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
  '20',
  '30',
  '40',
  '50',
  '60',
  '70',
  '80',
  '90',
  '100',
  '120',
  '140',
  '160',
  '180',
  '200',
  '220',
  '240',
  '260'
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

interface ImageItem {
  img_path: string;
  caption_path?: string;
}

interface CaptioningSectionProps {
  datasetName: string;
  imgList: ImageItem[];
  isExpanded: boolean;
  onToggleExpanded: () => void;
  onCaptionComplete: () => void;
}

export default function CaptioningSection({
  datasetName,
  imgList,
  isExpanded,
  onToggleExpanded,
  onCaptionComplete,
}: CaptioningSectionProps) {
  // Service state
  const [captionServiceAvailable, setCaptionServiceAvailable] = useState<boolean | null>(null);
  const [isStartingService, setIsStartingService] = useState(false);
  const [isServiceLoading, setIsServiceLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState<{
    progress: number;
    message: string;
  } | null>(null);

  // Captioning state
  const [isCaptioning, setIsCaptioning] = useState(false);
  const [captionProgress, setCaptionProgress] = useState<{
    total: number;
    completed: number;
    current?: string;
  } | null>(null);

  // Caption options
  const [captionStyle, setCaptionStyle] = useState('Descriptive');
  const [customPrompt, setCustomPrompt] = useState('');
  const [overwriteExisting, setOverwriteExisting] = useState(false);

  // Advanced captioning options
  const [captionLength, setCaptionLength] = useState('long');
  const [extraOptions, setExtraOptions] = useState<string[]>([]);
  const [personName, setPersonName] = useState('');
  const [temperature, setTemperature] = useState(0.6);
  const [topP, setTopP] = useState(0.9);
  const [maxTokens, setMaxTokens] = useState(256);

  // Status message
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

  // Check caption service availability on component mount
  useEffect(() => {
    checkCaptionService();
  }, []);

  const checkCaptionService = async () => {
    try {
      const response = await apiClient.get('/api/datasets/caption/health');
      setCaptionServiceAvailable(response.data.available);
    } catch (error) {
      setCaptionServiceAvailable(false);
    }
  };

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

  const startCaptionService = async () => {
    setIsStartingService(true);
    setIsServiceLoading(false);
    setLoadingProgress(null);

    try {
      const response = await apiClient.post('/api/datasets/caption/start');
      if (response.data.success) {
        setIsServiceLoading(true);
        pollProgress();
      } else {
        console.error('Failed to start caption service:', response.data.error);
        setCaptionServiceAvailable(false);
      }
    } catch (error) {
      console.error('Error starting caption service:', error);
      setCaptionServiceAvailable(false);
    } finally {
      setIsStartingService(false);
    }
  };

  const pollProgress = async () => {
    const maxAttempts = 120; // 2 minutes max
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await apiClient.get('/api/datasets/caption/progress');
        const data = response.data;

        if (data.ready) {
          setLoadingProgress(null);
          setIsServiceLoading(false);
          setCaptionServiceAvailable(true);
          return;
        }

        if (data.progress !== undefined) {
          setLoadingProgress({
            progress: data.progress,
            message: data.message || 'Loading model...'
          });
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          setIsServiceLoading(false);
          setCaptionServiceAvailable(false);
          setLoadingProgress(null);
        }
      } catch (error) {
        console.error('Error polling progress:', error);
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          setIsServiceLoading(false);
          setCaptionServiceAvailable(false);
          setLoadingProgress(null);
        }
      }
    };

    poll();
  };

  const handleCaption = async () => {
    if (imgList.length === 0) {
      alert('No images to caption');
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

          // Small delay to make progress visible (for testing)
          await new Promise(resolve => setTimeout(resolve, 500));
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
      onCaptionComplete();
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
        <LuWand className="w-5 h-5 text-purple-600 dark:text-purple-400" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Auto-Captioning</h2>
        {captionServiceAvailable === false && !isStartingService && !isServiceLoading && (
          <span className="text-sm text-red-600 dark:text-red-400 ml-2">(Service Unavailable)</span>
        )}
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Service Status and Controls */}
          {captionServiceAvailable === null && (
            <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-md">
              <p className="text-sm text-gray-600 dark:text-gray-400">Checking captioning service...</p>
            </div>
          )}

          {captionServiceAvailable === false && !isStartingService && !isServiceLoading && (
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
              <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-2">
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
                      ></div>
                    </div>
                  </div>
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
                      <option key={type} value={type}>{type}</option>
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
                      <option key={length} value={length}>{length}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Person/Character Name Input - Only show if name option is selected */}
              {extraOptions.includes('If there is a person/character in the image you must refer to them as {name}.') && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Person / Character Name
                  </label>
                  <input
                    type="text"
                    value={personName}
                    onChange={(e) => setPersonName(e.target.value)}
                    disabled={isCaptioning || !captionServiceAvailable}
                    placeholder="Enter the name to use for the person/character"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed"
                  />
                </div>
              )}

              {/* Extra Options */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Extra Options
                </label>
                <div className="max-h-32 overflow-y-auto border border-gray-300 dark:border-gray-600 rounded-md p-2 bg-white dark:bg-gray-700">
                  <div className="space-y-1">
                    {EXTRA_OPTIONS.map((option, index) => (
                      <label key={index} className="flex items-start gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={extraOptions.includes(option)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setExtraOptions([...extraOptions, option]);
                            } else {
                              setExtraOptions(extraOptions.filter(o => o !== option));
                            }
                            clearCurrentSetting();
                          }}
                          disabled={isCaptioning || !captionServiceAvailable}
                          className="mt-0.5 rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500 disabled:cursor-not-allowed"
                        />
                        <span className="text-gray-700 dark:text-gray-300 leading-tight">{option}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              {/* Generation Settings */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Generation Settings
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                      Temperature: {temperature}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => {
                        setTemperature(parseFloat(e.target.value));
                        clearCurrentSetting();
                      }}
                      disabled={isCaptioning || !captionServiceAvailable}
                      className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                      Top-p: {topP}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={topP}
                      onChange={(e) => {
                        setTopP(parseFloat(e.target.value));
                        clearCurrentSetting();
                      }}
                      disabled={isCaptioning || !captionServiceAvailable}
                      className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                      Max Tokens: {maxTokens}
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="1000"
                      step="50"
                      value={maxTokens}
                      onChange={(e) => {
                        setMaxTokens(parseInt(e.target.value));
                        clearCurrentSetting();
                      }}
                      disabled={isCaptioning || !captionServiceAvailable}
                      className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
                    />
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
                  placeholder="Enter a custom prompt to override the default caption style..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed resize-vertical"
                />
              </div>

              {/* Options */}
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={overwriteExisting}
                    onChange={(e) => setOverwriteExisting(e.target.checked)}
                    disabled={isCaptioning || !captionServiceAvailable}
                    className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500 disabled:cursor-not-allowed"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">Overwrite existing captions</span>
                </label>
              </div>

              {/* Progress Display */}
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
                    ></div>
                  </div>
                  {captionProgress.current && (
                    <p className="text-xs text-gray-500 dark:text-gray-400">{captionProgress.current}</p>
                  )}
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button
                  onClick={handleCaption}
                  disabled={isCaptioning || !captionServiceAvailable || imgList.length === 0}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-md font-medium transition-colors"
                >
                  {isCaptioning ? 'Captioning...' : `Caption ${imgList.length} Images`}
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
            </>
          )}
        </div>
      )}
    </div>
  );
}
