'use client';

import { useState, useEffect, useMemo } from 'react';
import { JobConfig } from '@/types';
import { Button } from '@headlessui/react';
import { FaChevronRight, FaChevronLeft, FaCheck } from 'react-icons/fa';
import Card from '@/components/Card';
import { TextInput, NumberInput, SelectInput, Checkbox, FormGroup } from '@/components/formInputs';
import { modelArchs } from '../options';
import { handleModelArchChange } from '../utils';
import { apiClient } from '@/utils/api';

// Wizard components
import PreflightModal from './components/PreflightModal';
import AdvisorPanel from './components/AdvisorPanel';
import SummaryHeader from './components/SummaryHeader';
import { StepRenderer } from './components/StepRenderer';
import { setNestedValue as setNestedConfigValue } from './fieldConfig';

// Types and utilities
import {
  SystemProfile,
  UserIntent,
  DatasetInfo,
  WizardStep,
  ConfigSummary,
  AdvisorMessage,
  PerformancePrediction
} from './utils/types';
import {
  generateSmartDefaults,
  estimateVRAMUsage,
  estimateTrainingTime,
  estimateDiskSpace,
  handleUnifiedMemory,
  calculateRegularizationMemoryCost
} from './utils/smartDefaults';

interface Props {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  onExit: () => void;
}

// Define all wizard steps
const wizardSteps: WizardStep[] = [
  { id: 'model', title: 'Model', description: 'Select base model architecture' },
  { id: 'quantization', title: 'Quantization', description: 'Configure model precision', isAdaptive: true },
  { id: 'target', title: 'Target', description: 'Configure LoRA/LoKr settings' },
  { id: 'dataset', title: 'Dataset', description: 'Analyze and configure your dataset' },
  { id: 'resolution', title: 'Resolution', description: 'Set training resolution and augmentation' },
  { id: 'memory', title: 'Memory', description: 'Batch size and caching configuration' },
  { id: 'optimizer', title: 'Optimizer', description: 'Learning rate and optimizer settings' },
  { id: 'advanced', title: 'Advanced', description: 'Noise scheduler, loss function, and CFG settings' },
  { id: 'regularization', title: 'Regularization', description: 'Prevent overfitting', isOptional: true },
  { id: 'training', title: 'Training', description: 'Steps and timestep configuration' },
  { id: 'sampling', title: 'Sampling', description: 'Configure preview generation' },
  { id: 'save', title: 'Save', description: 'Checkpoint and save settings' },
  { id: 'logging', title: 'Logging', description: 'W&B integration and logging settings' },
  { id: 'monitoring', title: 'Monitoring', description: 'Performance analysis and optimization' },
  { id: 'review', title: 'Review', description: 'Review and submit configuration' }
];

export default function ComprehensiveWizard({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  onExit
}: Props) {
  // Pre-flight state
  const [showPreflight, setShowPreflight] = useState(true);
  const [systemProfile, setSystemProfile] = useState<SystemProfile | null>(null);
  const [userIntent, setUserIntent] = useState<UserIntent | null>(null);

  // Wizard navigation
  const [currentStep, setCurrentStep] = useState(0);

  // Dataset analysis
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Advisor messages
  const [advisorMessages, setAdvisorMessages] = useState<AdvisorMessage[]>([]);

  // Custom resolution mode
  const [useCustomResolution, setUseCustomResolution] = useState(false);

  const currentStepDef = wizardSteps[currentStep];

  // Filter to image models
  const imageModels = useMemo(() => modelArchs.filter(m => m.group === 'image'), []);

  const selectedModelArch = useMemo(() => {
    const arch = jobConfig.config.process[0]?.model?.arch;
    return arch ? modelArchs.find(a => a.name === arch) : undefined;
  }, [jobConfig.config.process[0]?.model?.arch]);

  // Get model-specific resolution info
  const getModelResolutionInfo = (archName: string) => {
    const info: Record<string, { native: number; recommended: number[]; maxSupported: number; supportsAspect: boolean }> = {
      sd1: { native: 512, recommended: [512], maxSupported: 768, supportsAspect: false },
      sd15: { native: 512, recommended: [512], maxSupported: 768, supportsAspect: false },
      sd2: { native: 768, recommended: [512, 768], maxSupported: 1024, supportsAspect: false },
      sdxl: { native: 1024, recommended: [768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      sd3: { native: 1024, recommended: [512, 768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      flux: { native: 1024, recommended: [512, 768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      flux_kontext: { native: 1024, recommended: [512, 768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      flex1: { native: 1024, recommended: [512, 768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      flex2: { native: 1024, recommended: [512, 768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      chroma: { native: 1024, recommended: [512, 768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      lumina2: { native: 1024, recommended: [768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      qwen_image: { native: 1024, recommended: [768, 1024, 1536, 2048], maxSupported: 2048, supportsAspect: true },
      qwen_image_edit: { native: 1024, recommended: [768, 1024, 1536, 2048], maxSupported: 2048, supportsAspect: true },
      qwen_image_edit_plus: { native: 1024, recommended: [768, 1024, 1536, 2048], maxSupported: 2048, supportsAspect: true },
      hidream: { native: 1024, recommended: [768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      hidream_e1: { native: 1024, recommended: [768, 1024, 1536], maxSupported: 2048, supportsAspect: true },
      omnigen2: { native: 1024, recommended: [768, 1024, 1536], maxSupported: 2048, supportsAspect: true }
    };
    return info[archName] || { native: 1024, recommended: [512, 768, 1024], maxSupported: 2048, supportsAspect: true };
  };

  // Calculate smart resolution options based on model and dataset
  const getSmartResolutionOptions = () => {
    if (!selectedModelArch) return [];

    const modelInfo = getModelResolutionInfo(selectedModelArch.name);
    let options = [...modelInfo.recommended];

    // If dataset has high-res images, consider adding higher resolutions
    if (datasetInfo) {
      const datasetMaxRes = Math.max(datasetInfo.most_common_resolution[0], datasetInfo.most_common_resolution[1]);

      // Add 1536 if dataset supports it and model can handle it
      if (datasetMaxRes >= 1536 && modelInfo.maxSupported >= 1536 && !options.includes(1536)) {
        options.push(1536);
      }

      // Add 2048 for very high-res datasets
      if (datasetMaxRes >= 2048 && modelInfo.maxSupported >= 2048 && !options.includes(2048)) {
        options.push(2048);
      }
    }

    // Sort and deduplicate
    options = [...new Set(options)].sort((a, b) => a - b);

    return options;
  };

  // Get recommended resolution
  const getRecommendedResolution = () => {
    if (!selectedModelArch || !datasetInfo) return 1024;

    const modelInfo = getModelResolutionInfo(selectedModelArch.name);
    const datasetMaxRes = Math.max(datasetInfo.most_common_resolution[0], datasetInfo.most_common_resolution[1]);

    // If dataset has high-res images and model supports it, recommend higher resolution
    // Check from highest to lowest
    if (datasetMaxRes >= 2048 && modelInfo.maxSupported >= 2048) {
      return 2048;
    } else if (datasetMaxRes >= 1536 && modelInfo.maxSupported >= 1536) {
      return 1536;
    } else if (datasetMaxRes >= 1024 && modelInfo.maxSupported >= 1024) {
      return 1024;
    } else if (datasetMaxRes >= 768 && modelInfo.maxSupported >= 768) {
      return 768;
    }

    return modelInfo.native;
  };

  // Calculate config summary
  const configSummary = useMemo<ConfigSummary>(() => {
    const process = jobConfig.config.process[0];
    // Handle both new [width, height] format and old multi-res [512, 768, 1024] format
    const resArray = process.datasets?.[0]?.resolution;
    const resolution = resArray?.length === 2 ? resArray[0] : (resArray ? Math.max(...resArray) : undefined);
    const steps = process.train?.steps;
    const batchSize = process.train?.batch_size || 1;

    let estimatedVRAM = '~0 GB';
    if (systemProfile && resolution && selectedModelArch) {
      // Convert boolean quantize + qtype to string format for estimation
      let quantizationStr: string | null = null;
      if (process.model?.quantize === true) {
        const qtype = process.model?.qtype || 'qfloat8';
        if (qtype.startsWith('uint4') || qtype.startsWith('uint3')) {
          quantizationStr = '4bit';
        } else {
          quantizationStr = '8bit';
        }
      }

      const vram = estimateVRAMUsage(
        selectedModelArch.name,
        resolution,
        batchSize,
        quantizationStr,
        process.train?.gradient_checkpointing || false
      );
      estimatedVRAM = `~${vram.toFixed(1)} GB`;
    }

    const warnings: string[] = [];
    if (!selectedModelArch) warnings.push('Model not selected');
    if (!resolution) warnings.push('Resolution not set');
    if (!steps) warnings.push('Steps not configured');
    if (!datasetInfo) warnings.push('Dataset not analyzed');

    return {
      model: selectedModelArch?.label || '',
      resolution: resolution ? `${resolution}px` : '',
      steps: steps || null,
      estimatedVRAM,
      warnings
    };
  }, [jobConfig, systemProfile, selectedModelArch, datasetInfo]);

  // Calculate performance predictions
  const performancePrediction = useMemo<PerformancePrediction | undefined>(() => {
    if (!systemProfile || !datasetInfo) return undefined;

    const process = jobConfig.config.process[0];
    const resolution = process.datasets?.[0]?.resolution?.[0] || 1024;
    const steps = process.train?.steps || 0;
    const batchSize = process.train?.batch_size || 1;
    const saveEvery = process.save?.save_every || 500;
    const cacheToDisK = process.datasets?.[0]?.cache_latents_to_disk || false;

    if (!steps) return undefined;

    const { stepTime, totalMinutes } = estimateTrainingTime(steps, batchSize, resolution, systemProfile);
    const diskSpace = estimateDiskSpace(datasetInfo, steps, saveEvery, cacheToDisK);

    // Convert boolean quantize + qtype to string format for estimation
    let quantizationStr: string | null = null;
    if (process.model?.quantize === true) {
      const qtype = process.model?.qtype || 'qfloat8';
      if (qtype.startsWith('uint4') || qtype.startsWith('uint3')) {
        quantizationStr = '4bit';
      } else {
        quantizationStr = '8bit';
      }
    }

    const vram = selectedModelArch
      ? estimateVRAMUsage(
          selectedModelArch.name,
          resolution,
          batchSize,
          quantizationStr,
          process.train?.gradient_checkpointing || false
        )
      : 0;

    return {
      estimatedVRAM: `${vram.toFixed(1)} GB (of ${systemProfile.gpu.vramGB} GB)`,
      estimatedStepTime: `~${stepTime.toFixed(2)} seconds`,
      totalTrainingTime:
        totalMinutes < 60
          ? `~${Math.round(totalMinutes)} minutes`
          : `~${(totalMinutes / 60).toFixed(1)} hours`,
      diskSpaceNeeded: `${diskSpace.toFixed(1)} GB`,
      memoryUsage: `~${Math.min(systemProfile.memory.availableRAM, datasetInfo.total_images * 0.1 + 8).toFixed(0)} GB RAM`
    };
  }, [jobConfig, systemProfile, datasetInfo, selectedModelArch]);

  // Update advisor messages based on current configuration
  useEffect(() => {
    if (!systemProfile || !userIntent) return;

    const messages: AdvisorMessage[] = [...handleUnifiedMemory(systemProfile)];

    // Add step-specific recommendations
    if (currentStepDef.id === 'memory' && datasetInfo) {
      const process = jobConfig.config.process[0];
      const batchSize = process.train?.batch_size || 1;
      const resolution = process.datasets?.[0]?.resolution?.[0] || 1024;

      if (batchSize > 1 && resolution >= 1024 && systemProfile.gpu.vramGB < 16) {
        messages.push({
          type: 'warning',
          title: 'High VRAM Usage',
          message: `Batch size ${batchSize} at ${resolution}px may exceed your ${systemProfile.gpu.vramGB}GB VRAM. Consider enabling auto-scale batch size.`
        });
      }
    }

    // Regularization step recommendations
    if (currentStepDef.id === 'regularization') {
      const process = jobConfig.config.process[0];
      const captionDropout = process.datasets?.[0]?.caption_dropout_rate || 0.05;
      const weightDecay = process.train?.optimizer_params?.weight_decay || 0.0001;
      const gradCheckpoint = process.train?.gradient_checkpointing;
      const useEma = process.train?.ema_config?.use_ema;
      const diffPreserve = process.train?.diff_output_preservation;

      // Caption dropout recommendation
      if (userIntent.trainingType === 'person') {
        messages.push({
          type: 'tip',
          title: 'Person Training Tip',
          message: `For training people/characters, a caption dropout of 0.05-0.1 helps the model generalize facial features. Current: ${captionDropout}`
        });
      } else if (userIntent.trainingType === 'style') {
        messages.push({
          type: 'tip',
          title: 'Style Training Tip',
          message: `For style training, lower caption dropout (0.02-0.05) keeps style consistent. Current: ${captionDropout}`
        });
      }

      // Small dataset warning
      if (datasetInfo && datasetInfo.total_images < 20) {
        messages.push({
          type: 'warning',
          title: 'Small Dataset',
          message: `With only ${datasetInfo.total_images} images, consider higher weight decay (0.01) and enabling EMA to prevent overfitting.`
        });
        if (weightDecay < 0.001) {
          messages.push({
            type: 'info',
            title: 'Weight Decay Suggestion',
            message: 'Increase weight decay to 0.001-0.01 for small datasets to prevent overfitting.'
          });
        }
      }

      // Gradient checkpointing for limited VRAM
      if (!gradCheckpoint && systemProfile.gpu.vramGB < 16) {
        messages.push({
          type: 'warning',
          title: 'Enable Gradient Checkpointing',
          message: `With ${systemProfile.gpu.vramGB}GB VRAM, enable gradient checkpointing to avoid out-of-memory errors.`
        });
      }

      // EMA recommendation
      if (!useEma && userIntent.priority === 'quality') {
        messages.push({
          type: 'tip',
          title: 'Consider EMA',
          message: 'EMA smooths training noise and often produces more stable, higher-quality results. Recommended for quality-focused training.'
        });
      }

      // Differential preservation for concept training
      if (!diffPreserve && userIntent.trainingType === 'concept') {
        messages.push({
          type: 'tip',
          title: 'Output Preservation',
          message: 'For concept training, consider enabling differential output preservation to prevent the model from forgetting its general knowledge.'
        });
      }
    }

    setAdvisorMessages(messages);
  }, [systemProfile, userIntent, currentStepDef, jobConfig, datasetInfo]);

  // Handle pre-flight completion
  const handlePreflightComplete = (profile: SystemProfile, intent: UserIntent) => {
    setSystemProfile(profile);
    setUserIntent(intent);
    setShowPreflight(false);

    // Apply smart defaults if we have dataset info
    if (datasetInfo) {
      applySmartDefaults(profile, intent, datasetInfo);
    }
  };

  // Handle config changes from StepRenderer (data-driven fields)
  const handleStepRendererConfigChange = (newConfig: JobConfig) => {
    // Apply all changes from the new config
    // StepRenderer uses setNestedValue which returns a new config object
    // We need to extract all the individual path changes and apply them
    const processConfig = newConfig.config.process[0];

    // Helper to recursively set nested object values
    const setNestedObject = (obj: any, basePath: string) => {
      Object.entries(obj).forEach(([key, value]) => {
        const path = `${basePath}.${key}`;
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          setNestedObject(value, path);
        } else {
          setJobConfig(value, path);
        }
      });
    };

    // Apply model changes
    if (processConfig.model) {
      setNestedObject(processConfig.model, 'config.process[0].model');
    }

    // Apply network changes
    if (processConfig.network) {
      setNestedObject(processConfig.network, 'config.process[0].network');
    }

    // Apply train changes
    if (processConfig.train) {
      setNestedObject(processConfig.train, 'config.process[0].train');
    }

    // Apply dataset changes
    if (processConfig.datasets && processConfig.datasets[0]) {
      setNestedObject(processConfig.datasets[0], 'config.process[0].datasets[0]');
    }

    // Apply sample changes
    if (processConfig.sample) {
      setNestedObject(processConfig.sample, 'config.process[0].sample');
    }

    // Apply save changes
    if (processConfig.save) {
      setNestedObject(processConfig.save, 'config.process[0].save');
    }

    // Apply logging changes
    if (processConfig.logging) {
      setNestedObject(processConfig.logging, 'config.process[0].logging');
    }

    // Apply monitoring changes
    if ((processConfig as any).monitoring) {
      setNestedObject((processConfig as any).monitoring, 'config.process[0].monitoring');
    }
  };

  // Apply smart defaults
  const applySmartDefaults = (profile: SystemProfile, intent: UserIntent, dataset: DatasetInfo, recommendedResolution?: number) => {
    // Use recommended resolution if provided (from dataset analysis), otherwise fall back to config
    const resolution = recommendedResolution || jobConfig.config.process[0].datasets?.[0]?.resolution?.[0] || 1024;
    const modelArch = jobConfig.config.process[0].model?.arch;

    if (!modelArch) {
      console.error('[SmartDefaults] Model architecture not set - cannot calculate optimal defaults');
      return; // Don't apply defaults without knowing the model
    }

    const defaults = generateSmartDefaults(profile, intent, dataset, resolution, 'lora', modelArch);

    // Apply training defaults
    Object.entries(defaults.train).forEach(([key, value]) => {
      setJobConfig(value, `config.process[0].train.${key}`);
    });

    // Apply dataset defaults (but NOT resolution - that's set separately based on dataset analysis)
    Object.entries(defaults.dataset).forEach(([key, value]) => {
      if (key === 'resolution') {
        // Skip resolution - it's already set based on dataset analysis before this function is called
        // We don't want to overwrite the recommended resolution with a stale value
      } else {
        setJobConfig(value, `config.process[0].datasets[0].${key}`);
      }
    });

    // Apply network defaults
    Object.entries(defaults.network).forEach(([key, value]) => {
      setJobConfig(value, `config.process[0].network.${key}`);
    });

    // Apply model defaults (low_vram, etc.)
    if (defaults.model) {
      Object.entries(defaults.model).forEach(([key, value]) => {
        setJobConfig(value, `config.process[0].model.${key}`);
      });
    }

    // Enable aspect ratio bucketing by default if model supports it and dataset has non-square images
    if (selectedModelArch) {
      const modelInfo = getModelResolutionInfo(selectedModelArch.name);
      if (modelInfo.supportsAspect) {
        // Check if dataset has non-square images
        const hasNonSquareImages = dataset.most_common_resolution[0] !== dataset.most_common_resolution[1];
        // Also enable if there are multiple resolutions (likely different aspect ratios)
        const hasMultipleResolutions = Object.keys(dataset.resolutions).length > 1;

        if (hasNonSquareImages || hasMultipleResolutions) {
          setJobConfig(true, 'config.process[0].datasets[0].enable_ar_bucket');
        }
      }
    }
  };

  // Analyze dataset
  const analyzeDataset = async (path: string) => {
    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      const response = await apiClient.post('/api/dataset/analyze', { path });
      setDatasetInfo(response.data);

      // Auto-fill dataset config
      setJobConfig(path, 'config.process[0].datasets[0].folder_path');
      if (response.data.caption_ext) {
        setJobConfig(response.data.caption_ext, 'config.process[0].datasets[0].caption_ext');
      }

      // Calculate and set recommended resolution based on dataset and model
      let recommendedRes: number | undefined;
      if (selectedModelArch && response.data.most_common_resolution) {
        const modelInfo = getModelResolutionInfo(selectedModelArch.name);
        const datasetMaxRes = Math.max(
          response.data.most_common_resolution[0],
          response.data.most_common_resolution[1]
        );

        recommendedRes = modelInfo.native;
        // Check from highest to lowest resolution
        if (datasetMaxRes >= 2048 && modelInfo.maxSupported >= 2048) {
          recommendedRes = 2048;
        } else if (datasetMaxRes >= 1536 && modelInfo.maxSupported >= 1536) {
          recommendedRes = 1536;
        } else if (datasetMaxRes >= 1024 && modelInfo.maxSupported >= 1024) {
          recommendedRes = 1024;
        } else if (datasetMaxRes >= 768 && modelInfo.maxSupported >= 768) {
          recommendedRes = 768;
        }

        setJobConfig([recommendedRes, recommendedRes], 'config.process[0].datasets[0].resolution');
      }

      // Apply smart defaults if we have system profile
      // Pass the recommended resolution so batch size is calculated correctly
      if (systemProfile && userIntent) {
        applySmartDefaults(systemProfile, userIntent, response.data, recommendedRes);
      }
    } catch (error: any) {
      setAnalysisError(error.response?.data?.error || 'Failed to analyze dataset');
      setDatasetInfo(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Navigation
  const nextStep = () => {
    if (currentStep < wizardSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const goToStep = (stepIndex: number) => {
    if (stepIndex >= 0 && stepIndex < wizardSteps.length) {
      setCurrentStep(stepIndex);
    }
  };

  // Show pre-flight modal first
  if (showPreflight) {
    return <PreflightModal onComplete={handlePreflightComplete} onCancel={onExit} />;
  }

  return (
    <div className="h-full flex flex-col">
      {/* Progress Bar */}
      <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 p-4">
        <div className="flex justify-between mb-2 overflow-x-auto">
          {wizardSteps.map((s, idx) => (
            <button
              key={s.id}
              onClick={() => goToStep(idx)}
              className="flex flex-col items-center min-w-[60px] mx-1"
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center border-2 transition-colors text-sm ${
                  idx < currentStep
                    ? 'bg-green-500 border-green-500 text-white'
                    : idx === currentStep
                    ? 'bg-blue-500 border-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 border-gray-300 dark:border-gray-600'
                }`}
              >
                {idx < currentStep ? <FaCheck className="text-xs" /> : idx + 1}
              </div>
              <div className="text-xs mt-1 text-center whitespace-nowrap">{s.title}</div>
            </button>
          ))}
        </div>
        <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${((currentStep + 1) / wizardSteps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Summary Header */}
      <SummaryHeader summary={configSummary} />

      {/* Main Content Area */}
      <div className="flex-grow flex overflow-hidden">
        {/* Step Content (70%) */}
        <div className="flex-grow overflow-y-auto p-6">
          <Card className="p-6">
            <h2 className="text-2xl font-bold mb-2">{currentStepDef.title}</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-6">{currentStepDef.description}</p>

            {/* Step 1: Model Selection */}
            {currentStepDef.id === 'model' && (
              <div className="space-y-4">
                <FormGroup label="Base Model Architecture" tooltip="Select the model architecture to train on">
                  <div className="grid grid-cols-2 gap-3">
                    {imageModels.map(model => (
                      <button
                        key={model.name}
                        onClick={() => {
                          handleModelArchChange(
                            jobConfig.config.process[0].model?.arch || '',
                            model.name,
                            jobConfig,
                            setJobConfig
                          );
                        }}
                        className={`p-4 rounded-lg border-2 transition-colors text-left ${
                          jobConfig.config.process[0].model?.arch === model.name
                            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                            : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                        }`}
                      >
                        <div className="font-semibold">{model.label}</div>
                        <div className="text-xs text-gray-600 dark:text-gray-400 mt-1">{model.name}</div>
                      </button>
                    ))}
                  </div>
                </FormGroup>

                {selectedModelArch && (
                  <FormGroup label="Model Path" tooltip="Path to the model files or HuggingFace repo">
                    <TextInput
                      value={jobConfig.config.process[0].model?.name_or_path || ''}
                      onChange={value => setJobConfig(value, 'config.process[0].model.name_or_path')}
                      placeholder="e.g., black-forest-labs/FLUX.1-dev"
                    />
                  </FormGroup>
                )}
              </div>
            )}

            {/* Step 2: Quantization */}
            {currentStepDef.id === 'quantization' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="quantization"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />

              </div>
            )}

            {/* Step 3: Target Configuration */}
            {currentStepDef.id === 'target' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="target"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}

            {/* Step 4: Dataset */}
            {currentStepDef.id === 'dataset' && (
              <div className="space-y-4">
                {/* Dataset folder selection - requires special UI for folder picker */}
                <FormGroup label="Dataset Folder" tooltip="Path to your training images">
                  <div className="flex gap-2">
                    <div className="flex-1">
                      <SelectInput
                        value={jobConfig.config.process[0].datasets?.[0]?.folder_path || ''}
                        onChange={value => setJobConfig(value, 'config.process[0].datasets[0].folder_path')}
                        options={datasetOptions}
                      />
                    </div>
                    <Button
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                      onClick={() => analyzeDataset(jobConfig.config.process[0].datasets?.[0]?.folder_path || '')}
                      disabled={!jobConfig.config.process[0].datasets?.[0]?.folder_path || isAnalyzing}
                    >
                      {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                    </Button>
                  </div>
                </FormGroup>

                {/* Data-driven dataset configuration fields */}
                <StepRenderer
                  stepId="dataset"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}

            {/* Step 5: Resolution */}
            {currentStepDef.id === 'resolution' && (
              <div className="space-y-4">
                {/* Resolution Info Card */}
                {selectedModelArch && (
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-md">
                    <h3 className="font-semibold mb-2">Resolution Recommendation</h3>
                    <div className="space-y-2 text-sm">
                      <p>
                        <strong>Model:</strong> {selectedModelArch.label} (native: {getModelResolutionInfo(selectedModelArch.name).native}px)
                      </p>
                      {datasetInfo && (
                        <p>
                          <strong>Dataset:</strong> {datasetInfo.most_common_resolution[0]}x{datasetInfo.most_common_resolution[1]} (
                          {Math.round((datasetInfo.most_common_resolution[0] * datasetInfo.most_common_resolution[1]) / 1000000)}MP)
                        </p>
                      )}
                      {datasetInfo && (
                        <p className="text-blue-700 dark:text-blue-300">
                          <strong>Recommended:</strong> {getRecommendedResolution()}px
                          {getRecommendedResolution() >= 1536 && ' (high-res training enabled by your dataset)'}
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Resolution Presets */}
                <FormGroup label="Training Resolution" tooltip="Select resolution based on model and dataset capabilities">
                  {!useCustomResolution ? (
                    <div className="grid grid-cols-3 gap-3">
                      {getSmartResolutionOptions().map(res => {
                        const isSelected = jobConfig.config.process[0].datasets?.[0]?.resolution?.[0] === res;
                        const isRecommended = datasetInfo && res === getRecommendedResolution();
                        const modelInfo = selectedModelArch ? getModelResolutionInfo(selectedModelArch.name) : null;
                        const isNative = modelInfo && res === modelInfo.native;

                        let sublabel = '';
                        if (isRecommended) {
                          sublabel = 'Recommended';
                        } else if (isNative) {
                          sublabel = 'Native';
                        } else if (res >= 1536) {
                          sublabel = 'High-res';
                        } else if (res <= 768) {
                          sublabel = 'Faster';
                        }

                        return (
                          <button
                            key={res}
                            onClick={() => {
                              setJobConfig([res, res], 'config.process[0].datasets[0].resolution');
                              setUseCustomResolution(false);
                            }}
                            className={`p-3 rounded-lg border-2 transition-colors text-center ${
                              isSelected
                                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                                : isRecommended
                                ? 'border-green-400 bg-green-50 dark:bg-green-900/10 hover:border-green-500'
                                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                            }`}
                          >
                            <div className="font-semibold">{res}px</div>
                            {sublabel && (
                              <div className={`text-xs mt-1 ${
                                isRecommended ? 'text-green-600 dark:text-green-400 font-medium' : 'text-gray-500 dark:text-gray-400'
                              }`}>
                                {sublabel}
                              </div>
                            )}
                          </button>
                        );
                      })}

                      {/* Custom button */}
                      <button
                        onClick={() => setUseCustomResolution(true)}
                        className="p-3 rounded-lg border-2 transition-colors text-center border-gray-300 dark:border-gray-600 hover:border-gray-400"
                      >
                        <div className="font-semibold">Custom</div>
                        <div className="text-xs mt-1 text-gray-500 dark:text-gray-400">
                          Enter value
                        </div>
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex items-center gap-3">
                        <NumberInput
                          value={jobConfig.config.process[0].datasets?.[0]?.resolution?.[0] ?? 1024}
                          onChange={value => {
                            setJobConfig([value, value], 'config.process[0].datasets[0].resolution');
                          }}
                          min={256}
                          max={2048}
                          step={64}
                        />
                        <span className="text-sm text-gray-600 dark:text-gray-400">pixels</span>
                      </div>
                      <button
                        onClick={() => setUseCustomResolution(false)}
                        className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
                      >
                        ← Back to presets
                      </button>
                    </div>
                  )}
                </FormGroup>

                {/* VRAM Warning for high resolutions */}
                {systemProfile && jobConfig.config.process[0].datasets?.[0]?.resolution?.[0] >= 1536 && (() => {
                  // Use unified memory if available, otherwise use GPU VRAM
                  // If GPU VRAM is 0, it's likely unified memory or CPU-only, so use system RAM
                  let effectiveVRAM: number;
                  let memoryType: string;

                  if (systemProfile.gpu.isUnifiedMemory || systemProfile.gpu.vramGB === 0) {
                    // Unified memory or no discrete GPU - use system RAM
                    effectiveVRAM = systemProfile.memory.unifiedMemory || systemProfile.memory.totalRAM;
                    memoryType = systemProfile.gpu.isUnifiedMemory ? 'unified memory' : 'system memory';
                  } else {
                    effectiveVRAM = systemProfile.gpu.vramGB;
                    memoryType = 'GPU';
                  }

                  if (effectiveVRAM < 24) {
                    return (
                      <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700 rounded-md">
                        <p className="text-yellow-800 dark:text-yellow-300 text-sm">
                          <strong>Warning:</strong> Training at {jobConfig.config.process[0].datasets?.[0]?.resolution?.[0]}px requires significant memory.
                          Your {effectiveVRAM}GB {memoryType} may struggle. Consider reducing batch size or enabling gradient checkpointing.
                        </p>
                      </div>
                    );
                  }
                  return null;
                })()}

                <FormGroup label="Enable Bucket Sampling" tooltip="Allow different aspect ratios during training">
                  <Checkbox
                    checked={!!jobConfig.config.process[0].datasets?.[0]?.enable_ar_bucket}
                    onChange={value => setJobConfig(value, 'config.process[0].datasets[0].enable_ar_bucket')}
                    label="Use aspect ratio bucketing (recommended for non-square images)"
                  />
                </FormGroup>

                {/* Aspect ratio info */}
                {selectedModelArch && getModelResolutionInfo(selectedModelArch.name).supportsAspect && (
                  <div className="text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 p-3 rounded">
                    <p>
                      {selectedModelArch.label} supports aspect ratio bucketing, which means your LoRA will learn from images of different shapes without cropping.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Step 6: Memory & Batch */}
            {currentStepDef.id === 'memory' && (
              <div className="space-y-4">
                <FormGroup label="Auto-Scale Batch Size" tooltip="Automatically find optimal batch size">
                  <Checkbox
                    checked={!!jobConfig.config.process[0].train?.auto_scale_batch_size}
                    onChange={value => setJobConfig(value, 'config.process[0].train.auto_scale_batch_size')}
                    label="Enable auto-scaling (recommended)"
                  />
                </FormGroup>

                {jobConfig.config.process[0].train?.auto_scale_batch_size ? (
                  <div className="grid grid-cols-3 gap-4">
                    <FormGroup label="Initial" tooltip="Starting batch size for auto-scaling. The system will begin with this value and adjust based on available memory. For unified memory systems (Apple Silicon), available pool is shared between system and GPU.">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.batch_size ?? 1}
                        onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                        min={1}
                        max={32}
                      />
                    </FormGroup>
                    <FormGroup label="Min" tooltip="Minimum batch size to use. Auto-scaling won't go below this value. Lower values use less memory but train slower. Set to 1 for memory-constrained systems or large models like Flux.">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.min_batch_size ?? 1}
                        onChange={value => setJobConfig(value, 'config.process[0].train.min_batch_size')}
                        min={1}
                        max={32}
                      />
                    </FormGroup>
                    <FormGroup label="Max" tooltip="Maximum batch size to attempt. Higher values require more memory but improve training stability. Each doubling roughly doubles memory usage. Recommended max: SD1.5=8, SDXL=4, Flux=2 (on 24GB VRAM).">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.max_batch_size ?? 8}
                        onChange={value => setJobConfig(value, 'config.process[0].train.max_batch_size')}
                        min={1}
                        max={32}
                      />
                    </FormGroup>
                  </div>
                ) : (
                  <FormGroup label="Batch Size" tooltip="Number of images processed per training step. Higher = faster training and better gradients but more memory usage. SD1.5: ~6GB base, +2GB per batch. SDXL: ~12GB base, +4GB per batch. Flux: ~16GB base, +6GB per batch. For unified memory systems (Apple Silicon), this uses shared RAM/VRAM pool.">
                    <NumberInput
                      value={jobConfig.config.process[0].train?.batch_size ?? 1}
                      onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                      min={1}
                      max={32}
                    />
                  </FormGroup>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <FormGroup label="Workers" tooltip="Number of data loading workers">
                    <NumberInput
                      value={jobConfig.config.process[0].datasets?.[0]?.num_workers ?? 4}
                      onChange={value => setJobConfig(value, 'config.process[0].datasets[0].num_workers')}
                      min={0}
                      max={16}
                    />
                  </FormGroup>
                  <FormGroup label="GPU Prefetch" tooltip="Batches to prefetch to GPU">
                    <NumberInput
                      value={jobConfig.config.process[0].datasets?.[0]?.gpu_prefetch_batches ?? 0}
                      onChange={value => setJobConfig(value, 'config.process[0].datasets[0].gpu_prefetch_batches')}
                      min={0}
                      max={5}
                    />
                  </FormGroup>
                </div>
              </div>
            )}

            {/* Step 7: Optimizer */}
            {currentStepDef.id === 'optimizer' && (
              <div className="space-y-4">
                <FormGroup label="Optimizer" tooltip="Algorithm for training">
                  <SelectInput
                    value={jobConfig.config.process[0].train?.optimizer || 'adamw8bit'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                    options={[
                      { value: 'adamw8bit', label: 'AdamW 8-bit (Recommended)' },
                      { value: 'adamw', label: 'AdamW (Full Precision)' },
                      { value: 'adafactor', label: 'Adafactor (Memory Efficient)' },
                      { value: 'lion', label: 'Lion (Memory Efficient, Fast)' },
                      { value: 'lion8bit', label: 'Lion 8-bit' },
                      { value: 'prodigy', label: 'Prodigy (Adaptive LR)' },
                      { value: 'dadaptation', label: 'D-Adaptation (Adaptive)' }
                    ]}
                  />
                </FormGroup>

                <FormGroup label="Learning Rate" tooltip="Speed of learning">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.lr ?? 0.0001}
                    onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                    min={0.000001}
                    max={0.01}
                  />
                </FormGroup>

                <FormGroup label="Gradient Accumulation Steps" tooltip="Effective batch size multiplier - accumulate gradients over N steps">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.gradient_accumulation ?? 1}
                    onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                    min={1}
                    max={64}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Effective batch size = batch_size × accumulation steps. Higher values simulate larger batches with less VRAM.
                  </div>
                </FormGroup>

                <FormGroup label="Max Gradient Norm" tooltip="Gradient clipping threshold to prevent exploding gradients">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.max_grad_norm ?? 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.max_grad_norm')}
                    min={0.1}
                    max={10}
                    step={0.1}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Clips gradients to prevent training instability. 1.0 is standard, lower values are more conservative.
                  </div>
                </FormGroup>

                <FormGroup label="Separate Component Learning Rates" tooltip="Set different learning rates for UNet and Text Encoder">
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm font-medium">UNet Learning Rate (optional)</label>
                      <NumberInput
                        value={jobConfig.config.process[0].train?.unet_lr ?? 0}
                        onChange={value => setJobConfig(value || undefined, 'config.process[0].train.unet_lr')}
                        min={0}
                        max={0.01}
                      />
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Leave at 0 to use main LR. Set separately for fine-grained control.
                      </div>
                    </div>
                    <div>
                      <label className="text-sm font-medium">Text Encoder Learning Rate (optional)</label>
                      <NumberInput
                        value={jobConfig.config.process[0].train?.text_encoder_lr ?? 0}
                        onChange={value => setJobConfig(value || undefined, 'config.process[0].train.text_encoder_lr')}
                        min={0}
                        max={0.01}
                      />
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Usually lower than UNet LR (e.g., 1e-5 vs 1e-4). Only used if training text encoder.
                      </div>
                    </div>
                  </div>
                </FormGroup>

                <FormGroup label="Model Components to Train" tooltip="Choose which parts of the model to train. UNet/Transformer learns visual concepts while Text Encoder learns language understanding. Memory cost varies: SD1.5 TE adds ~1GB, SDXL TE adds ~3-5GB (two encoders), Flux TE adds ~8GB (T5-XXL). For unified memory systems, this directly reduces available system RAM.">
                  <div className="space-y-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.train_unet}
                      onChange={value => setJobConfig(value, 'config.process[0].train.train_unet')}
                      label="Train UNet/Transformer (main diffusion model)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.train_text_encoder}
                      onChange={value => setJobConfig(value, 'config.process[0].train.train_text_encoder')}
                      label="Train Text Encoder (improves prompt understanding)"
                    />
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Training text encoder can improve prompt adherence but uses more VRAM and may cause instability.
                    </div>
                  </div>
                </FormGroup>

                <FormGroup label="Prompt Dropout Probability" tooltip="Randomly drop entire prompts during training">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.prompt_dropout_prob ?? 0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.prompt_dropout_prob')}
                    min={0}
                    max={0.5}
                    step={0.01}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Drops prompts before encoding. Helps with unconditional generation. 0 = disabled, 0.1 = 10% dropout.
                  </div>
                </FormGroup>
              </div>
            )}

            {/* Step 8: Advanced Training */}
            {currentStepDef.id === 'advanced' && (
              <div className="space-y-4">
                <FormGroup label="Learning Rate Scheduler" tooltip="How learning rate changes during training">
                  <SelectInput
                    value={jobConfig.config.process[0].train?.lr_scheduler || 'constant'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.lr_scheduler')}
                    options={[
                      { value: 'constant', label: 'Constant - Simple, no decay' },
                      { value: 'linear', label: 'Linear - Gradual decay' },
                      { value: 'cosine', label: 'Cosine - Smooth S-curve decay' },
                      { value: 'cosine_with_restarts', label: 'Cosine with Restarts - Periodic warm restarts' },
                      { value: 'polynomial', label: 'Polynomial - Configurable decay' }
                    ]}
                  />
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    {jobConfig.config.process[0].train?.lr_scheduler === 'cosine_with_restarts' && (
                      <span>Cosine with restarts helps escape local minima by periodically resetting the learning rate.</span>
                    )}
                    {jobConfig.config.process[0].train?.lr_scheduler === 'linear' && (
                      <span>Linear decay provides stable convergence by gradually reducing learning rate.</span>
                    )}
                    {jobConfig.config.process[0].train?.lr_scheduler === 'constant' && (
                      <span>Constant LR is simplest but may not produce optimal results for long training runs.</span>
                    )}
                  </div>
                </FormGroup>

                <FormGroup label="Noise Scheduler" tooltip="Algorithm for noise schedule during training">
                  <SelectInput
                    value={jobConfig.config.process[0].train?.noise_scheduler || 'ddpm'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.noise_scheduler')}
                    options={[
                      { value: 'ddpm', label: 'DDPM - Stable, default choice' },
                      { value: 'euler', label: 'Euler - Faster convergence' },
                      { value: 'lms', label: 'LMS - Linear multi-step' }
                    ]}
                  />
                </FormGroup>

                <FormGroup label="Loss Target" tooltip="What the model learns to predict">
                  <SelectInput
                    value={jobConfig.config.process[0].train?.loss_target || 'noise'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.loss_target')}
                    options={[
                      { value: 'noise', label: 'Noise - Predict the noise (default, most stable)' },
                      { value: 'source', label: 'Source - Predict the clean image' },
                      { value: 'differential_noise', label: 'Differential Noise - Predict noise difference (advanced)' }
                    ]}
                  />
                </FormGroup>

                <FormGroup label="Noise Offset" tooltip="Helps with dark and bright images">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.noise_offset ?? 0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.noise_offset')}
                    min={0}
                    max={0.2}
                    step={0.01}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Small values (0.02-0.1) help preserve dynamic range in very dark/bright images. 0 = disabled.
                  </div>
                </FormGroup>

                <FormGroup label="Min SNR Gamma" tooltip="Signal-to-noise ratio weighting for loss">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.min_snr_gamma ?? 0}
                    onChange={value => setJobConfig(value || undefined, 'config.process[0].train.min_snr_gamma')}
                    min={0}
                    max={20}
                    step={0.5}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Values like 5.0 provide balanced training across noise levels. 0 = disabled.
                  </div>
                </FormGroup>

                <FormGroup label="SNR Gamma" tooltip="Signal-to-noise ratio value for loss weighting">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.snr_gamma ?? 0}
                    onChange={value => setJobConfig(value || undefined, 'config.process[0].train.snr_gamma')}
                    min={0}
                    max={20}
                    step={0.5}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Direct SNR gamma value (alternative to min_snr_gamma). 0 = disabled.
                  </div>
                </FormGroup>

                <FormGroup label="Timestep Type" tooltip="How to sample timesteps during training">
                  <SelectInput
                    value={jobConfig.config.process[0].train?.timestep_type || 'sigmoid'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      { value: 'sigmoid', label: 'Sigmoid - Balanced distribution (default)' },
                      { value: 'linear', label: 'Linear - Uniform sampling' },
                      { value: 'lognorm_blend', label: 'LogNorm Blend - Focus on mid-range' },
                      { value: 'next_sample', label: 'Next Sample - For distillation' },
                      { value: 'weighted', label: 'Weighted - Custom weighting' },
                      { value: 'one_step', label: 'One Step - Single step training' }
                    ]}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Sigmoid is recommended for most cases. Linear provides uniform coverage. LogNorm focuses on mid-noise levels.
                  </div>
                </FormGroup>

                <FormGroup label="Text Encoder Memory Management" tooltip="Memory optimization strategies for text encoder. Caching uses RAM: SD1.5 ~500MB, SDXL ~1.5GB, Flux ~4GB. Unloading to CPU frees VRAM but adds latency. NOTE: On unified memory systems (Apple Silicon), unloading to CPU provides NO benefit as RAM/VRAM are shared - use caching instead. On discrete GPUs, both options are beneficial.">
                  <div className="space-y-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.cache_text_embeddings}
                      onChange={value => setJobConfig(value, 'config.process[0].train.cache_text_embeddings')}
                      label="Cache text embeddings (faster, uses more RAM)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.unload_text_encoder}
                      onChange={value => setJobConfig(value, 'config.process[0].train.unload_text_encoder')}
                      label="Unload text encoder to CPU (saves VRAM)"
                    />
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Cache embeddings speeds up training. Unload TE frees VRAM but slower. Best used together.
                  </div>
                </FormGroup>

                <FormGroup label="Classifier-Free Guidance Training" tooltip="Train with CFG to improve prompt adherence at inference. Doubles compute per step as model processes both conditional and unconditional paths. Cost: ~50% longer training time, ~30% more memory. Memory impact: SD1.5 +2GB, SDXL +5GB, Flux +8GB. For Flux models, consider using Flux CFG Mode in model settings instead. On unified memory systems, this significantly reduces available shared pool.">
                  <Checkbox
                    checked={!!jobConfig.config.process[0].train?.do_cfg}
                    onChange={value => setJobConfig(value, 'config.process[0].train.do_cfg')}
                    label="Enable CFG training (advanced)"
                  />
                  {jobConfig.config.process[0].train?.do_cfg && (
                    <div className="mt-2">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.cfg_scale ?? 3.0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.cfg_scale')}
                        min={1}
                        max={10}
                        step={0.5}
                      />
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        CFG scale during training (typically 2.0-5.0)
                      </div>
                    </div>
                  )}
                </FormGroup>

                <FormGroup label="Data Augmentation" tooltip="Image transformations that artificially expand your dataset. Flipping effectively doubles your data but should be avoided for asymmetric subjects like faces or text. Works equally well for all model architectures (SD1.5, SDXL, Flux). Cost: No memory overhead, minimal compute impact.">
                  <div className="space-y-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].datasets[0]?.flip_x}
                      onChange={value => setJobConfig(value, 'config.process[0].datasets[0].flip_x')}
                      label="Horizontal flip (doubles dataset, avoid for faces)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].datasets[0]?.flip_y}
                      onChange={value => setJobConfig(value, 'config.process[0].datasets[0].flip_y')}
                      label="Vertical flip (rarely useful)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].datasets[0]?.random_crop}
                      onChange={value => setJobConfig(value, 'config.process[0].datasets[0].random_crop')}
                      label="Random crop (helps generalization for small datasets)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].datasets[0]?.random_scale}
                      onChange={value => setJobConfig(value, 'config.process[0].datasets[0].random_scale')}
                      label="Random scale (usually not helpful for LoRA)"
                    />
                  </div>
                </FormGroup>

                <FormGroup label="Latent Caching" tooltip="Cache encoded images for faster training">
                  <div className="space-y-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].datasets[0]?.cache_latents}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].datasets[0].cache_latents');
                        if (value && jobConfig.config.process[0].datasets[0]?.cache_latents_to_disk) {
                          setJobConfig(false, 'config.process[0].datasets[0].cache_latents_to_disk');
                        }
                      }}
                      label="Cache latents in RAM (5-10x faster, uses memory)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].datasets[0]?.cache_latents_to_disk}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].datasets[0].cache_latents_to_disk');
                        if (value && jobConfig.config.process[0].datasets[0]?.cache_latents) {
                          setJobConfig(false, 'config.process[0].datasets[0].cache_latents');
                        }
                      }}
                      label="Cache latents to disk (slower but saves RAM)"
                    />
                  </div>
                </FormGroup>
              </div>
            )}

            {/* Step 9: Regularization */}
            {currentStepDef.id === 'regularization' && (
              <div className="space-y-4">
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                  Regularization helps prevent overfitting. These settings are optional but recommended.
                </p>

                {/* Memory impact summary */}
                {systemProfile && (
                  <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 mb-4">
                    <h4 className="font-medium text-blue-800 dark:text-blue-200 text-sm mb-2">Memory Impact Summary</h4>
                    {(() => {
                      const modelArch = jobConfig.config.process[0].model?.arch || 'flux';
                      const useEma = !!jobConfig.config.process[0].train?.ema_config?.use_ema;
                      const useDiffPreserve = !!jobConfig.config.process[0].train?.diff_output_preservation;
                      const gradCheckpoint = !!jobConfig.config.process[0].train?.gradient_checkpointing;
                      const costs = calculateRegularizationMemoryCost(modelArch, useEma, useDiffPreserve, gradCheckpoint);
                      const totalCost = costs.ema + costs.diffOutputPreservation + costs.gradientCheckpointingSavings;

                      return (
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span>EMA:</span>
                            <span className={useEma ? 'text-orange-600 dark:text-orange-400 font-medium' : 'text-gray-500'}>
                              {useEma ? `+${costs.ema}GB` : '0GB'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Output Preservation:</span>
                            <span className={useDiffPreserve ? 'text-orange-600 dark:text-orange-400 font-medium' : 'text-gray-500'}>
                              {useDiffPreserve ? `+${costs.diffOutputPreservation}GB` : '0GB'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Gradient Checkpointing:</span>
                            <span className={gradCheckpoint ? 'text-green-600 dark:text-green-400 font-medium' : 'text-gray-500'}>
                              {gradCheckpoint ? `${costs.gradientCheckpointingSavings.toFixed(1)}GB (saves memory)` : '0GB'}
                            </span>
                          </div>
                          <div className="border-t border-blue-200 dark:border-blue-700 pt-1 mt-1 flex justify-between font-medium">
                            <span>Net Impact:</span>
                            <span className={totalCost > 0 ? 'text-orange-600 dark:text-orange-400' : 'text-green-600 dark:text-green-400'}>
                              {totalCost > 0 ? `+${totalCost.toFixed(1)}GB` : `${totalCost.toFixed(1)}GB`}
                            </span>
                          </div>
                          {systemProfile.gpu.isUnifiedMemory && totalCost > 5 && (
                            <p className="text-orange-600 dark:text-orange-400 mt-2">
                              ⚠️ High memory usage for unified memory system. Consider disabling EMA or enabling gradient checkpointing.
                            </p>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}

                {/* Caption Dropout */}
                <FormGroup
                  label="Caption Dropout Rate"
                  tooltip="Randomly drops captions during training to improve generalization. Higher values = more generalization but less prompt adherence."
                >
                  <NumberInput
                    value={jobConfig.config.process[0].datasets?.[0]?.caption_dropout_rate ?? 0.05}
                    onChange={value => setJobConfig(value, 'config.process[0].datasets[0].caption_dropout_rate')}
                    min={0}
                    max={0.5}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Recommended: 0.05 (5%). Set to 0 for strict prompt following.
                  </p>
                </FormGroup>

                {/* Weight Decay */}
                <FormGroup
                  label="Weight Decay"
                  tooltip="L2 regularization strength. Prevents weights from growing too large."
                >
                  <NumberInput
                    value={jobConfig.config.process[0].train?.optimizer_params?.weight_decay ?? 0.0001}
                    onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                    min={0}
                    max={0.1}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Recommended: 0.0001 for most cases. Higher values (0.01) for small datasets.
                  </p>
                </FormGroup>

                {/* Gradient Checkpointing */}
                <FormGroup
                  label="Gradient Checkpointing"
                  tooltip="Trades compute for memory. Allows training larger models with less VRAM."
                >
                  <div className="flex items-center gap-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.gradient_checkpointing}
                      onChange={value => setJobConfig(value, 'config.process[0].train.gradient_checkpointing')}
                      label="Enable gradient checkpointing"
                    />
                    <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 px-2 py-0.5 rounded">
                      Saves ~30% memory
                    </span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Recommended for limited VRAM. Trades compute time for memory savings.
                  </p>
                </FormGroup>

                {/* EMA */}
                <FormGroup
                  label="Exponential Moving Average (EMA)"
                  tooltip="Maintains a smoothed version of weights for more stable results"
                >
                  <div className="flex items-center gap-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.ema_config?.use_ema}
                      onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                      label="Enable EMA (reduces training noise)"
                    />
                    <span className="text-xs bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 px-2 py-0.5 rounded">
                      +1GB memory
                    </span>
                  </div>
                  {systemProfile?.gpu.isUnifiedMemory && (
                    <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                      ⚠️ EMA adds memory overhead. Consider disabling for unified memory systems with limited RAM.
                    </p>
                  )}
                </FormGroup>

                {jobConfig.config.process[0].train?.ema_config?.use_ema && (
                  <div className="ml-6">
                    <FormGroup label="EMA Decay" tooltip="Controls how quickly the averaged weights adapt to new updates. Higher values (closer to 1.0) create smoother, more stable weights but respond slower to new learning. Lower values are more responsive but less smooth. Cost: Stores duplicate LoRA weights (~20-150MB for SD1.5 LoRA, ~50-400MB for SDXL LoRA, ~100-800MB for Flux LoRA depending on rank). On unified memory systems, this reduces available shared RAM/VRAM pool.">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.ema_config?.ema_decay ?? 0.99}
                        onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                        min={0.9}
                        max={0.9999}
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Higher = smoother but slower adaptation. Recommended: 0.99
                      </p>
                    </FormGroup>
                  </div>
                )}

                {/* Diff Output Preservation */}
                <FormGroup
                  label="Differential Output Preservation"
                  tooltip="Preserves model's original capabilities while learning new concepts"
                >
                  <div className="flex items-center gap-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.diff_output_preservation}
                      onChange={value => setJobConfig(value, 'config.process[0].train.diff_output_preservation')}
                      label="Enable output preservation (prevents catastrophic forgetting)"
                    />
                    <span className="text-xs bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 px-2 py-0.5 rounded">
                      +3GB memory
                    </span>
                  </div>
                  {systemProfile?.gpu.isUnifiedMemory && (
                    <p className="text-xs text-orange-600 dark:text-orange-400 mt-1">
                      ⚠️ Requires storing original model outputs. Significant memory overhead.
                    </p>
                  )}
                </FormGroup>

                {jobConfig.config.process[0].train?.diff_output_preservation && (
                  <div className="ml-6 space-y-3">
                    <FormGroup label="Preservation Strength" tooltip="Balance between preserving original model behavior vs learning new concepts. Higher values maintain more original capabilities but learn new concepts slower. Lower values learn faster but risk forgetting existing knowledge. Memory overhead is from parent feature (Differential Output Preservation) which stores original model outputs.">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.diff_output_preservation_multiplier ?? 1.0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')}
                        min={0.1}
                        max={10}
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Higher = more preservation, less concept learning
                      </p>
                    </FormGroup>

                    <FormGroup label="Preservation Class" tooltip="The general category of your training subject. Must match your concept type: 'person' for faces/portraits, 'dog'/'cat' for animals, 'style' for artistic styles, 'object' for items. Common for SD1.5/SDXL. For Flux models, this may have less impact due to different architecture.">
                      <TextInput
                        value={jobConfig.config.process[0].train?.diff_output_preservation_class ?? 'person'}
                        onChange={value => setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')}
                        placeholder="person"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Class to preserve (e.g., &quot;person&quot;, &quot;dog&quot;, &quot;style&quot;)
                      </p>
                    </FormGroup>
                  </div>
                )}

                {/* Loss Type */}
                <FormGroup
                  label="Loss Function"
                  tooltip="Different loss functions can affect training quality"
                >
                  <SelectInput
                    value={jobConfig.config.process[0].train?.loss_type || 'mse'}
                    onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                    options={[
                      { value: 'mse', label: 'MSE (Mean Squared Error) - Standard' },
                      { value: 'mae', label: 'MAE (Mean Absolute Error) - Robust to outliers' },
                      { value: 'wavelet', label: 'Wavelet - Better for fine details' }
                    ]}
                  />
                </FormGroup>

                {/* Advanced Preservation Strategies */}
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg space-y-4">
                  <h4 className="font-medium text-purple-800 dark:text-purple-200">Advanced Preservation (Experimental)</h4>

                  <FormGroup label="Blank Prompt Preservation" tooltip="Preserve model's understanding of blank/empty prompts">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.blank_prompt_preservation}
                      onChange={value => setJobConfig(value, 'config.process[0].train.blank_prompt_preservation')}
                      label="Enable blank prompt preservation"
                    />
                    {jobConfig.config.process[0].train?.blank_prompt_preservation && (
                      <div className="mt-2">
                        <NumberInput
                          value={jobConfig.config.process[0].train?.blank_prompt_preservation_multiplier ?? 1.0}
                          onChange={value => setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')}
                          min={0.1}
                          max={10}
                          step={0.1}
                        />
                      </div>
                    )}
                  </FormGroup>

                  <FormGroup label="Inverted Mask Prior" tooltip="Use inverted mask as regularization">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.inverted_mask_prior}
                      onChange={value => setJobConfig(value, 'config.process[0].train.inverted_mask_prior')}
                      label="Enable inverted mask prior"
                    />
                    {jobConfig.config.process[0].train?.inverted_mask_prior && (
                      <div className="mt-2">
                        <NumberInput
                          value={jobConfig.config.process[0].train?.inverted_mask_prior_multiplier ?? 0.5}
                          onChange={value => setJobConfig(value, 'config.process[0].train.inverted_mask_prior_multiplier')}
                          min={0.1}
                          max={5}
                          step={0.1}
                        />
                      </div>
                    )}
                  </FormGroup>

                  <FormGroup label="Prior Divergence" tooltip="Regularization technique that penalizes deviation from the original model's prior distribution. Helps prevent catastrophic forgetting. Most effective for SD1.5/SDXL models; experimental for Flux. Cost: ~15-20% slower training, minimal memory overhead. Works on both discrete GPUs and unified memory systems.">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.do_prior_divergence}
                      onChange={value => setJobConfig(value, 'config.process[0].train.do_prior_divergence')}
                      label="Enable prior divergence (advanced)"
                    />
                  </FormGroup>

                  <FormGroup label="Weight Jitter" tooltip="Add noise to weights for regularization">
                    <NumberInput
                      value={jobConfig.config.process[0].train?.weight_jitter ?? 0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.weight_jitter')}
                      min={0}
                      max={0.1}
                      step={0.001}
                    />
                  </FormGroup>
                </div>

                {/* Feature Extractors */}
                <div className="p-4 bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-lg space-y-4">
                  <h4 className="font-medium text-indigo-800 dark:text-indigo-200">Feature Extraction (Advanced)</h4>

                  <FormGroup label="Latent Feature Extractor" tooltip="Path to model for extracting latent features">
                    <TextInput
                      value={jobConfig.config.process[0].train?.latent_feature_extractor_path || ''}
                      onChange={value => setJobConfig(value || undefined, 'config.process[0].train.latent_feature_extractor_path')}
                      placeholder="Optional: path/to/feature_extractor"
                    />
                    {jobConfig.config.process[0].train?.latent_feature_extractor_path && (
                      <div className="mt-2">
                        <label className="text-xs text-gray-500">Feature Loss Weight</label>
                        <NumberInput
                          value={jobConfig.config.process[0].train?.latent_feature_loss_weight ?? 1.0}
                          onChange={value => setJobConfig(value, 'config.process[0].train.latent_feature_loss_weight')}
                          min={0}
                          max={10}
                          step={0.1}
                        />
                      </div>
                    )}
                  </FormGroup>

                  <FormGroup label="Diffusion Feature Extractor" tooltip="Path to diffusion-based feature extractor">
                    <TextInput
                      value={jobConfig.config.process[0].train?.diffusion_feature_extractor_path || ''}
                      onChange={value => setJobConfig(value || undefined, 'config.process[0].train.diffusion_feature_extractor_path')}
                      placeholder="Optional: path/to/diffusion_feature_extractor"
                    />
                    {jobConfig.config.process[0].train?.diffusion_feature_extractor_path && (
                      <div className="mt-2">
                        <label className="text-xs text-gray-500">Diffusion Feature Weight</label>
                        <NumberInput
                          value={jobConfig.config.process[0].train?.diffusion_feature_extractor_weight ?? 1.0}
                          onChange={value => setJobConfig(value, 'config.process[0].train.diffusion_feature_extractor_weight')}
                          min={0}
                          max={10}
                          step={0.1}
                        />
                      </div>
                    )}
                  </FormGroup>
                </div>

                {/* Advanced Noise & Training Modes */}
                <div className="p-4 bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg space-y-4">
                  <h4 className="font-medium text-orange-800 dark:text-orange-200">Advanced Training Modes</h4>

                  <FormGroup label="Optimal Noise Pairing" tooltip="Number of noise samples to find optimal pairing">
                    <NumberInput
                      value={jobConfig.config.process[0].train?.optimal_noise_pairing_samples ?? 1}
                      onChange={value => setJobConfig(value, 'config.process[0].train.optimal_noise_pairing_samples')}
                      min={1}
                      max={16}
                    />
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Higher values find better noise pairings but slower. 1 = disabled.
                    </div>
                  </FormGroup>

                  <FormGroup label="Advanced Noise Options" tooltip="Fine-grained control over noise behavior during training. Consistent noise ensures reproducibility for debugging. Blended blur noise can help with fine detail learning. Works with all model architectures (SD1.5, SDXL, Flux). Cost: Minimal performance impact but may affect training dynamics.">
                    <div className="space-y-2">
                      <Checkbox
                        checked={!!jobConfig.config.process[0].train?.force_consistent_noise}
                        onChange={value => setJobConfig(value, 'config.process[0].train.force_consistent_noise')}
                        label="Force consistent noise for same image at size"
                      />
                      <Checkbox
                        checked={!!jobConfig.config.process[0].train?.blended_blur_noise}
                        onChange={value => setJobConfig(value, 'config.process[0].train.blended_blur_noise')}
                        label="Blend blur with noise"
                      />
                    </div>
                  </FormGroup>

                  <FormGroup label="Turbo Training Mode" tooltip="Experimental accelerated training mode using specialized optimization for faster convergence. Primarily designed for SD1.5/SDXL; experimental support for Flux. May reduce training time by 30-50% but results can be less stable. Cost: Similar memory usage, potentially faster but less predictable results.">
                    <div className="space-y-2">
                      <Checkbox
                        checked={!!jobConfig.config.process[0].train?.train_turbo}
                        onChange={value => setJobConfig(value, 'config.process[0].train.train_turbo')}
                        label="Enable turbo training mode (experimental)"
                      />
                      {jobConfig.config.process[0].train?.train_turbo && (
                        <Checkbox
                          checked={!!jobConfig.config.process[0].train?.show_turbo_outputs}
                          onChange={value => setJobConfig(value, 'config.process[0].train.show_turbo_outputs')}
                          label="Show turbo outputs during training"
                        />
                      )}
                    </div>
                  </FormGroup>

                  <FormGroup label="FreeU Mode" tooltip="Applies learned attention scaling factors during training to improve generation quality. Designed for UNet-based models (SD1.5, SDXL); NOT compatible with Flux transformer architecture. Cost: ~10% slower training, minimal memory overhead. Best for style and detail-focused training.">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.free_u}
                      onChange={value => setJobConfig(value, 'config.process[0].train.free_u')}
                      label="Enable FreeU training mode"
                    />
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      FreeU applies attention scaling to improve generation quality.
                    </div>
                  </FormGroup>
                </div>
              </div>
            )}

            {/* Step 9: Training Core */}
            {currentStepDef.id === 'training' && (
              <div className="space-y-4">
                <FormGroup label="Training Name" tooltip="Unique identifier for this training run. Used in output folder names, checkpoints, and logs. Choose a descriptive name to easily identify this training session later.">
                  <TextInput
                    value={jobConfig.config.name}
                    onChange={value => setJobConfig(value, 'config.name')}
                    placeholder="my-lora-training"
                  />
                </FormGroup>

                <FormGroup label="Training Steps" tooltip="Total number of training iterations. More steps = longer training time but potentially better results. Cost varies by model: SD1.5 ~1-2 sec/step, SDXL ~3-5 sec/step, Flux ~5-10 sec/step. 1000-2000 steps typical for SD1.5, 2000-4000 for SDXL, 1500-3000 for Flux.">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.steps ?? 1000}
                    onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                    min={100}
                    max={50000}
                  />
                </FormGroup>

                <FormGroup label="Timestep Range" tooltip="Focus training on specific noise levels (0=clean, 999=pure noise)">
                  <div className="flex gap-4">
                    <div className="flex-1">
                      <label className="text-xs text-gray-500 dark:text-gray-400">Min Timestep</label>
                      <NumberInput
                        value={jobConfig.config.process[0].train?.min_denoising_steps ?? 0}
                        onChange={value => setJobConfig(value, 'config.process[0].train.min_denoising_steps')}
                        min={0}
                        max={998}
                      />
                    </div>
                    <div className="flex-1">
                      <label className="text-xs text-gray-500 dark:text-gray-400">Max Timestep</label>
                      <NumberInput
                        value={jobConfig.config.process[0].train?.max_denoising_steps ?? 999}
                        onChange={value => setJobConfig(value, 'config.process[0].train.max_denoising_steps')}
                        min={1}
                        max={999}
                      />
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Full range (0-999) is recommended for general training. Narrowing the range focuses learning on specific denoising stages.
                  </div>
                </FormGroup>
              </div>
            )}

            {/* Step 10: Sampling */}
            {currentStepDef.id === 'sampling' && (
              <div className="space-y-4">
                <FormGroup label="Sample Every N Steps" tooltip="Generate preview images during training">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.sample_every ?? 250}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                    min={50}
                    max={5000}
                  />
                </FormGroup>

                <FormGroup label="Sampling Control" tooltip="Control when and if sampling occurs">
                  <div className="space-y-2">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.disable_sampling}
                      onChange={value => setJobConfig(value, 'config.process[0].train.disable_sampling')}
                      label="Disable sampling entirely (faster training)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.skip_first_sample}
                      onChange={value => setJobConfig(value, 'config.process[0].train.skip_first_sample')}
                      label="Skip first sample (start training immediately)"
                    />
                    <Checkbox
                      checked={!!jobConfig.config.process[0].train?.force_first_sample}
                      onChange={value => setJobConfig(value, 'config.process[0].train.force_first_sample')}
                      label="Force first sample (sample before any training)"
                    />
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Skip first sample is recommended to start training faster. Disable sampling entirely if you only care about final weights.
                  </div>
                </FormGroup>

                <FormGroup label="Sampler" tooltip="Algorithm for generating preview images">
                  <SelectInput
                    value={jobConfig.config.process[0].sample?.sampler || 'flowmatch'}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                    options={[
                      { value: 'flowmatch', label: 'FlowMatch - Best for Flux/Flow models' },
                      { value: 'ddpm', label: 'DDPM - Classic diffusion sampler' },
                      { value: 'euler', label: 'Euler - Fast and efficient' },
                      { value: 'euler_a', label: 'Euler Ancestral - More creative' }
                    ]}
                  />
                </FormGroup>

                <FormGroup label="Sample Steps" tooltip="Number of denoising steps for previews">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.sample_steps ?? 25}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                    min={10}
                    max={100}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Higher = better quality but slower previews. 20-30 is typical.
                  </div>
                </FormGroup>

                <FormGroup label="Guidance Scale" tooltip="CFG scale for preview generation">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.guidance_scale ?? 4}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                    min={1}
                    max={20}
                    step={0.5}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Higher = stronger prompt adherence. 3-7 is typical (Flux uses lower: 3-4, SDXL uses higher: 7-8).
                  </div>
                </FormGroup>

                <FormGroup label="Preview Seed" tooltip="Random seed for reproducible previews">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.seed ?? 42}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                    min={0}
                    max={999999}
                  />
                  <Checkbox
                    checked={!!jobConfig.config.process[0].sample?.walk_seed}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                    label="Walk seed (increment for each sample)"
                  />
                </FormGroup>

                <FormGroup label="Preview Dimensions" tooltip="Custom width and height for preview images">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-500 dark:text-gray-400">Width</label>
                      <NumberInput
                        value={jobConfig.config.process[0].sample?.width ?? 1024}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                        min={256}
                        max={2048}
                        step={64}
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-500 dark:text-gray-400">Height</label>
                      <NumberInput
                        value={jobConfig.config.process[0].sample?.height ?? 1024}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                        min={256}
                        max={2048}
                        step={64}
                      />
                    </div>
                  </div>
                </FormGroup>

                <FormGroup label="Negative Prompt" tooltip="Default negative prompt for previews">
                  <TextInput
                    value={jobConfig.config.process[0].sample?.neg || ''}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.neg')}
                    placeholder="Optional: blurry, low quality, distorted"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Note: Negative prompts only work for models that support them (SD1.5, SDXL). Flux ignores this.
                  </div>
                </FormGroup>

                <FormGroup label="Network Multiplier" tooltip="Strength of LoRA network during preview generation">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.network_multiplier ?? 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.network_multiplier')}
                    min={0}
                    max={2}
                    step={0.1}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    1.0 = full strength. Lower values reduce LoRA influence. Useful for testing at different strengths.
                  </div>
                </FormGroup>

                <FormGroup label="Guidance Rescale" tooltip="Rescale guidance to prevent over-saturation">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.guidance_rescale ?? 0}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_rescale')}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    0 = disabled. Values like 0.7 help reduce over-saturation from high guidance scales.
                  </div>
                </FormGroup>

                <FormGroup label="Preview Image Format" tooltip="File format for saved preview images">
                  <SelectInput
                    value={jobConfig.config.process[0].sample?.format || 'jpg'}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.format')}
                    options={[
                      { value: 'jpg', label: 'JPEG (smaller, lossy)' },
                      { value: 'png', label: 'PNG (lossless, larger)' },
                      { value: 'webp', label: 'WebP (modern, efficient)' }
                    ]}
                  />
                </FormGroup>

                <FormGroup label="Refiner Start (SDXL only)" tooltip="When to start refiner model (0.0-1.0)">
                  <NumberInput
                    value={jobConfig.config.process[0].sample?.refiner_start_at ?? 0.5}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.refiner_start_at')}
                    min={0}
                    max={1}
                    step={0.1}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Only for SDXL with refiner. 0.5 = start refiner halfway through generation.
                  </div>
                </FormGroup>
              </div>
            )}

            {/* Step 11: Save Settings */}
            {currentStepDef.id === 'save' && (
              <div className="space-y-4">
                <FormGroup label="Save Every N Steps" tooltip="How frequently to save checkpoint copies during training. More frequent saves let you recover from issues but use more disk space. Checkpoint sizes vary by model and rank: SD1.5 LoRA ~20-150MB, SDXL LoRA ~50-400MB, Flux LoRA ~100-800MB. Each save takes 5-30 seconds. Stored on local disk, not in GPU/unified memory.">
                  <NumberInput
                    value={jobConfig.config.process[0].save?.save_every ?? 500}
                    onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
                    min={100}
                    max={10000}
                  />
                </FormGroup>

                <FormGroup label="Max Checkpoints to Keep" tooltip="Set to 0 to keep all checkpoints">
                  <div className="space-y-2">
                    <Checkbox
                      checked={jobConfig.config.process[0].save?.max_step_saves_to_keep === 0}
                      onChange={checked => {
                        if (checked) {
                          setJobConfig(0, 'config.process[0].save.max_step_saves_to_keep');
                        } else {
                          setJobConfig(5, 'config.process[0].save.max_step_saves_to_keep');
                        }
                      }}
                      label="Keep all checkpoints (uses more disk space)"
                    />
                    {jobConfig.config.process[0].save?.max_step_saves_to_keep !== 0 && (
                      <NumberInput
                        value={jobConfig.config.process[0].save?.max_step_saves_to_keep ?? 5}
                        onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
                        min={1}
                        max={100}
                      />
                    )}
                  </div>
                </FormGroup>

                <FormGroup label="Save Format" tooltip="File format for saved LoRA weights">
                  <SelectInput
                    value={jobConfig.config.process[0].save?.save_format || 'safetensors'}
                    onChange={value => setJobConfig(value, 'config.process[0].save.save_format')}
                    options={[
                      { value: 'safetensors', label: 'SafeTensors (recommended, safe & fast)' },
                      { value: 'diffusers', label: 'Diffusers (HuggingFace compatible)' },
                      { value: 'ckpt', label: 'Checkpoint (legacy, less secure)' }
                    ]}
                  />
                </FormGroup>

                <FormGroup label="Save Data Type" tooltip="Precision for saved weights">
                  <SelectInput
                    value={jobConfig.config.process[0].save?.dtype || 'bf16'}
                    onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
                    options={[
                      { value: 'bf16', label: 'BFloat16 (recommended for modern GPUs)' },
                      { value: 'float16', label: 'Float16 (smaller, wider compatibility)' },
                      { value: 'float32', label: 'Float32 (full precision, largest)' }
                    ]}
                  />
                </FormGroup>

                <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg space-y-4">
                  <h4 className="font-medium">HuggingFace Hub Integration</h4>

                  <FormGroup label="Push to Hub" tooltip="Automatically upload LoRA to HuggingFace Hub">
                    <Checkbox
                      checked={!!jobConfig.config.process[0].save?.push_to_hub}
                      onChange={value => setJobConfig(value, 'config.process[0].save.push_to_hub')}
                      label="Enable automatic upload to HuggingFace Hub"
                    />
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Requires HuggingFace Hub authentication. Run &apos;huggingface-cli login&apos; first.
                    </div>
                  </FormGroup>

                  {jobConfig.config.process[0].save?.push_to_hub && (
                    <div className="space-y-3 ml-4 border-l-2 border-gray-200 dark:border-gray-700 pl-4">
                      <FormGroup label="Repository ID" tooltip="HuggingFace repo in format username/repo-name">
                        <TextInput
                          value={jobConfig.config.process[0].save?.hf_repo_id || ''}
                          onChange={value => setJobConfig(value, 'config.process[0].save.hf_repo_id')}
                          placeholder="your-username/your-lora-name"
                        />
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          Format: username/repository-name. Will be created if it doesn&apos;t exist.
                        </div>
                      </FormGroup>

                      <FormGroup label="Private Repository" tooltip="Set repository visibility on HuggingFace Hub. Private repos are only visible to you and require HuggingFace Pro subscription ($9/month). Public repos are free and allow others to use your LoRA. Cost: No performance or memory overhead, only affects sharing. Upload speed depends on LoRA size: SD1.5 fastest, Flux slowest.">
                        <Checkbox
                          checked={!!jobConfig.config.process[0].save?.hf_private}
                          onChange={value => setJobConfig(value, 'config.process[0].save.hf_private')}
                          label="Make repository private (requires HF Pro)"
                        />
                      </FormGroup>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Step 13: Logging */}
            {currentStepDef.id === 'logging' && (
              <div className="space-y-4">
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                  Configure logging and experiment tracking to monitor training progress.
                </p>

                <FormGroup label="Weights & Biases Integration" tooltip="Log metrics to W&B for experiment tracking">
                  <Checkbox
                    checked={!!jobConfig.config.process[0].logging?.use_wandb}
                    onChange={value => setJobConfig(value, 'config.process[0].logging.use_wandb')}
                    label="Enable W&B logging"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Requires W&B account and API key configured. Visit wandb.ai to set up.
                  </div>
                </FormGroup>

                {jobConfig.config.process[0].logging?.use_wandb && (
                  <div className="space-y-4 ml-4 border-l-2 border-gray-200 dark:border-gray-700 pl-4">
                    <FormGroup label="Project Name" tooltip="W&B project name">
                      <input
                        type="text"
                        value={jobConfig.config.process[0].logging?.project_name || 'ai-toolkit'}
                        onChange={e => setJobConfig(e.target.value, 'config.process[0].logging.project_name')}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md"
                        placeholder="ai-toolkit"
                      />
                    </FormGroup>

                    <FormGroup label="Run Name" tooltip="Unique name for this training run">
                      <input
                        type="text"
                        value={jobConfig.config.process[0].logging?.run_name || ''}
                        onChange={e => setJobConfig(e.target.value, 'config.process[0].logging.run_name')}
                        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md"
                        placeholder="Leave empty for auto-generated name"
                      />
                    </FormGroup>
                  </div>
                )}

                <FormGroup label="Log Frequency" tooltip="How often to log metrics (every N steps)">
                  <NumberInput
                    value={jobConfig.config.process[0].logging?.log_every ?? 100}
                    onChange={value => setJobConfig(value, 'config.process[0].logging.log_every')}
                    min={1}
                    max={1000}
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Lower values provide more detail but may slow training slightly. 50-100 is typical.
                  </div>
                </FormGroup>

                <FormGroup label="Verbose Output" tooltip="Enable detailed logging that shows internal training metrics, gradient statistics, and debug information. Helpful for diagnosing training issues but creates larger log files. Cost: Minimal performance impact (~1-2% slower), ~10-50MB additional log storage per run. Useful for all model types (SD1.5, SDXL, Flux) when troubleshooting.">
                  <Checkbox
                    checked={!!jobConfig.config.process[0].logging?.verbose}
                    onChange={value => setJobConfig(value, 'config.process[0].logging.verbose')}
                    label="Enable verbose logging (detailed output)"
                  />
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Prints additional debug information during training. Useful for troubleshooting.
                  </div>
                </FormGroup>
              </div>
            )}

            {/* Step 14: Monitoring */}
            {currentStepDef.id === 'monitoring' && (
              <div className="space-y-4">
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                  Performance monitoring tracks memory usage, training metrics, and system health during training.
                  After training completes, it provides optimization recommendations.
                </p>

                <FormGroup
                  label="Performance Monitoring"
                  tooltip="Automatically collect metrics during training for post-run analysis"
                >
                  <Checkbox
                    checked={jobConfig.config.process[0].monitoring?.enabled ?? true}
                    onChange={value => setJobConfig(value, 'config.process[0].monitoring.enabled')}
                    label="Enable performance monitoring (recommended)"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Monitors memory usage, training progress, and system health with minimal overhead.
                  </p>
                </FormGroup>

                {jobConfig.config.process[0].monitoring?.enabled !== false && (
                  <div className="space-y-4 ml-4 border-l-2 border-gray-200 dark:border-gray-700 pl-4">
                    <FormGroup
                      label="Sample Interval (seconds)"
                      tooltip="How often to collect system metrics"
                    >
                      <NumberInput
                        value={jobConfig.config.process[0].monitoring?.sample_interval_seconds ?? 5}
                        onChange={value => setJobConfig(value, 'config.process[0].monitoring.sample_interval_seconds')}
                        min={1}
                        max={60}
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Lower = more detail, slightly higher overhead. 5 seconds is recommended.
                      </p>
                    </FormGroup>

                    <FormGroup
                      label="Track Process Memory"
                      tooltip="Monitor memory usage of main process and workers separately"
                    >
                      <Checkbox
                        checked={jobConfig.config.process[0].monitoring?.track_per_process ?? true}
                        onChange={value => setJobConfig(value, 'config.process[0].monitoring.track_per_process')}
                        label="Track per-process memory breakdown"
                      />
                    </FormGroup>

                    <FormGroup
                      label="Auto-Analyze on Complete"
                      tooltip="Automatically generate optimization recommendations when training finishes"
                    >
                      <Checkbox
                        checked={jobConfig.config.process[0].monitoring?.analyze_on_complete ?? true}
                        onChange={value => setJobConfig(value, 'config.process[0].monitoring.analyze_on_complete')}
                        label="Generate recommendations after training"
                      />
                    </FormGroup>

                    <div className="grid grid-cols-2 gap-4">
                      <FormGroup
                        label="Warning Threshold"
                        tooltip="Memory usage percentage that triggers warnings"
                      >
                        <NumberInput
                          value={(jobConfig.config.process[0].monitoring?.memory_warning_threshold ?? 0.85) * 100}
                          onChange={value => value !== null && setJobConfig(value / 100, 'config.process[0].monitoring.memory_warning_threshold')}
                          min={50}
                          max={95}
                        />
                        <p className="text-xs text-gray-500 mt-1">% memory usage</p>
                      </FormGroup>

                      <FormGroup
                        label="Critical Threshold"
                        tooltip="Memory usage percentage that indicates critical issues"
                      >
                        <NumberInput
                          value={(jobConfig.config.process[0].monitoring?.memory_critical_threshold ?? 0.95) * 100}
                          onChange={value => value !== null && setJobConfig(value / 100, 'config.process[0].monitoring.memory_critical_threshold')}
                          min={60}
                          max={99}
                        />
                        <p className="text-xs text-gray-500 mt-1">% memory usage</p>
                      </FormGroup>
                    </div>
                  </div>
                )}

                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    <strong>After training:</strong> View the Performance tab in your job details to see memory usage timeline,
                    optimization recommendations, and any errors that occurred during training.
                  </p>
                </div>
              </div>
            )}

            {/* Step 13: Review */}
            {currentStepDef.id === 'review' && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="font-semibold mb-2">Model</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{selectedModelArch?.label || 'Not selected'}</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Training Name</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{jobConfig.config.name}</p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Dataset</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {datasetInfo ? `${datasetInfo.total_images} images` : 'Not analyzed'}
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Resolution</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {jobConfig.config.process[0].datasets?.[0]?.resolution?.[0] || 1024}x
                      {jobConfig.config.process[0].datasets?.[0]?.resolution?.[1] || 1024}
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Steps</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {jobConfig.config.process[0].train?.steps || 'Not set'}
                    </p>
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Batch Size</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {jobConfig.config.process[0].train?.batch_size || 1}
                      {jobConfig.config.process[0].train?.auto_scale_batch_size && ' (auto-scaling)'}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </Card>

          {/* Navigation */}
          <div className="flex justify-between mt-6">
            <div className="space-x-2">
              <Button
                className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400"
                onClick={onExit}
              >
                Exit Wizard
              </Button>
              {currentStep > 0 && (
                <Button
                  className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400"
                  onClick={prevStep}
                >
                  <FaChevronLeft className="inline mr-2" />
                  Previous
                </Button>
              )}
            </div>
            {currentStep < wizardSteps.length - 1 ? (
              <Button
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                onClick={nextStep}
              >
                Next
                <FaChevronRight className="inline ml-2" />
              </Button>
            ) : (
              <Button
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
                onClick={() => handleSubmit({ preventDefault: () => {} } as React.FormEvent<HTMLFormElement>)}
                disabled={status === 'saving'}
              >
                <FaCheck className="inline mr-2" />
                {status === 'saving' ? 'Saving...' : runId ? 'Update Job' : 'Create Job'}
              </Button>
            )}
          </div>
        </div>

        {/* Advisor Panel (30%) */}
        <div className="w-80 flex-shrink-0 hidden lg:block">
          <AdvisorPanel
            messages={advisorMessages}
            experienceLevel={userIntent?.experienceLevel || 'beginner'}
            performance={performancePrediction}
            currentStepId={currentStepDef.id}
          />
        </div>
      </div>
    </div>
  );
}
