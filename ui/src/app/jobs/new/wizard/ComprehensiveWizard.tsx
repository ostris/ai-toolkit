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
                <StepRenderer
                  stepId="memory"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}

            {/* Step 7: Optimizer */}
            {currentStepDef.id === 'optimizer' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="optimizer"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}

            {/* Step 8: Advanced Training */}
            {currentStepDef.id === 'advanced' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="advanced"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}

            {/* Step 9: Regularization */}
            {currentStepDef.id === 'regularization' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="regularization"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}
            {/* Step 9: Training Core */}
            {currentStepDef.id === 'training' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="training"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}
            {currentStepDef.id === 'sampling' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="sampling"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}
            {currentStepDef.id === 'save' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="save"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}
            {currentStepDef.id === 'logging' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="logging"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}
            {currentStepDef.id === 'monitoring' && (
              <div className="space-y-4">
                <StepRenderer
                  stepId="monitoring"
                  selectedModel={jobConfig.config.process[0].model?.arch || ''}
                  jobConfig={jobConfig}
                  onConfigChange={handleStepRendererConfigChange}
                />
              </div>
            )}
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
