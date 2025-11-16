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

  const selectedModelArch = useMemo(
    () => modelArchs.find(a => a.name === jobConfig.config.process[0].model?.arch),
    [jobConfig.config.process[0].model?.arch]
  );

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
                <FormGroup label="Model Quantization" tooltip="Reduce model precision to save VRAM">
                  <SelectInput
                    value={
                      jobConfig.config.process[0].model?.quantize === true
                        ? (jobConfig.config.process[0].model?.qtype?.startsWith('uint4') || jobConfig.config.process[0].model?.qtype?.startsWith('uint3') ? '4bit' : '8bit')
                        : 'none'
                    }
                    onChange={(value: string) => {
                      if (value === 'none') {
                        setJobConfig(false, 'config.process[0].model.quantize');
                      } else if (value === '8bit') {
                        setJobConfig(true, 'config.process[0].model.quantize');
                        setJobConfig('qfloat8', 'config.process[0].model.qtype');
                      } else if (value === '4bit') {
                        setJobConfig(true, 'config.process[0].model.quantize');
                        setJobConfig('uint4', 'config.process[0].model.qtype');
                      }
                    }}
                    options={[
                      { value: 'none', label: 'No Quantization (Full Precision)' },
                      { value: '8bit', label: '8-bit (Recommended for most cases)' },
                      { value: '4bit', label: '4-bit (Maximum VRAM savings)' }
                    ]}
                  />
                </FormGroup>

                {jobConfig.config.process[0].model?.quantize === true && (
                  <FormGroup label="Text Encoder Quantization" tooltip="Also quantize the text encoder to save more VRAM">
                    <SelectInput
                      value={jobConfig.config.process[0].model?.quantize_te === true ? 'yes' : 'no'}
                      onChange={(value: string) => {
                        if (value === 'yes') {
                          setJobConfig(true, 'config.process[0].model.quantize_te');
                          setJobConfig('qfloat8', 'config.process[0].model.qtype_te');
                        } else {
                          setJobConfig(false, 'config.process[0].model.quantize_te');
                        }
                      }}
                      options={[
                        { value: 'no', label: 'No (Full Precision Text Encoder)' },
                        { value: 'yes', label: 'Yes (8-bit Text Encoder)' }
                      ]}
                    />
                  </FormGroup>
                )}
              </div>
            )}

            {/* Step 3: Target Configuration */}
            {currentStepDef.id === 'target' && (
              <div className="space-y-4">
                <FormGroup label="Training Target Type" tooltip="LoRA is recommended for most cases">
                  <SelectInput
                    value={jobConfig.config.process[0].network?.type || 'lora'}
                    onChange={value => setJobConfig(value, 'config.process[0].network.type')}
                    options={[
                      { value: 'lora', label: 'LoRA (Low-Rank Adaptation)' },
                      { value: 'lokr', label: 'LoKr (Kronecker Product)' }
                    ]}
                  />
                </FormGroup>

                <div className="grid grid-cols-2 gap-4">
                  <FormGroup label="Rank" tooltip="Higher rank = more learning capacity, larger file size">
                    <NumberInput
                      value={jobConfig.config.process[0].network?.linear ?? 16}
                      onChange={value => setJobConfig(value, 'config.process[0].network.linear')}
                      min={1}
                      max={128}
                    />
                  </FormGroup>
                  <FormGroup label="Alpha" tooltip="Usually set equal to rank">
                    <NumberInput
                      value={jobConfig.config.process[0].network?.linear_alpha ?? 16}
                      onChange={value => setJobConfig(value, 'config.process[0].network.linear_alpha')}
                      min={1}
                      max={128}
                    />
                  </FormGroup>
                </div>
              </div>
            )}

            {/* Step 4: Dataset */}
            {currentStepDef.id === 'dataset' && (
              <div className="space-y-4">
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
                      disabled={isAnalyzing || !jobConfig.config.process[0].datasets?.[0]?.folder_path}
                    >
                      {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                    </Button>
                  </div>
                </FormGroup>

                {analysisError && (
                  <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-300 rounded-md">
                    <p className="text-red-800 dark:text-red-300">{analysisError}</p>
                  </div>
                )}

                {datasetInfo && (
                  <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-300 rounded-md">
                    <h3 className="font-semibold mb-2">Dataset Analysis</h3>
                    <ul className="space-y-1 text-sm">
                      <li>✓ Total images: {datasetInfo.total_images}</li>
                      <li>
                        ✓ Most common resolution: {datasetInfo.most_common_resolution[0]}x
                        {datasetInfo.most_common_resolution[1]}
                      </li>
                      <li>✓ Captions: {datasetInfo.has_captions ? `Found (.${datasetInfo.caption_ext})` : 'Not found'}</li>
                    </ul>
                  </div>
                )}

                <FormGroup label="Trigger Word" tooltip="Word to use in prompts to activate your LoRA">
                  <TextInput
                    value={jobConfig.config.process[0].trigger_word || ''}
                    onChange={value => setJobConfig(value, 'config.process[0].trigger_word')}
                    placeholder="e.g., ohwx, sks, or unique identifier"
                  />
                </FormGroup>
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
                    <FormGroup label="Initial">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.batch_size ?? 1}
                        onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                        min={1}
                        max={32}
                      />
                    </FormGroup>
                    <FormGroup label="Min">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.min_batch_size ?? 1}
                        onChange={value => setJobConfig(value, 'config.process[0].train.min_batch_size')}
                        min={1}
                        max={32}
                      />
                    </FormGroup>
                    <FormGroup label="Max">
                      <NumberInput
                        value={jobConfig.config.process[0].train?.max_batch_size ?? 8}
                        onChange={value => setJobConfig(value, 'config.process[0].train.max_batch_size')}
                        min={1}
                        max={32}
                      />
                    </FormGroup>
                  </div>
                ) : (
                  <FormGroup label="Batch Size">
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
                      { value: 'adafactor', label: 'Adafactor (Memory Efficient)' }
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

                <FormGroup label="Classifier-Free Guidance Training">
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

                <FormGroup label="Data Augmentation">
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
                    <FormGroup label="EMA Decay">
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
                    <FormGroup label="Preservation Strength">
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

                    <FormGroup label="Preservation Class">
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
              </div>
            )}

            {/* Step 9: Training Core */}
            {currentStepDef.id === 'training' && (
              <div className="space-y-4">
                <FormGroup label="Training Name">
                  <TextInput
                    value={jobConfig.config.name}
                    onChange={value => setJobConfig(value, 'config.name')}
                    placeholder="my-lora-training"
                  />
                </FormGroup>

                <FormGroup label="Training Steps">
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
              </div>
            )}

            {/* Step 11: Save Settings */}
            {currentStepDef.id === 'save' && (
              <div className="space-y-4">
                <FormGroup label="Save Every N Steps">
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

                <FormGroup label="Verbose Output">
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
                          onChange={value => setJobConfig(value / 100, 'config.process[0].monitoring.memory_warning_threshold')}
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
                          onChange={value => setJobConfig(value / 100, 'config.process[0].monitoring.memory_critical_threshold')}
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
                onClick={handleSubmit as any}
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
