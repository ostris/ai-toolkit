'use client';
import { useState, useEffect, useMemo } from 'react';
import { JobConfig, SelectOption } from '@/types';
import { TextInput, SelectInput, Checkbox, FormGroup, NumberInput } from '@/components/formInputs';
import Card from '@/components/Card';
import { Button } from '@headlessui/react';
import { FaChevronRight, FaChevronLeft, FaCheck } from 'react-icons/fa';
import { modelArchs, ModelArch } from './options';
import { apiClient } from '@/utils/api';
import { handleModelArchChange } from './utils';

type Props = {
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
};

type DatasetInfo = {
  total_images: number;
  most_common_resolution: [number, number];
  resolutions: Record<string, number>;
  has_captions: boolean;
  caption_ext: string;
  formats: Record<string, number>;
};

const wizardSteps = [
  { id: 'model', title: 'Select Model', description: 'Choose your base model architecture' },
  { id: 'dataset', title: 'Dataset', description: 'Analyze and configure your dataset' },
  { id: 'resolution', title: 'Resolution', description: 'Choose training resolution' },
  { id: 'training', title: 'Training', description: 'Configure training parameters' },
  { id: 'review', title: 'Review', description: 'Review and save configuration' },
];

export default function GuidedWizard({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  onExit,
}: Props) {
  const [currentStep, setCurrentStep] = useState(0);
  const [datasetPath, setDatasetPath] = useState('');
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  const step = wizardSteps[currentStep];

  // Filter to image models only
  const imageModels = useMemo(() => {
    return modelArchs.filter(m => m.group === 'image');
  }, []);

  const selectedModelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch);
  }, [jobConfig.config.process[0].model.arch]);

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
    } catch (error: any) {
      setAnalysisError(error.response?.data?.error || 'Failed to analyze dataset');
      setDatasetInfo(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

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

  const getModelResolutionInfo = (archName: string) => {
    const info: Record<string, any> = {
      sd1: { native: 512, recommended: [512, 768], supportsAspect: false },
      sd2: { native: 768, recommended: [512, 768], supportsAspect: false },
      sd3: { native: 1024, recommended: [512, 768, 1024], supportsAspect: true },
      sd15: { native: 512, recommended: [512, 768], supportsAspect: false },
      sdxl: { native: 1024, recommended: [768, 1024, 1536], supportsAspect: true },
      flux: { native: 1024, recommended: [512, 768, 1024], supportsAspect: true },
      flux_kontext: { native: 1024, recommended: [512, 768, 1024], supportsAspect: true },
      flex1: { native: 1024, recommended: [512, 768, 1024], supportsAspect: true },
      flex2: { native: 1024, recommended: [512, 768, 1024], supportsAspect: true },
      chroma: { native: 1024, recommended: [512, 768, 1024], supportsAspect: true },
      lumina2: { native: 1024, recommended: [768, 1024, 1536], supportsAspect: true },
      qwen_image: { native: 1024, recommended: [768, 1024, 1536, 2048], supportsAspect: true },
      qwen_image_edit: { native: 1024, recommended: [768, 1024, 1536, 2048], supportsAspect: true },
      qwen_image_edit_plus: { native: 1024, recommended: [768, 1024, 1536, 2048], supportsAspect: true },
      hidream: { native: 1024, recommended: [768, 1024, 1536], supportsAspect: true },
      hidream_e1: { native: 1024, recommended: [768, 1024, 1536], supportsAspect: true },
      omnigen2: { native: 1024, recommended: [768, 1024, 1536], supportsAspect: true },
    };
    return info[archName] || { native: 1024, recommended: [512, 768, 1024], supportsAspect: true };
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {wizardSteps.map((s, idx) => (
            <div key={s.id} className="flex flex-col items-center flex-1">
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                  idx < currentStep
                    ? 'bg-green-500 border-green-500 text-white'
                    : idx === currentStep
                    ? 'bg-blue-500 border-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 border-gray-300 dark:border-gray-600'
                }`}
              >
                {idx < currentStep ? <FaCheck /> : idx + 1}
              </div>
              <div className="text-xs mt-1 text-center">{s.title}</div>
            </div>
          ))}
        </div>
        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${((currentStep + 1) / wizardSteps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Step Content */}
      <Card className="p-6">
        <h2 className="text-2xl font-bold mb-2">{step.title}</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">{step.description}</p>

        {/* Step 1: Model Selection */}
        {step.id === 'model' && (
          <div className="space-y-4">
            <FormGroup label="Base Model Architecture" tooltip="Select the model architecture to train on">
              <div className="grid grid-cols-2 gap-3">
                {imageModels.map(model => (
                  <button
                    key={model.name}
                    onClick={() => {
                      handleModelArchChange(
                        jobConfig.config.process[0].model.arch,
                        model.name,
                        jobConfig,
                        setJobConfig
                      );
                    }}
                    className={`p-4 rounded-lg border-2 transition-colors text-left ${
                      jobConfig.config.process[0].model.arch === model.name
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
                  value={jobConfig.config.process[0].model.name_or_path}
                  onChange={value => setJobConfig(value, 'config.process[0].model.name_or_path')}
                  placeholder="e.g., black-forest-labs/FLUX.1-dev"
                />
              </FormGroup>
            )}
          </div>
        )}

        {/* Step 2: Dataset */}
        {step.id === 'dataset' && (
          <div className="space-y-4">
            <FormGroup label="Dataset Folder" tooltip="Path to your training images">
              <div className="flex gap-2">
                <div className="flex-1">
                  <SelectInput
                    value={datasetPath || jobConfig.config.process[0].datasets[0].folder_path}
                    onChange={value => {
                      setDatasetPath(value);
                      setJobConfig(value, 'config.process[0].datasets[0].folder_path');
                    }}
                    options={datasetOptions}
                  />
                </div>
                <Button
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                  onClick={() => analyzeDataset(datasetPath || jobConfig.config.process[0].datasets[0].folder_path)}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze'}
                </Button>
              </div>
            </FormGroup>

            {!datasetInfo && !analysisError && !isAnalyzing && (
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-md">
                <p className="text-blue-800 dark:text-blue-300">
                  Please analyze your dataset to continue. This will scan the images and provide recommendations.
                </p>
              </div>
            )}

            {analysisError && (
              <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-300 dark:border-red-700 rounded-md">
                <p className="text-red-800 dark:text-red-300">{analysisError}</p>
              </div>
            )}

            {datasetInfo && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-300 dark:border-green-700 rounded-md">
                <h3 className="font-semibold mb-2">Dataset Analysis</h3>
                <ul className="space-y-1 text-sm">
                  <li>✓ Total images: {datasetInfo.total_images}</li>
                  <li>
                    ✓ Most common resolution: {datasetInfo.most_common_resolution[0]}x
                    {datasetInfo.most_common_resolution[1]}
                  </li>
                  <li>✓ Unique resolutions: {Object.keys(datasetInfo.resolutions).length}</li>
                  <li>✓ Captions: {datasetInfo.has_captions ? `Found (.${datasetInfo.caption_ext})` : 'Not found'}</li>
                  <li>
                    ✓ Formats: {Object.entries(datasetInfo.formats).map(([ext, count]) => `${ext}(${count})`).join(', ')}
                  </li>
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Step 3: Resolution */}
        {step.id === 'resolution' && (
          <div className="space-y-4">
            {datasetInfo && selectedModelArch && (
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md mb-4">
                <h3 className="font-semibold mb-2">Resolution Recommendation</h3>
                <p className="text-sm mb-2">
                  • Dataset: {datasetInfo.most_common_resolution[0]}x{datasetInfo.most_common_resolution[1]} (
                  {Math.round((datasetInfo.most_common_resolution[0] * datasetInfo.most_common_resolution[1]) / 1000000)}MP)
                </p>
                <p className="text-sm mb-2">
                  • Model: {selectedModelArch.label} (
                  {getModelResolutionInfo(selectedModelArch.name).supportsAspect
                    ? 'supports aspect ratios'
                    : 'requires square images'}
                  )
                </p>
                <p className="text-sm mb-2">
                  • Recommended training resolutions: {getModelResolutionInfo(selectedModelArch.name).recommended.join(', ')}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  💡 Tip: Training at lower resolutions (1024-1536) is faster and uses less VRAM, while still producing
                  LoRAs that work well at higher resolutions during inference. For your {
                  Math.round((datasetInfo.most_common_resolution[0] * datasetInfo.most_common_resolution[1]) / 1000000)
                  }MP images, consider 1024-1536 with aspect ratio bucketing enabled.
                </p>
              </div>
            )}

            <FormGroup label="Training Resolution" tooltip="Resolution for training (width)">
              <NumberInput
                value={jobConfig.config.process[0].datasets[0].resolution?.[0] || 1024}
                onChange={value => {
                  setJobConfig([value, value], 'config.process[0].datasets[0].resolution');
                }}
                min={256}
                max={2048}
                step={64}
              />
            </FormGroup>

            <FormGroup label="Enable Bucket Sampling" tooltip="Allow different aspect ratios during training">
              <Checkbox
                checked={!!jobConfig.config.process[0].datasets[0].enable_ar_bucket}
                onChange={value => setJobConfig(value, 'config.process[0].datasets[0].enable_ar_bucket')}
                label="Use aspect ratio bucketing (recommended for non-square images)"
              />
            </FormGroup>
          </div>
        )}

        {/* Step 4: Training Parameters */}
        {step.id === 'training' && (
          <div className="space-y-4">
            {datasetInfo && (() => {
              // Calculate recommended values based on resolution and dataset size
              const resolution = jobConfig.config.process[0].datasets[0].resolution?.[0] || 1024;
              const recommendedSteps = Math.max(500, Math.min(3000, datasetInfo.total_images * 10));

              // Calculate batch size based on resolution and dataset size
              // Memory usage roughly scales with resolution^2
              let recommendedInitialBatch = 1;
              let recommendedMaxBatch = 1;
              if (resolution <= 512) {
                recommendedInitialBatch = 2;
                recommendedMaxBatch = datasetInfo.total_images < 50 ? 4 : datasetInfo.total_images < 200 ? 8 : 16;
              } else if (resolution <= 768) {
                recommendedInitialBatch = 1;
                recommendedMaxBatch = datasetInfo.total_images < 50 ? 2 : datasetInfo.total_images < 200 ? 4 : 8;
              } else if (resolution <= 1024) {
                recommendedInitialBatch = 1;
                recommendedMaxBatch = datasetInfo.total_images < 50 ? 2 : datasetInfo.total_images < 200 ? 4 : 6;
              } else if (resolution <= 1536) {
                recommendedInitialBatch = 1;
                recommendedMaxBatch = 2;
              } else {
                recommendedInitialBatch = 1;
                recommendedMaxBatch = 2;
              }

              return (
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md mb-4">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-semibold">Training Recommendations</h3>
                    <Button
                      className="px-3 py-1 text-xs bg-blue-600 text-white rounded-md hover:bg-blue-700"
                      onClick={() => {
                        setJobConfig(recommendedSteps, 'config.process[0].train.steps');
                        setJobConfig(recommendedInitialBatch, 'config.process[0].train.batch_size');
                        setJobConfig(true, 'config.process[0].train.auto_scale_batch_size');
                        setJobConfig(1, 'config.process[0].train.min_batch_size');
                        setJobConfig(recommendedMaxBatch, 'config.process[0].train.max_batch_size');
                        setJobConfig(0.0001, 'config.process[0].train.lr');
                      }}
                    >
                      Apply Recommendations
                    </Button>
                  </div>
                  <p className="text-sm mb-2">
                    • Dataset Size: {datasetInfo.total_images} images
                  </p>
                  <p className="text-sm mb-2">
                    • Training Resolution: {resolution}x{resolution}
                  </p>
                  <p className="text-sm mb-2">
                    • Recommended Steps: {recommendedSteps}
                    <span className="text-gray-600 dark:text-gray-400">
                      {' '}(~{Math.round(recommendedSteps / datasetInfo.total_images)} epochs)
                    </span>
                  </p>
                  <p className="text-sm mb-2">
                    • Recommended Batch Size: Auto-scaling from {recommendedInitialBatch} to {recommendedMaxBatch}
                    <span className="text-gray-600 dark:text-gray-400">
                      {' '}(optimized for {resolution}px)
                    </span>
                  </p>
                  <p className="text-sm mb-2">
                    • Recommended Learning Rate: 0.0001
                    <span className="text-gray-600 dark:text-gray-400"> (standard for LoRA training)</span>
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                    💡 Tip: Auto-scaling batch size is enabled by default. It starts at {recommendedInitialBatch} and automatically
                    increases up to {recommendedMaxBatch} based on available VRAM, preventing OOM errors while maximizing efficiency.
                  </p>
                </div>
              );
            })()}

            <FormGroup label="Training Name">
              <TextInput
                value={jobConfig.config.name}
                onChange={value => setJobConfig(value, 'config.name')}
                placeholder="my-lora-training"
              />
            </FormGroup>

            <FormGroup label="Training Steps">
              <NumberInput
                value={jobConfig.config.process[0].train?.steps || 1000}
                onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                min={100}
                max={10000}
              />
            </FormGroup>

            <FormGroup label="Batch Size Configuration">
              <Checkbox
                checked={!!jobConfig.config.process[0].train?.auto_scale_batch_size}
                onChange={value => setJobConfig(value, 'config.process[0].train.auto_scale_batch_size')}
                label="Enable auto-scaling batch size (recommended)"
              />
            </FormGroup>

            {jobConfig.config.process[0].train?.auto_scale_batch_size ? (
              <div className="grid grid-cols-3 gap-4">
                <FormGroup label="Initial Batch Size">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.batch_size || 1}
                    onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                    min={1}
                    max={32}
                  />
                </FormGroup>
                <FormGroup label="Min Batch Size">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.min_batch_size || 1}
                    onChange={value => setJobConfig(value, 'config.process[0].train.min_batch_size')}
                    min={1}
                    max={32}
                  />
                </FormGroup>
                <FormGroup label="Max Batch Size">
                  <NumberInput
                    value={jobConfig.config.process[0].train?.max_batch_size || 8}
                    onChange={value => setJobConfig(value, 'config.process[0].train.max_batch_size')}
                    min={1}
                    max={32}
                  />
                </FormGroup>
              </div>
            ) : (
              <FormGroup label="Batch Size">
                <NumberInput
                  value={jobConfig.config.process[0].train?.batch_size || 1}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  min={1}
                  max={32}
                />
              </FormGroup>
            )}

            <FormGroup label="Learning Rate">
              <NumberInput
                value={jobConfig.config.process[0].train?.lr || 0.0001}
                onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                min={0.00001}
                max={0.001}
              />
            </FormGroup>
          </div>
        )}

        {/* Step 5: Review */}
        {step.id === 'review' && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold mb-2">Model</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">{selectedModelArch?.label}</p>
                <p className="text-xs text-gray-500 dark:text-gray-500">{jobConfig.config.process[0].model.arch}</p>
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
                  {jobConfig.config.process[0].datasets[0].resolution?.[0] || 1024}x
                  {jobConfig.config.process[0].datasets[0].resolution?.[1] || 1024}
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Training Steps</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {jobConfig.config.process[0].train?.steps || 'Not set'}
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Batch Size</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {jobConfig.config.process[0].train?.batch_size || 1}
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
            className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400 dark:hover:bg-gray-600"
            onClick={onExit}
          >
            Exit Wizard
          </Button>
          {currentStep > 0 && (
            <Button
              className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400 dark:hover:bg-gray-600"
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
            disabled={step.id === 'dataset' && !datasetInfo}
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
  );
}
