'use client';

import { AdvisorMessage, UserIntent, PerformancePrediction } from '../utils/types';
import { FaInfoCircle, FaLightbulb, FaExclamationTriangle, FaExclamationCircle, FaChartBar } from 'react-icons/fa';

interface Props {
  messages: AdvisorMessage[];
  experienceLevel: UserIntent['experienceLevel'];
  performance?: PerformancePrediction;
  currentStepId: string;
}

// Educational content for each step
const educationalContent: Record<string, { title: string; content: string; beginnerExtra?: string }> = {
  model: {
    title: 'Model Architecture',
    content: 'The base model defines what your LoRA can generate. Each architecture has different capabilities and VRAM requirements.',
    beginnerExtra: 'FLUX models are great for high-quality images. SDXL is more established and widely supported. SD1.5 uses less VRAM but lower resolution.'
  },
  quantization: {
    title: 'Quantization',
    content: 'Quantization reduces model precision to save VRAM, with minimal quality impact. 8-bit is a good balance, 4-bit saves more but may affect quality.',
    beginnerExtra: 'Think of it like compressing an image - you save space but might lose some detail. For most training, 8-bit works great.'
  },
  target: {
    title: 'Training Target',
    content: 'LoRA (Low-Rank Adaptation) trains a small adapter that modifies the base model. Rank determines how much the LoRA can learn - higher = more capacity but larger file.',
    beginnerExtra: 'Rank 16-32 works well for most cases. Higher ranks (64+) are for complex styles or large datasets.'
  },
  dataset: {
    title: 'Dataset Setup',
    content: 'Your training images and captions define what the LoRA learns. Quality and variety of images matter more than quantity.',
    beginnerExtra: 'Aim for 20-200 high-quality images. Each image should have a text caption describing it. The trigger word is what you\'ll use in prompts to activate your LoRA.'
  },
  resolution: {
    title: 'Resolution & Augmentation',
    content: 'Training resolution affects VRAM usage and what the LoRA learns about detail. Augmentation like flipping can increase effective dataset size.',
    beginnerExtra: 'Higher resolution uses more VRAM but captures more detail. Aspect ratio bucketing lets you use images of different shapes without cropping.'
  },
  memory: {
    title: 'Memory & Batch Configuration',
    content: 'Batch size determines how many images are processed together. Larger batches can improve training stability but use more VRAM.',
    beginnerExtra: 'Auto-scaling batch size is recommended - it automatically finds the optimal size for your GPU without crashing.'
  },
  optimizer: {
    title: 'Optimizer & Learning Rate',
    content: 'The optimizer controls how the model learns. Learning rate determines the speed of learning - too high causes instability, too low is slow.',
    beginnerExtra: 'AdamW8bit is efficient and works well. Learning rate of 1e-4 (0.0001) is standard for LoRA training.'
  },
  regularization: {
    title: 'Regularization',
    content: 'Regularization techniques prevent overfitting, which is when the model memorizes training images instead of learning concepts.',
    beginnerExtra: 'DOP (Differential Output Preservation) helps maintain model quality for prompts not related to your training. Recommended for most cases.'
  },
  training: {
    title: 'Training Core',
    content: 'Steps determine how long you train. More steps = more learning, but too many can overfit. Timestep settings affect what aspects the model focuses on.',
    beginnerExtra: 'A good rule: 10-15 steps per training image. For 100 images, try 1000-1500 steps.'
  },
  sampling: {
    title: 'Sample Generation',
    content: 'Sampling generates preview images during training so you can monitor progress without stopping the training run.',
    beginnerExtra: 'Set sample_every to generate previews every N steps. This helps you see if training is going well.'
  },
  save: {
    title: 'Save Settings',
    content: 'Checkpoints let you save the model during training. You can later pick the best performing checkpoint.',
    beginnerExtra: 'Saving every 250-500 steps is common. Keep 3-5 recent checkpoints in case you need to roll back.'
  },
  review: {
    title: 'Review Configuration',
    content: 'Double-check all settings before starting. The summary shows estimated resource usage and potential issues.',
    beginnerExtra: 'Look for any warnings (yellow) or errors (red). Green checkmarks mean everything looks good!'
  }
};

export default function AdvisorPanel({ messages, experienceLevel, performance, currentStepId }: Props) {
  const education = educationalContent[currentStepId] || {
    title: 'Configuration',
    content: 'Configure the settings for this section.'
  };

  // Filter messages based on type
  const errors = messages.filter(m => m.type === 'error');
  const warnings = messages.filter(m => m.type === 'warning');
  const tips = messages.filter(m => m.type === 'tip' || m.type === 'info');

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 border-l dark:border-gray-700 overflow-y-auto">
      {/* Educational Section */}
      <div className="p-4 border-b dark:border-gray-700">
        <div className="flex items-center gap-2 mb-2">
          <FaInfoCircle className="text-blue-500" />
          <h3 className="font-semibold">{education.title}</h3>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400">{education.content}</p>
        {experienceLevel === 'beginner' && education.beginnerExtra && (
          <p className="text-sm text-blue-600 dark:text-blue-400 mt-2 bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
            {education.beginnerExtra}
          </p>
        )}
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="p-4 border-b dark:border-gray-700 bg-red-50 dark:bg-red-900/10">
          <div className="flex items-center gap-2 mb-2">
            <FaExclamationCircle className="text-red-500" />
            <h3 className="font-semibold text-red-700 dark:text-red-300">Errors</h3>
          </div>
          <div className="space-y-2">
            {errors.map((msg, idx) => (
              <div key={idx} className="text-sm">
                <div className="font-medium text-red-700 dark:text-red-300">{msg.title}</div>
                <div className="text-red-600 dark:text-red-400">{msg.message}</div>
                {msg.autoFix && (
                  <button
                    onClick={msg.autoFix}
                    className="text-xs text-red-500 underline mt-1 hover:text-red-700"
                  >
                    Auto-fix
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="p-4 border-b dark:border-gray-700 bg-yellow-50 dark:bg-yellow-900/10">
          <div className="flex items-center gap-2 mb-2">
            <FaExclamationTriangle className="text-yellow-500" />
            <h3 className="font-semibold text-yellow-700 dark:text-yellow-300">Warnings</h3>
          </div>
          <div className="space-y-2">
            {warnings.map((msg, idx) => (
              <div key={idx} className="text-sm">
                <div className="font-medium text-yellow-700 dark:text-yellow-300">{msg.title}</div>
                <div className="text-yellow-600 dark:text-yellow-400">{msg.message}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tips & Recommendations */}
      {tips.length > 0 && (
        <div className="p-4 border-b dark:border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <FaLightbulb className="text-green-500" />
            <h3 className="font-semibold">Recommendations</h3>
          </div>
          <div className="space-y-2">
            {tips.map((msg, idx) => (
              <div key={idx} className="text-sm">
                <div className="font-medium text-green-700 dark:text-green-300">{msg.title}</div>
                <div className="text-gray-600 dark:text-gray-400">{msg.message}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Predictions */}
      {performance && (
        <div className="p-4 flex-grow">
          <div className="flex items-center gap-2 mb-2">
            <FaChartBar className="text-purple-500" />
            <h3 className="font-semibold">Performance Estimates</h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">VRAM Usage:</span>
              <span className="font-medium">{performance.estimatedVRAM}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Step Time:</span>
              <span className="font-medium">{performance.estimatedStepTime}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Time:</span>
              <span className="font-medium">{performance.totalTrainingTime}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Disk Space:</span>
              <span className="font-medium">{performance.diskSpaceNeeded}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Memory Usage:</span>
              <span className="font-medium">{performance.memoryUsage}</span>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {tips.length === 0 && warnings.length === 0 && errors.length === 0 && !performance && (
        <div className="p-4 flex-grow">
          <div className="text-center text-gray-500 dark:text-gray-400 py-8">
            <FaLightbulb className="text-3xl mx-auto mb-2 opacity-50" />
            <p className="text-sm">Recommendations will appear as you configure settings</p>
          </div>
        </div>
      )}
    </div>
  );
}
