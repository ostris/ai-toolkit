'use client';

import { useState, useEffect } from 'react';
import { SystemProfile, UserIntent } from '../utils/types';
import { FaCheck, FaCog, FaMemory, FaHdd, FaMicrochip } from 'react-icons/fa';
import { Button } from '@headlessui/react';
import { NumberInput, SelectInput } from '@/components/formInputs';

interface Props {
  onComplete: (profile: SystemProfile, intent: UserIntent) => void;
  onCancel: () => void;
}

export default function PreflightModal({ onComplete, onCancel }: Props) {
  const [step, setStep] = useState<'detecting' | 'hardware' | 'intent'>('detecting');
  const [profile, setProfile] = useState<SystemProfile | null>(null);
  const [detectionError, setDetectionError] = useState<string | null>(null);

  // User overrides for hardware
  const [gpuType, setGpuType] = useState<'nvidia' | 'amd' | 'unified_memory' | 'cpu_only'>('nvidia');
  const [gpuName, setGpuName] = useState('');
  const [vramGB, setVramGB] = useState(24);
  const [totalRAM, setTotalRAM] = useState(64);
  const [availableRAM, setAvailableRAM] = useState(32);
  const [storageType, setStorageType] = useState<'hdd' | 'ssd' | 'nvme'>('ssd');
  const [cpuCores, setCpuCores] = useState(8);

  // User intent
  const [trainingType, setTrainingType] = useState<'person' | 'style' | 'object' | 'concept' | 'other'>('style');
  const [priority, setPriority] = useState<'quality' | 'speed' | 'memory_efficiency'>('speed');
  const [experienceLevel, setExperienceLevel] = useState<'beginner' | 'intermediate' | 'advanced'>('beginner');

  // Auto-detect system profile
  useEffect(() => {
    async function detectSystem() {
      try {
        const response = await fetch('/api/system/profile');
        const data = await response.json();

        if (data.error) {
          setDetectionError(data.error);
        } else {
          setProfile(data);
          // Pre-fill overrides with detected values
          setGpuType(data.gpu.type);
          setGpuName(data.gpu.name);
          setVramGB(data.gpu.vramGB);
          setTotalRAM(data.memory.totalRAM);
          setAvailableRAM(data.memory.availableRAM);
          setStorageType(data.storage.type);
          setCpuCores(data.cpu.cores);
        }
      } catch (error) {
        setDetectionError('Failed to connect to system profiler');
      }
      setStep('hardware');
    }

    detectSystem();
  }, []);

  const handleConfirmHardware = () => {
    setStep('intent');
  };

  const handleComplete = () => {
    // For unified memory systems, vramGB field contains the unified memory amount
    // and totalRAM/availableRAM should match since it's all one pool
    const isUnified = gpuType === 'unified_memory';

    const finalProfile: SystemProfile = {
      gpu: {
        type: gpuType,
        name: gpuName || 'Custom GPU',
        vramGB: vramGB,
        isUnifiedMemory: isUnified
      },
      memory: {
        // For unified memory, use the vramGB value (which is the unified memory amount)
        totalRAM: isUnified ? vramGB : totalRAM,
        availableRAM: isUnified ? Math.floor(vramGB * 0.95) : availableRAM, // 95% available for unified
        unifiedMemory: isUnified ? vramGB : undefined
      },
      storage: {
        type: storageType,
        availableSpaceGB: profile?.storage.availableSpaceGB || 100
      },
      cpu: {
        cores: cpuCores,
        name: profile?.cpu.name || 'Unknown CPU'
      }
    };

    const intent: UserIntent = {
      trainingType,
      priority,
      experienceLevel
    };

    onComplete(finalProfile, intent);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <h2 className="text-2xl font-bold mb-4">Configure Your Training Environment</h2>

          {/* Step 1: Detecting System */}
          {step === 'detecting' && (
            <div className="text-center py-8">
              <FaCog className="animate-spin text-4xl mx-auto mb-4 text-blue-500" />
              <p className="text-gray-600 dark:text-gray-400">Detecting system hardware...</p>
            </div>
          )}

          {/* Step 2: Hardware Profile */}
          {step === 'hardware' && (
            <div className="space-y-6">
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                We&apos;ve detected your hardware. Please verify and adjust if needed.
              </p>

              {detectionError && (
                <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700 rounded-md mb-4">
                  <p className="text-yellow-800 dark:text-yellow-300 text-sm">
                    Detection issue: {detectionError}. Please configure manually.
                  </p>
                </div>
              )}

              {/* Unified Memory Suggestion when VRAM is 0 */}
              {vramGB === 0 && gpuType !== 'unified_memory' && gpuType !== 'cpu_only' && (
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-300 dark:border-blue-700 rounded-md mb-4">
                  <p className="text-blue-800 dark:text-blue-300 text-sm mb-2">
                    <strong>GPU VRAM detected as 0.</strong> Are you using a unified memory system (Apple Silicon, NVIDIA Grace, DGX Spark)?
                  </p>
                  <button
                    onClick={() => {
                      setGpuType('unified_memory');
                      setVramGB(totalRAM); // Use total RAM as unified memory
                    }}
                    className="text-sm text-blue-600 dark:text-blue-400 underline hover:text-blue-800"
                  >
                    Yes, switch to Unified Memory
                  </button>
                  <span className="text-gray-400 mx-2">|</span>
                  <button
                    onClick={() => setGpuType('cpu_only')}
                    className="text-sm text-gray-600 dark:text-gray-400 underline hover:text-gray-800"
                  >
                    No, use CPU only
                  </button>
                </div>
              )}

              {/* GPU Section */}
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-3">
                  <FaMicrochip className="text-blue-500" />
                  <h3 className="font-semibold">GPU</h3>
                  {profile?.gpu && profile.gpu.type !== 'cpu_only' && (
                    <FaCheck className="text-green-500 ml-auto" />
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">GPU Type</label>
                    <SelectInput
                      value={gpuType}
                      onChange={(value: any) => setGpuType(value)}
                      options={[
                        { value: 'nvidia', label: 'NVIDIA (Discrete)' },
                        { value: 'amd', label: 'AMD (ROCm)' },
                        { value: 'unified_memory', label: 'Unified Memory (Apple Silicon, DGX Spark, Grace)' },
                        { value: 'cpu_only', label: 'CPU Only' }
                      ]}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      {gpuType === 'unified_memory' ? 'Unified Memory (GB)' : 'VRAM (GB)'}
                    </label>
                    <NumberInput
                      value={vramGB}
                      onChange={setVramGB}
                      min={0}
                      max={256}
                      step={1}
                    />
                  </div>
                </div>

                {gpuName && gpuType !== 'cpu_only' && (
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                    Detected: {gpuName}
                  </p>
                )}

                {gpuType === 'unified_memory' && (
                  <p className="text-sm text-blue-600 dark:text-blue-400 mt-2">
                    Unified memory: GPU and CPU share the same memory pool
                  </p>
                )}
              </div>

              {/* Memory Section - Only show for discrete GPU systems */}
              {gpuType !== 'unified_memory' && (
                <div className="border rounded-lg p-4 dark:border-gray-700">
                  <div className="flex items-center gap-2 mb-3">
                    <FaMemory className="text-purple-500" />
                    <h3 className="font-semibold">System Memory</h3>
                    <FaCheck className="text-green-500 ml-auto" />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Total RAM (GB)</label>
                      <NumberInput
                        value={totalRAM}
                        onChange={setTotalRAM}
                        min={4}
                        max={1024}
                        step={1}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Available RAM (GB)</label>
                      <NumberInput
                        value={availableRAM}
                        onChange={setAvailableRAM}
                        min={1}
                        max={totalRAM}
                        step={1}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Storage Section */}
              <div className="border rounded-lg p-4 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-3">
                  <FaHdd className="text-green-500" />
                  <h3 className="font-semibold">Storage</h3>
                  <FaCheck className="text-green-500 ml-auto" />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">Storage Type</label>
                    <SelectInput
                      value={storageType}
                      onChange={(value: any) => setStorageType(value)}
                      options={[
                        { value: 'hdd', label: 'HDD (Slow)' },
                        { value: 'ssd', label: 'SSD (Fast)' },
                        { value: 'nvme', label: 'NVMe (Fastest)' }
                      ]}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">CPU Cores</label>
                    <NumberInput
                      value={cpuCores}
                      onChange={setCpuCores}
                      min={1}
                      max={128}
                      step={1}
                    />
                  </div>
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <Button
                  className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400"
                  onClick={onCancel}
                >
                  Cancel
                </Button>
                <Button
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                  onClick={handleConfirmHardware}
                >
                  Confirm Hardware
                </Button>
              </div>
            </div>
          )}

          {/* Step 3: User Intent */}
          {step === 'intent' && (
            <div className="space-y-6">
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Tell us about your training goals so we can provide better recommendations.
              </p>

              <div>
                <label className="block font-medium mb-2">What are you training?</label>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { value: 'person', label: 'Person/Character', desc: 'Faces, bodies, specific people' },
                    { value: 'style', label: 'Art Style', desc: 'Visual aesthetics, artistic styles' },
                    { value: 'object', label: 'Object/Product', desc: 'Specific items, products' },
                    { value: 'concept', label: 'Concept', desc: 'Abstract ideas, poses, compositions' },
                    { value: 'other', label: 'Other', desc: 'General purpose training' }
                  ].map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => setTrainingType(opt.value as any)}
                      className={`p-3 rounded-lg border-2 text-left transition-colors ${
                        trainingType === opt.value
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                      }`}
                    >
                      <div className="font-medium">{opt.label}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block font-medium mb-2">What&apos;s your priority?</label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { value: 'quality', label: 'Quality', desc: 'Best results, longer training' },
                    { value: 'speed', label: 'Speed', desc: 'Faster iterations, good results' },
                    { value: 'memory_efficiency', label: 'Memory Efficient', desc: 'Maximize on limited hardware' }
                  ].map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => setPriority(opt.value as any)}
                      className={`p-3 rounded-lg border-2 text-left transition-colors ${
                        priority === opt.value
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                      }`}
                    >
                      <div className="font-medium">{opt.label}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block font-medium mb-2">Your experience level?</label>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    { value: 'beginner', label: 'Beginner', desc: 'Show me detailed explanations' },
                    { value: 'intermediate', label: 'Intermediate', desc: 'Some guidance is helpful' },
                    { value: 'advanced', label: 'Advanced', desc: 'Just the essentials' }
                  ].map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => setExperienceLevel(opt.value as any)}
                      className={`p-3 rounded-lg border-2 text-left transition-colors ${
                        experienceLevel === opt.value
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
                      }`}
                    >
                      <div className="font-medium">{opt.label}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <Button
                  className="px-4 py-2 bg-gray-300 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-400"
                  onClick={() => setStep('hardware')}
                >
                  Back
                </Button>
                <Button
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                  onClick={handleComplete}
                >
                  Start Wizard
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
