import Card from '@/components/Card';
import { Checkbox, NumberInput, SelectInput, TextInput } from '@/components/formInputs';

interface MultiGPUConfigProps {
  useMultiGPU: boolean;
  setUseMultiGPU: (value: boolean) => void;
  numGPUs: number;
  setNumGPUs: (value: number) => void;
  accelerateConfig: any;
  setAccelerateConfig: (value: any) => void;
  gpuList: any[];
}

export default function MultiGPUConfig({
  useMultiGPU,
  setUseMultiGPU,
  numGPUs,
  setNumGPUs,
  accelerateConfig,
  setAccelerateConfig,
  gpuList,
}: MultiGPUConfigProps) {
  const availableGPUs = gpuList.length;

  const handleMultiGPUToggle = (enabled: boolean) => {
    setUseMultiGPU(enabled);
    if (enabled) {
      // Set default accelerate config for multi-GPU
      setAccelerateConfig({
        compute_environment: 'LOCAL_MACHINE',
        distributed_type: 'MULTI_GPU',
        downcast_bf16: 'no',
        gpu_ids: gpuList.slice(0, Math.min(numGPUs, availableGPUs)).map(gpu => gpu.index).join(','),
        machine_rank: 0,
        main_training_function: 'main',
        mixed_precision: 'fp16',
        num_machines: 1,
        num_processes: Math.min(numGPUs, availableGPUs),
        rdzv_backend: 'static',
        same_network: true,
        tpu_env: [],
        tpu_use_cluster: false,
        tpu_use_sudo: false,
        use_cpu: false,
      });
    }
  };

  const handleNumGPUsChange = (value: number) => {
    const newNumGPUs = Math.min(value, availableGPUs);
    setNumGPUs(newNumGPUs);
    
    if (useMultiGPU && accelerateConfig) {
      setAccelerateConfig({
        ...accelerateConfig,
        gpu_ids: gpuList.slice(0, newNumGPUs).map(gpu => gpu.index).join(','),
        num_processes: newNumGPUs,
      });
    }
  };

  return (
    <Card title="Multi-GPU Configuration">
      <div className="space-y-4">
        <Checkbox
          label="Enable Multi-GPU Training"
          checked={useMultiGPU}
          onChange={handleMultiGPUToggle}
        />

        {useMultiGPU && (
          <>
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg p-3">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                Multi-GPU training will use Hugging Face Accelerate for distributed training.
                This requires proper Accelerate configuration.
              </p>
            </div>

            <NumberInput
              label="Number of GPUs"
              value={numGPUs}
              onChange={handleNumGPUsChange}
              min={2}
              max={availableGPUs}
            />

            <SelectInput
              label="Mixed Precision"
              value={accelerateConfig?.mixed_precision || 'fp16'}
              onChange={(value: string) => setAccelerateConfig({
                ...accelerateConfig,
                mixed_precision: value
              })}
              options={[
                { value: 'no', label: 'No Mixed Precision' },
                { value: 'fp16', label: 'FP16' },
                { value: 'bf16', label: 'BF16' },
              ]}
            />

            <SelectInput
              label="Distributed Type"
              value={accelerateConfig?.distributed_type || 'MULTI_GPU'}
              onChange={(value: string) => setAccelerateConfig({
                ...accelerateConfig,
                distributed_type: value
              })}
              options={[
                { value: 'NO', label: 'No Distributed Training' },
                { value: 'MULTI_GPU', label: 'Multi-GPU' },
                { value: 'MULTI_CPU', label: 'Multi-CPU' },
              ]}
            />

            <TextInput
              label="GPU IDs"
              value={accelerateConfig?.gpu_ids || ''}
              onChange={(value: string) => setAccelerateConfig({
                ...accelerateConfig,
                gpu_ids: value
              })}
              placeholder="0,1,2,3"
            />
          </>
        )}
      </div>
    </Card>
  );
} 