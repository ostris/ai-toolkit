import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import LoraBrowserModal, { LoraPick } from '@/components/generate/LoraBrowserModal';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  SelectInput,
  SliderInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { CaptionJobConfig, CaptionLora } from '@/types';
import { handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import {
  batchSizeOptions,
  captionFormatOptions,
  captionerTypes,
  defaultQtype,
  groupedCaptionerTypes,
  maxNewTokensOptions,
  maxResOptions,
  quantizationOptions,
} from '@/helpers/captionOptions';

type Props = {
  jobConfig: CaptionJobConfig;
  setJobConfig: (value: any, key?: string) => void;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  showGPUSelect: boolean;
};

const CaptionSimpleJob: React.FC<Props> = ({ jobConfig, setJobConfig, gpuIDs, setGpuIDs, gpuList, showGPUSelect }) => {
  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);
  const additionalSections = selectedCaptionOption?.additionalSections || [];
  const captionPrompts = selectedCaptionOption?.captionPrompts || {};
  const promptPresetNames = Object.keys(captionPrompts);
  const minNewTokens = selectedCaptionOption?.minNewTokens ?? 0;
  const newTokensOptions = maxNewTokensOptions.filter(option => parseInt(option.value) >= minNewTokens);
  const [loraModalOpen, setLoraModalOpen] = useState(false);
  const loras: CaptionLora[] = jobConfig.config.process[0].caption.loras || [];
  const setLoras = (next: CaptionLora[]) => setJobConfig(next, 'config.process[0].caption.loras');
  const addLora = (pick: LoraPick) => {
    if (loras.some(l => l.path === pick.path)) return;
    setLoras([...loras, { path: pick.path, name: pick.name, strength: 1.0 }]);
  };

  return (
    <div className="text-sm text-gray-400">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="Captioner Type"
            value={jobConfig.config.process[0].type}
            onChange={value => {
              handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
            }}
            options={groupedCaptionerTypes}
          />
        </div>
        {showGPUSelect && (
          <div>
            <SelectInput
              label="GPU ID"
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
      </div>
      <div className="mt-4">
        <CreatableSelectInput
          label="Name or Path"
          value={jobConfig.config.process[0].caption.model_name_or_path}
          docKey="config.process[0].caption.model_name_or_path"
          onChange={(value: string | null) => {
            if (value?.trim() === '') {
              value = null;
            }
            setJobConfig(value, 'config.process[0].caption.model_name_or_path');
          }}
          placeholder=""
          options={selectedCaptionOption?.name_or_path_options || []}
          required
        />
      </div>
      {additionalSections.includes('caption.model_name_or_path2') && (
        <div className="mt-4">
          <CreatableSelectInput
            label="Name or Path 2"
            value={jobConfig.config.process[0].caption.model_name_or_path2 || ''}
            onChange={(value: string | null) => {
              if (value?.trim() === '') {
                value = null;
              }
              setJobConfig(value, 'config.process[0].caption.model_name_or_path2');
            }}
            placeholder=""
            options={selectedCaptionOption?.name_or_path2_options || []}
          />
        </div>
      )}
      {additionalSections.includes('caption.caption_format') && (
        <div className="mt-4">
          <SelectInput
            label="Caption Format"
            value={jobConfig.config.process[0].caption.caption_format || 'ace_step'}
            onChange={value => setJobConfig(value, 'config.process[0].caption.caption_format')}
            options={captionFormatOptions}
          />
        </div>
      )}
      {additionalSections.includes('caption.fixed_caption') && (
        <div className="mt-4">
          <TextInput
            label="Fixed Caption"
            value={jobConfig.config.process[0].caption.fixed_caption || ''}
            onChange={value => {
              if (value?.trim() === '') {
                //@ts-ignore
                value = undefined;
              }
              setJobConfig(value, 'config.process[0].caption.fixed_caption');
            }}
            placeholder="Enter fixed caption (if you want the same caption for all audio files)"
          />
        </div>
      )}
      {selectedCaptionOption?.supportsLoras && (
        <div className="mt-4">
          <div className="text-xs text-gray-300 mb-1">LoRAs</div>
          <div className="space-y-2">
            {loras.map((l, i) => (
              <div key={l.path} className="bg-gray-950/60 border border-gray-800 rounded-md px-2 py-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs truncate flex-1 text-gray-200" title={l.path}>
                    {l.name}
                  </span>
                  <input
                    type="number"
                    step={0.05}
                    min={-2}
                    max={3}
                    value={l.strength}
                    onChange={e => {
                      const v = parseFloat(e.target.value);
                      const next = [...loras];
                      next[i] = { ...l, strength: isNaN(v) ? 0 : v };
                      setLoras(next);
                    }}
                    className="w-16 text-xs px-1.5 py-0.5 bg-gray-950 border border-gray-700 rounded text-gray-100 text-right"
                  />
                  <button
                    type="button"
                    onClick={() => setLoras(loras.filter(x => x.path !== l.path))}
                    className="text-gray-500 hover:text-red-400"
                    title="Remove"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() => setLoraModalOpen(true)}
              className="w-full px-2 py-1 rounded-md bg-gray-800 hover:bg-gray-700 text-gray-200 text-xs flex items-center justify-center gap-1"
            >
              <Plus className="w-3.5 h-3.5" /> Add LoRA
            </button>
          </div>
          <div className="text-[11px] text-gray-500 mt-1">
            Applied as sidechains at caption time; the model weights are never merged.
          </div>
          <LoraBrowserModal
            isOpen={loraModalOpen}
            onClose={() => setLoraModalOpen(false)}
            onPick={addLora}
            cloudLoras={selectedCaptionOption?.cloudLoras}
          />
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="Quantize"
            value={jobConfig.config.process[0].caption.quantize ? jobConfig.config.process[0].caption.qtype : ''}
            onChange={value => {
              if (value === '') {
                setJobConfig(false, 'config.process[0].caption.quantize');
                value = defaultQtype;
              } else {
                setJobConfig(true, 'config.process[0].caption.quantize');
              }
              setJobConfig(value, 'config.process[0].caption.qtype');
            }}
            options={quantizationOptions}
          />
          <div className="mt-4">
            <CreatableSelectInput
              label="Caption Extension"
              value={jobConfig.config.process[0].caption.caption_extension || 'txt'}
              onChange={value => {
                setJobConfig(value, 'config.process[0].caption.caption_extension');
              }}
              options={[
                { value: 'txt', label: 'txt' },
                { value: 'json', label: 'json' },
                { value: 'caption', label: 'caption' },
              ]}
            />
          </div>
          {additionalSections.includes('caption.max_res') && (
            <div className="mt-4">
              <SelectInput
                label="Max Resolution"
                value={`${jobConfig.config.process[0].caption.max_res || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_res');
                  }
                }}
                options={maxResOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.max_new_tokens') && (
            <div className="mt-4">
              <SelectInput
                label="Max New Tokens"
                value={`${jobConfig.config.process[0].caption.max_new_tokens || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_new_tokens');
                  }
                }}
                options={newTokensOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.batch_size') && (
            <div className="mt-4">
              <SelectInput
                label="Batch Size"
                value={`${jobConfig.config.process[0].caption.batch_size || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.batch_size');
                  }
                }}
                options={batchSizeOptions}
              />
            </div>
          )}
        </div>
        <div>
          <FormGroup label="Options">
            <Checkbox
              label="Low VRAM"
              checked={jobConfig.config.process[0].caption.low_vram}
              onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
            />
            <Checkbox
              label="Recaption"
              checked={jobConfig.config.process[0].caption.recaption}
              onChange={value => setJobConfig(value, 'config.process[0].caption.recaption')}
            />
            <Checkbox
              label="Compile Models"
              checked={jobConfig.config.process[0].caption.compile || false}
              onChange={value => setJobConfig(value, 'config.process[0].caption.compile')}
            />
            {additionalSections.includes('caption.extract_vocals_before_transcribe') && (
              <Checkbox
                label="Extract Vocals Before Transcribing"
                checked={jobConfig.config.process[0].caption.extract_vocals_before_transcribe || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.extract_vocals_before_transcribe')}
              />
            )}
            {additionalSections.includes('caption.keep_timestamps') && (
              <Checkbox
                label="Keep Lyric Timestamps"
                checked={jobConfig.config.process[0].caption.keep_timestamps || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.keep_timestamps')}
              />
            )}
            {additionalSections.includes('caption.thinking') && (
              <Checkbox
                label="Thinking"
                checked={jobConfig.config.process[0].caption.thinking || false}
                onChange={value => setJobConfig(value, 'config.process[0].caption.thinking')}
              />
            )}
            {additionalSections.includes('caption.layer_offloading') && (
              <>
                <Checkbox
                  label="Layer Offloading"
                  checked={jobConfig.config.process[0].caption.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].caption.layer_offloading')}
                />
                {jobConfig.config.process[0].caption.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Offload %"
                      value={Math.round((jobConfig.config.process[0].caption.layer_offloading_percent ?? 1) * 100)}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].caption.layer_offloading_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </FormGroup>
        </div>
      </div>
      {additionalSections.includes('caption.caption_prompt') && (
        <div className="mt-4">
          {promptPresetNames.length > 1 && (
            <div className="mb-4">
              <SelectInput
                label="Prompt Preset"
                value={
                  promptPresetNames.find(
                    name => captionPrompts[name] === jobConfig.config.process[0].caption.caption_prompt,
                  ) || ''
                }
                onChange={value => {
                  if (captionPrompts[value] !== undefined) {
                    setJobConfig(captionPrompts[value], 'config.process[0].caption.caption_prompt');
                  }
                }}
                options={[
                  { value: '', label: '- Custom -' },
                  ...promptPresetNames.map(name => ({ value: name, label: name })),
                ]}
              />
            </div>
          )}
          <TextAreaInput
            label="Caption Prompt"
            value={jobConfig.config.process[0].caption.caption_prompt || ''}
            onChange={value => {
              setJobConfig(value, 'config.process[0].caption.caption_prompt');
            }}
            placeholder="Enter caption prompt"
          />
        </div>
      )}
    </div>
  );
};

export default CaptionSimpleJob;
