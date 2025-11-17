'use client';

import React, { memo, useCallback } from 'react';
import { FieldConfig, getNestedValue } from '../fieldConfig';
import { TextInput, NumberInput, SelectInput, Checkbox, SliderInput } from '@/components/formInputs';
import { X } from 'lucide-react';

interface FieldRendererProps {
  field: FieldConfig;
  value: any;
  onChange: (id: string, value: any) => void;
  jobConfig?: any; // Full config for compound fields
  onCompoundChange?: (changes: { path: string; value: any }[]) => void; // For compound fields
}

function FieldRendererComponent({ field, value, onChange, jobConfig, onCompoundChange }: FieldRendererProps) {
  const handleChange = (newValue: any) => onChange(field.id, newValue);

  // Create a doc object from the description if provided
  const doc = field.description ? { title: field.label, description: field.description } : undefined;

  switch (field.type) {
    case 'compound':
      if (!field.compoundOptions || !jobConfig || !onCompoundChange) {
        return null;
      }
      // Derive current UI value from jobConfig using first matching option
      let currentValue = field.defaultValue;
      for (const option of field.compoundOptions) {
        const allMatch = option.sets.every(
          ({ path, value: expectedValue }) => getNestedValue(jobConfig, path) === expectedValue
        );
        if (allMatch) {
          currentValue = option.value;
          break;
        }
      }
      return (
        <SelectInput
          label={field.label}
          value={currentValue}
          onChange={(newValue: any) => {
            const selectedOption = field.compoundOptions!.find(opt => opt.value === newValue);
            if (selectedOption) {
              onCompoundChange(selectedOption.sets);
            }
          }}
          options={field.compoundOptions.map(opt => ({ value: opt.value, label: opt.label }))}
          doc={doc}
        />
      );

    case 'boolean':
      // Special handling for array-based booleans (like ignore_if_contains for HiDream)
      if (field.id.includes('ignore_if_contains')) {
        const isEnabled = Array.isArray(value) && value.length > 0;
        return (
          <Checkbox
            label={field.label}
            checked={isEnabled}
            onChange={checked => {
              handleChange(checked ? ['ff_i.experts', 'ff_i.gate'] : []);
            }}
            doc={doc}
          />
        );
      }
      return <Checkbox label={field.label} checked={value ?? field.defaultValue ?? false} onChange={handleChange} doc={doc} />;

    case 'number':
      return (
        <NumberInput
          label={field.label}
          value={value ?? field.defaultValue ?? null}
          onChange={handleChange}
          min={field.min}
          max={field.max}
          step={field.numberStep}
          doc={doc}
        />
      );

    case 'select':
      return (
        <SelectInput
          label={field.label}
          value={value ?? field.defaultValue ?? ''}
          onChange={handleChange}
          options={field.options ?? []}
          doc={doc}
        />
      );

    case 'string':
      return (
        <TextInput
          label={field.label}
          value={value ?? field.defaultValue ?? ''}
          onChange={handleChange}
          placeholder={field.placeholder}
          doc={doc}
        />
      );

    case 'slider':
      // Special handling for percentage sliders (0-1 range displayed as 0-100%)
      const isPercentage = field.max === 1 && field.min === 0;
      if (isPercentage) {
        return (
          <div>
            <SliderInput
              label={field.label}
              value={((value ?? field.defaultValue ?? 1) * 100)}
              onChange={(newVal: number) => handleChange(newVal / 100)}
              min={0}
              max={100}
              step={(field.numberStep ?? 0.1) * 100}
              doc={doc}
            />
            <div className="text-xs text-gray-500 mt-1">
              {Math.round((value ?? field.defaultValue ?? 1) * 100)}% of layers can be offloaded
            </div>
          </div>
        );
      }
      return (
        <SliderInput
          label={field.label}
          value={value ?? field.defaultValue ?? field.min ?? 0}
          onChange={handleChange}
          min={field.min ?? 0}
          max={field.max ?? 100}
          step={field.numberStep}
          doc={doc}
        />
      );

    case 'sample_prompts_array':
      const samples = Array.isArray(value) ? value : field.defaultValue ?? [{ prompt: '' }];

      // Helper to update a specific sample's field
      const updateSampleField = (index: number, fieldName: string, newValue: any) => {
        const newSamples = [...samples];
        if (newValue === '' || newValue === null || newValue === undefined) {
          // Remove the field if empty
          const { [fieldName]: _, ...rest } = newSamples[index];
          newSamples[index] = rest;
        } else {
          newSamples[index] = { ...newSamples[index], [fieldName]: newValue };
        }
        handleChange(newSamples);
      };

      // Helper to remove a sample
      const removeSample = (index: number) => {
        const newSamples = samples.filter((_: any, i: number) => i !== index);
        handleChange(newSamples);
      };

      // Helper to add a new sample
      const addSample = () => {
        handleChange([...samples, { prompt: '' }]);
      };

      return (
        <div className="space-y-4">
          {doc && (
            <div className="text-sm text-gray-400 mb-2">{doc.description}</div>
          )}
          {samples.map((sample: any, i: number) => (
            <div key={i} className="rounded-lg p-4 bg-gray-950 border border-gray-800">
              <div className="flex items-start gap-2">
                <div className="flex-1 space-y-3">
                  <TextInput
                    label={`Prompt ${i + 1}`}
                    value={sample.prompt ?? ''}
                    onChange={(newValue: string) => updateSampleField(i, 'prompt', newValue)}
                    placeholder="Enter prompt for sample generation"
                  />
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <NumberInput
                      label="Width"
                      value={sample.width ?? null}
                      onChange={(newValue: number | null) => updateSampleField(i, 'width', newValue)}
                      placeholder="Default"
                      min={64}
                      max={4096}
                      step={8}
                    />
                    <NumberInput
                      label="Height"
                      value={sample.height ?? null}
                      onChange={(newValue: number | null) => updateSampleField(i, 'height', newValue)}
                      placeholder="Default"
                      min={64}
                      max={4096}
                      step={8}
                    />
                    <NumberInput
                      label="Seed"
                      value={sample.seed ?? null}
                      onChange={(newValue: number | null) => updateSampleField(i, 'seed', newValue)}
                      placeholder="Default"
                      min={0}
                    />
                    <NumberInput
                      label="LoRA Scale"
                      value={sample.network_multiplier ?? null}
                      onChange={(newValue: number | null) => updateSampleField(i, 'network_multiplier', newValue)}
                      placeholder="1.0"
                      min={0}
                      max={2}
                      step={0.1}
                    />
                  </div>
                </div>
                {samples.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeSample(i)}
                    className="mt-6 p-1 text-gray-400 hover:text-red-400 transition-colors"
                    title="Remove prompt"
                  >
                    <X className="w-5 h-5" />
                  </button>
                )}
              </div>
            </div>
          ))}
          <button
            type="button"
            onClick={addSample}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-sm"
          >
            Add Prompt
          </button>
        </div>
      );

    default:
      return null;
  }
}

// Memoize to prevent re-renders when props haven't changed
// Custom comparison for better performance with complex objects
export const FieldRenderer = memo(
  FieldRendererComponent,
  (prevProps: FieldRendererProps, nextProps: FieldRendererProps) => {
    // Re-render if field definition changed
    if (prevProps.field.id !== nextProps.field.id) return false;
    // Re-render if value changed
    if (prevProps.value !== nextProps.value) return false;
    // Re-render if onChange reference changed (should be stable with useCallback)
    if (prevProps.onChange !== nextProps.onChange) return false;
    // For compound fields, check if relevant config paths changed
    if (prevProps.field.type === 'compound' && prevProps.jobConfig !== nextProps.jobConfig) {
      return false;
    }
    // For array fields like sample_prompts_array, do deep comparison
    if (prevProps.field.type === 'sample_prompts_array') {
      // Arrays are compared by reference above, which is sufficient for our immutable updates
      return true;
    }
    return true;
  }
);

FieldRenderer.displayName = 'FieldRenderer';
