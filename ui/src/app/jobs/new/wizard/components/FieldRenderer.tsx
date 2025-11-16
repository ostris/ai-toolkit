'use client';

import React from 'react';
import { FieldConfig } from '../fieldConfig';
import { TextInput, NumberInput, SelectInput, Checkbox, SliderInput } from '@/components/formInputs';

interface FieldRendererProps {
  field: FieldConfig;
  value: any;
  onChange: (id: string, value: any) => void;
}

export function FieldRenderer({ field, value, onChange }: FieldRendererProps) {
  const handleChange = (newValue: any) => onChange(field.id, newValue);

  // Create a doc object from the description if provided
  const doc = field.description ? { title: field.label, description: field.description } : undefined;

  switch (field.type) {
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

    default:
      return null;
  }
}
