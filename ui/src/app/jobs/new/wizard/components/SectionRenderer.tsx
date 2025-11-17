'use client';

import React from 'react';
import { SectionConfig, FieldConfig, getNestedValue } from '../fieldConfig';
import { FieldRenderer } from './FieldRenderer';
import { JobConfig } from '@/types';

interface SectionRendererProps {
  section: SectionConfig;
  fields: FieldConfig[];
  selectedModel: string;
  jobConfig: JobConfig;
  onChange: (id: string, value: any) => void;
  onCompoundChange?: (changes: { path: string; value: any }[]) => void;
}

export function SectionRenderer({ section, fields, selectedModel, jobConfig, onChange, onCompoundChange }: SectionRendererProps) {
  // Filter to fields in this section that apply to the selected model
  const visibleFields = fields.filter(field => {
    // Check section match
    if (field.section !== section.id) return false;

    // Check model applicability
    if (field.applicableModels && !field.applicableModels.includes(selectedModel)) {
      return false;
    }

    // Check showWhen condition
    if (field.showWhen) {
      const conditionValue = getNestedValue(jobConfig, field.showWhen.field);
      if (conditionValue !== field.showWhen.value) {
        return false;
      }
    }

    return true;
  });

  // Hide section entirely if no visible fields
  if (visibleFields.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-medium text-gray-200">{section.title}</h3>
        {section.description && <p className="text-sm text-gray-400 mt-1">{section.description}</p>}
      </div>
      <div className="space-y-3">
        {visibleFields.map(field => (
          <FieldRenderer
            key={field.id}
            field={field}
            value={getNestedValue(jobConfig, field.id)}
            onChange={onChange}
            jobConfig={jobConfig}
            onCompoundChange={onCompoundChange}
          />
        ))}
      </div>
    </div>
  );
}
