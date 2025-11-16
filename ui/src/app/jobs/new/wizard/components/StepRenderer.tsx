'use client';

import React from 'react';
import { WizardStepId, sections, fields, setNestedValue } from '../fieldConfig';
import { SectionRenderer } from './SectionRenderer';
import { JobConfig } from '@/types';

interface StepRendererProps {
  stepId: WizardStepId;
  selectedModel: string;
  jobConfig: JobConfig;
  onConfigChange: (newConfig: JobConfig) => void;
}

export function StepRenderer({ stepId, selectedModel, jobConfig, onConfigChange }: StepRendererProps) {
  // Get sections for this step, sorted by order
  const stepSections = sections
    .filter(s => {
      // Check if section belongs to this step
      if (s.step !== stepId) return false;

      // Check if section applies to selected model
      if (s.applicableModels && !s.applicableModels.includes(selectedModel)) {
        return false;
      }

      return true;
    })
    .sort((a, b) => a.order - b.order);

  // Handle field value changes
  const handleFieldChange = (fieldId: string, value: any) => {
    const newConfig = setNestedValue(jobConfig, fieldId, value);
    onConfigChange(newConfig);
  };

  // If no sections are visible for this step, return null
  if (stepSections.length === 0) {
    return null;
  }

  return (
    <div className="space-y-6">
      {stepSections.map(section => (
        <SectionRenderer
          key={section.id}
          section={section}
          fields={fields}
          selectedModel={selectedModel}
          jobConfig={jobConfig}
          onChange={handleFieldChange}
        />
      ))}
    </div>
  );
}
