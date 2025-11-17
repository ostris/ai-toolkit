'use client';

import React, { memo, useMemo, useCallback } from 'react';
import { WizardStepId, sections, fields, setNestedValue } from '../fieldConfig';
import { SectionRenderer } from './SectionRenderer';
import { JobConfig } from '@/types';

interface StepRendererProps {
  stepId: WizardStepId;
  selectedModel: string;
  jobConfig: JobConfig;
  onConfigChange: (newConfig: JobConfig) => void;
}

function StepRendererComponent({ stepId, selectedModel, jobConfig, onConfigChange }: StepRendererProps) {
  // Memoize sections for this step to avoid recalculating on every render
  const stepSections = useMemo(
    () =>
      sections
        .filter(s => {
          // Check if section belongs to this step
          if (s.step !== stepId) return false;

          // Check if section applies to selected model
          if (s.applicableModels && !s.applicableModels.includes(selectedModel)) {
            return false;
          }

          return true;
        })
        .sort((a, b) => a.order - b.order),
    [stepId, selectedModel]
  );

  // Handle field value changes with stable reference
  const handleFieldChange = useCallback(
    (fieldId: string, value: any) => {
      const newConfig = setNestedValue(jobConfig, fieldId, value);
      onConfigChange(newConfig);
    },
    [jobConfig, onConfigChange]
  );

  // Handle compound field changes (multiple paths at once) with stable reference
  const handleCompoundChange = useCallback(
    (changes: { path: string; value: any }[]) => {
      let newConfig = jobConfig;
      for (const { path, value } of changes) {
        newConfig = setNestedValue(newConfig, path, value);
      }
      onConfigChange(newConfig);
    },
    [jobConfig, onConfigChange]
  );

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
          onCompoundChange={handleCompoundChange}
        />
      ))}
    </div>
  );
}

// Memoize to prevent re-renders when step configuration hasn't changed
export const StepRenderer = memo(StepRendererComponent);

StepRenderer.displayName = 'StepRenderer';
