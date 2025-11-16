# Wizard Field Visibility by Model - Design Document

## Problem

The wizard UI shows all configuration options regardless of the selected model. Many options are model-specific (e.g., Flux CFG mode only applies to Flux models, SDXL text encoder selection only applies to SDXL). This confuses users and clutters the interface.

## Solution

Refactor the wizard from hardcoded JSX to data-driven field rendering with per-field model applicability metadata.

## Architecture

### Field Configuration Schema

```typescript
// ui/src/app/jobs/new/wizard/fieldConfig.ts

type WizardStep =
  | 'model' | 'quantization' | 'target' | 'dataset' | 'resolution'
  | 'memory' | 'optimizer' | 'advanced' | 'regularization' | 'training'
  | 'sampling' | 'save' | 'logging' | 'monitoring' | 'review';

type FieldType = 'boolean' | 'number' | 'string' | 'select' | 'multiselect' | 'text';

interface FieldConfig {
  id: string;                              // config path: 'config.process[0].model.use_flux_cfg'
  label: string;                           // display label
  description?: string;                    // help text/tooltip
  type: FieldType;
  defaultValue?: any;
  step: WizardStep;
  section?: string;                        // groups fields under a header
  applicableModels?: string[];             // whitelist; undefined = all models

  // Type-specific
  options?: { value: any; label: string }[];  // select/multiselect
  min?: number;                               // number
  max?: number;
  numberStep?: number;                        // increment for number inputs
  placeholder?: string;                       // string/text

  // Conditional visibility
  showWhen?: {
    field: string;
    value: any;
  };
}

interface SectionConfig {
  id: string;
  title: string;
  description?: string;
  step: WizardStep;
  order: number;  // display order within step
}
```

### Section Renderer

```typescript
// ui/src/app/jobs/new/wizard/components/SectionRenderer.tsx

interface SectionRendererProps {
  section: SectionConfig;
  fields: FieldConfig[];
  selectedModel: string;
  jobConfig: JobConfig;
  onChange: (id: string, value: any) => void;
}

function SectionRenderer({ section, fields, selectedModel, jobConfig, onChange }: SectionRendererProps) {
  // Filter to fields in this section that apply to the selected model
  const visibleFields = fields.filter(field => {
    if (field.section !== section.id) return false;
    if (field.applicableModels && !field.applicableModels.includes(selectedModel)) return false;

    // Check showWhen condition
    if (field.showWhen) {
      const conditionValue = getNestedValue(jobConfig, field.showWhen.field);
      if (conditionValue !== field.showWhen.value) return false;
    }

    return true;
  });

  // Hide section entirely if no visible fields
  if (visibleFields.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium">{section.title}</h3>
      {section.description && (
        <p className="text-sm text-gray-600">{section.description}</p>
      )}
      <div className="space-y-3">
        {visibleFields.map(field => (
          <FieldRenderer
            key={field.id}
            field={field}
            value={getNestedValue(jobConfig, field.id)}
            onChange={onChange}
          />
        ))}
      </div>
    </div>
  );
}
```

### Field Renderer

```typescript
// ui/src/app/jobs/new/wizard/components/FieldRenderer.tsx

interface FieldRendererProps {
  field: FieldConfig;
  value: any;
  onChange: (id: string, value: any) => void;
}

function FieldRenderer({ field, value, onChange }: FieldRendererProps) {
  const handleChange = (newValue: any) => onChange(field.id, newValue);

  switch (field.type) {
    case 'boolean':
      return (
        <FormGroup label={field.label} tooltip={field.description}>
          <Checkbox
            checked={value ?? field.defaultValue ?? false}
            onChange={handleChange}
          />
        </FormGroup>
      );

    case 'number':
      return (
        <FormGroup label={field.label} tooltip={field.description}>
          <NumberInput
            value={value ?? field.defaultValue}
            onChange={handleChange}
            min={field.min}
            max={field.max}
            step={field.numberStep}
          />
        </FormGroup>
      );

    case 'select':
      return (
        <FormGroup label={field.label} tooltip={field.description}>
          <Select
            value={value ?? field.defaultValue}
            onChange={handleChange}
            options={field.options ?? []}
          />
        </FormGroup>
      );

    case 'string':
    case 'text':
      return (
        <FormGroup label={field.label} tooltip={field.description}>
          <Input
            value={value ?? field.defaultValue ?? ''}
            onChange={handleChange}
            placeholder={field.placeholder}
          />
        </FormGroup>
      );

    default:
      return null;
  }
}
```

### Step Renderer Integration

```typescript
// In ComprehensiveWizard.tsx

function renderStep(stepId: WizardStep) {
  const stepSections = sections
    .filter(s => s.step === stepId)
    .sort((a, b) => a.order - b.order);

  return (
    <div className="space-y-6">
      {stepSections.map(section => (
        <SectionRenderer
          key={section.id}
          section={section}
          fields={allFields}
          selectedModel={jobConfig.config.process[0].model?.arch}
          jobConfig={jobConfig}
          onChange={handleFieldChange}
        />
      ))}
    </div>
  );
}
```

## Example Field Definitions

```typescript
// ui/src/app/jobs/new/wizard/fieldConfig.ts

export const sections: SectionConfig[] = [
  {
    id: 'model_quantization',
    title: 'Model Quantization',
    description: 'Reduce VRAM usage by quantizing model weights',
    step: 'quantization',
    order: 1,
  },
  {
    id: 'flux_options',
    title: 'Flux-Specific Options',
    step: 'quantization',
    order: 2,
  },
  {
    id: 'sdxl_options',
    title: 'SDXL-Specific Options',
    step: 'quantization',
    order: 3,
  },
  // ... more sections
];

export const fields: FieldConfig[] = [
  // Universal fields
  {
    id: 'config.process[0].model.quantize',
    label: 'Enable Quantization',
    description: 'Quantize transformer weights to reduce VRAM usage',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'model_quantization',
    // No applicableModels = applies to all (except those with disableSections)
  },
  {
    id: 'config.process[0].model.qtype',
    label: 'Quantization Type',
    description: 'Precision level for quantized weights',
    type: 'select',
    defaultValue: 'qfloat8',
    step: 'quantization',
    section: 'model_quantization',
    options: [
      { value: 'qfloat8', label: 'float8 (default)' },
      { value: 'uint4', label: '4 bit' },
      { value: 'uint3', label: '3 bit' },
    ],
    showWhen: {
      field: 'config.process[0].model.quantize',
      value: true,
    },
  },

  // Flux-specific fields
  {
    id: 'config.process[0].model.use_flux_cfg',
    label: 'Flux CFG Mode',
    description: 'Enable classifier-free guidance for distilled Flux models',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'flux_options',
    applicableModels: ['flux', 'flex1', 'flex2', 'chroma'],
  },
  {
    id: 'config.process[0].model.split_model',
    label: 'Split Model Across GPUs',
    description: 'Distribute model layers across multiple GPUs',
    type: 'boolean',
    defaultValue: false,
    step: 'quantization',
    section: 'flux_options',
    applicableModels: ['flux', 'flex1', 'flex2'],
  },

  // SDXL-specific fields
  {
    id: 'config.process[0].model.use_text_encoder_1',
    label: 'Use CLIP-L Text Encoder',
    type: 'boolean',
    defaultValue: true,
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
  },
  {
    id: 'config.process[0].model.use_text_encoder_2',
    label: 'Use OpenCLIP-G Text Encoder',
    type: 'boolean',
    defaultValue: true,
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
  },
  {
    id: 'config.process[0].model.refiner_name_or_path',
    label: 'Refiner Model Path',
    description: 'Optional path to SDXL refiner model',
    type: 'string',
    defaultValue: '',
    step: 'quantization',
    section: 'sdxl_options',
    applicableModels: ['sdxl'],
    placeholder: 'stabilityai/stable-diffusion-xl-refiner-1.0',
  },

  // Video model fields
  {
    id: 'config.process[0].datasets[0].num_frames',
    label: 'Number of Frames',
    description: 'Frames per video sample',
    type: 'number',
    defaultValue: 41,
    step: 'dataset',
    section: 'video_settings',
    applicableModels: [
      'wan21:1b', 'wan21:14b', 'wan21_i2v:14b', 'wan21_i2v:14b480p',
      'wan22_14b:t2v', 'wan22_14b_i2v', 'wan22_5b',
    ],
    min: 1,
    max: 241,
    numberStep: 8,
  },

  // ... 200+ more fields
];
```

## Migration Strategy

### Phase 1: Infrastructure (1-2 hours)
1. Create `fieldConfig.ts` with types and empty arrays
2. Create `FieldRenderer.tsx` component
3. Create `SectionRenderer.tsx` component
4. Add helper functions for nested value access/update

### Phase 2: Quantization Step (2-3 hours)
1. Extract all fields from the Quantization step JSX
2. Define sections and fields in config
3. Replace JSX with data-driven rendering
4. Test all model types

### Phase 3: Remaining Steps (4-6 hours)
Migrate one step at a time:
- Target (LoRA/LoKr settings)
- Dataset
- Resolution
- Memory
- Optimizer
- Advanced
- Regularization
- Training
- Sampling
- Save
- Logging
- Monitoring

### Phase 4: Cleanup (1 hour)
1. Remove old hardcoded JSX
2. Update tests
3. Verify all models work correctly

## Model Applicability Reference

Based on current `options.ts`:

| Model | Quantization | Timestep Type | Conv LoRA | Control Path | Video Frames |
|-------|-------------|---------------|-----------|--------------|--------------|
| flux | Yes | Yes | No | No | No |
| flux_kontext | Yes | Yes | No | Yes | No |
| flex1 | Yes | Yes | No | No | No |
| flex2 | Yes | Yes | No | No | No |
| chroma | Yes | Yes | No | No | No |
| sdxl | No | No | Yes | No | No |
| sd15 | No | No | Yes | No | No |
| wan21:1b | Yes | Yes | No | No | Yes |
| wan22_14b:t2v | Yes | Yes | No | No | Yes |
| qwen_image | Yes | Yes | No | No | No |
| qwen_image_edit | Yes | Yes | No | Yes | No |
| hidream | Yes | Yes | No | No | No |

## Benefits

1. **Cleaner UI** - Users only see relevant options for their model
2. **Reduced confusion** - No more wondering if an option applies
3. **Maintainability** - Add new fields/models by updating config, not JSX
4. **Consistency** - All fields rendered uniformly
5. **Smaller component** - ComprehensiveWizard.tsx drops from 40k+ lines to ~5k

## Risks and Mitigations

**Risk**: Missing field during migration
**Mitigation**: Automated comparison of old vs new rendered fields per model

**Risk**: Breaking existing functionality
**Mitigation**: Migrate one step at a time, test thoroughly

**Risk**: Performance with 200+ field configs
**Mitigation**: Memoize filtered field lists, lazy load step configs

## Success Criteria

- [ ] All model-specific fields only show for applicable models
- [ ] Section headers hidden when all fields in section are hidden
- [ ] No regression in existing functionality
- [ ] Config output identical to pre-refactor
- [ ] TypeScript compilation passes
- [ ] All wizard steps render correctly for each model type
