# Comprehensive Guided Config Wizard Design

**Date:** 2025-01-15
**Status:** Implementation In Progress
**Author:** Claude (with user collaboration)

## Implementation Status

**Core Components Implemented:**
- [x] System Profile API (`/api/system/profile`)
- [x] Python System Profiler (`toolkit/system_profiler.py`)
- [x] Pre-flight Modal (hardware detection + user intent questionnaire)
- [x] Smart Defaults Engine (batch size, caching, prefetching, workers)
- [x] Advisor Panel (educational content, recommendations, performance predictions)
- [x] Summary Header (persistent config overview)
- [x] Comprehensive Wizard with 12 steps
- [x] Integration with existing GuidedWizard (toggle for advanced mode)

**Files Created:**
- `ui/src/app/api/system/profile/route.ts` - API endpoint for system detection
- `toolkit/system_profiler.py` - Python script for hardware detection
- `ui/src/app/jobs/new/wizard/utils/types.ts` - TypeScript type definitions
- `ui/src/app/jobs/new/wizard/utils/smartDefaults.ts` - Smart defaults calculation engine
- `ui/src/app/jobs/new/wizard/components/PreflightModal.tsx` - Pre-flight configuration modal
- `ui/src/app/jobs/new/wizard/components/AdvisorPanel.tsx` - Side panel with real-time guidance
- `ui/src/app/jobs/new/wizard/components/SummaryHeader.tsx` - Persistent config summary
- `ui/src/app/jobs/new/wizard/ComprehensiveWizard.tsx` - Main wizard component

**Remaining Work:**
- Model-specific settings step (Step 12)
- Advanced regularization options (DOP, BPP, EMA)
- Validation endpoint (`/api/config/validate`)
- Recommendation endpoint (`/api/config/recommend`)
- Model info endpoint (`/api/models/{arch}/info`)
- Comparison view in review step
- Auto-fix functionality for validation errors

## Executive Summary

Redesign the web UI's guided config mode to provide comprehensive coverage of all training settings, intelligently grouped by dependency chains, with context-aware recommendations based on system hardware, model requirements, and dataset characteristics.

## Goals

1. **Comprehensive Control** - Expose all ~100+ settings available in SimpleJob
2. **Educational Presentation** - Explain what each setting does and why it matters
3. **Smart Defaults** - Pre-fill optimal values based on system/model/dataset analysis
4. **Cause-Effect Clarity** - Group interdependent settings together with clear relationships
5. **Unified Memory Support** - Handle Apple Silicon, AMD ROCm, and traditional NVIDIA setups
6. **Leverage Existing Optimizations** - Integrate completed TODO improvements (batch scaling, prefetching, caching)

## Architecture Overview

### Core Components

1. **Pre-flight Check Modal** - System detection + user intent questionnaire
2. **Linear Wizard Steps** - 10-12 granular steps with adaptive additions
3. **Progress-Aware Summary Header** - Persistent overview of current configuration
4. **Side Panel Advisor** - Real-time explanations, warnings, and performance predictions
5. **Review & Submit** - Editable summary with validation and comparison views

## Detailed Design

### 1. Pre-flight Check Modal

Displayed before wizard starts. Gathers context for smart defaults.

#### System Detection (Auto-detect with User Override)

```typescript
interface SystemProfile {
  gpu: {
    type: 'nvidia' | 'amd' | 'apple_silicon' | 'cpu_only';
    name: string;
    vramGB: number;
    driverVersion?: string;
    isUnifiedMemory: boolean;
  };
  memory: {
    totalRAM: number;
    availableRAM: number;
    unifiedMemory?: number; // Apple Silicon
  };
  storage: {
    type: 'hdd' | 'ssd' | 'nvme';
    availableSpaceGB: number;
  };
  cpu: {
    cores: number;
    name: string;
  };
}
```

**UI Presentation:**
- Show detected values with checkmarks
- Allow user to edit any value (e.g., "Reserve 4GB VRAM for other apps")
- "Confirm Hardware Profile" button to proceed

#### User Intent Questionnaire

```typescript
interface UserIntent {
  trainingType: 'person' | 'style' | 'object' | 'concept' | 'other';
  priority: 'quality' | 'speed' | 'memory_efficiency';
  experienceLevel: 'beginner' | 'intermediate' | 'advanced';
}
```

**Questions:**
1. "What are you training?" - Person/Style/Object/Concept/Other
2. "What's your priority?" - Quality (longer training, more regularization) / Speed (faster iterations) / Memory Efficiency (maximize on limited hardware)
3. "Experience level?" - Affects advisor verbosity (beginner = more explanations)

### 2. Wizard Steps (Adaptive 10-12)

Each step contains related settings grouped by dependency chains.

#### Step 1: Model Selection
**Settings:** Model architecture, model path
**Dynamic:** Shows image models vs video models based on grouping
**Advisor:** Model capabilities, VRAM requirements, native resolution info

#### Step 2: Quantization
**Settings:** Transformer quantization, text encoder quantization, ARAs
**Dynamic:** Only appears if model supports quantization; shows ARAs when available
**Advisor:** VRAM savings per quantization level, quality trade-offs

#### Step 3: Target Configuration
**Settings:** Target type (LoRA/LoKr), rank, alpha, LoKr factor
**Dynamic:** LoKr factor only when LoKr selected; conv rank only for supported models
**Advisor:** Rank vs quality vs file size trade-offs

#### Step 4: Dataset Setup
**Settings:** Dataset path, trigger word, caption extension, default caption
**Feature:** "Analyze Dataset" button (reuse existing functionality)
**Advisor:** Caption recommendations, trigger word best practices

#### Step 5: Resolution & Augmentation
**Settings:** Training resolution, aspect ratio bucketing, flip X/Y, cache latents
**Dynamic:** Aspect ratio options based on model support
**Advisor:** Resolution vs VRAM, model native resolution, dataset image sizes

#### Step 6: Memory & Batch Configuration
**Settings:**
- Batch size (initial)
- Auto-scale batch size (TODO #9)
- Min/max batch size
- Gradient accumulation
- Caching strategy (memory vs disk - TODO #2, #3)
- GPU prefetch batches (TODO #5)
- Num workers
- Persistent workers (TODO #4)

**Advisor:** VRAM calculations, memory vs speed trade-offs, unified memory handling

#### Step 7: Optimizer & Learning Rate
**Settings:** Optimizer (AdamW8bit, Adafactor), learning rate, weight decay
**Advisor:** LR recommendations by training type, optimizer pros/cons

#### Step 8: Regularization
**Settings:**
- Differential Output Preservation (DOP) + multiplier + class
- Blank Prompt Preservation (BPP) + multiplier
- EMA (use_ema, decay)
- Caption dropout rate

**Advisor:** When to use each regularization, overfitting prevention

#### Step 9: Training Core
**Settings:** Steps, timestep type, timestep bias (content_or_style), loss type
**Advisor:** Steps vs dataset size, timestep strategies for different goals

#### Step 10: Sampling Configuration
**Settings:**
- Sample every
- Sampler type
- Default dimensions (width, height)
- Guidance scale
- Sample steps
- Seed, walk seed
- Skip/force first sample
- Sample prompts list

**Advisor:** Sampling frequency recommendations, prompt writing tips

#### Step 11: Save Settings
**Settings:** Save every, max step saves to keep, save dtype, save format
**Advisor:** Disk space calculations, checkpoint strategy

#### Step 12 (Adaptive): Model-Specific Settings
**Appears for:** Models with additional sections (layer offloading, multistage, etc.)
**Settings:** Layer offloading percentages, multistage training config, etc.
**Advisor:** Model-specific optimizations and trade-offs

#### Step 13: Review & Submit
**Features:**
- Editable inline fields for quick adjustments
- Validation report (errors, warnings)
- Comparison view (your config vs recommended defaults)
- Performance predictions (VRAM, training time, disk space)
- YAML preview with comments

### 3. Progress-Aware Summary Header

Persistent header showing current configuration state.

```typescript
interface ConfigSummary {
  model: string;           // "FLUX" or "Not selected"
  resolution: string;      // "1024px" or "Not set"
  steps: number;          // 2000
  estimatedVRAM: string;  // "~12GB"
  warnings: string[];     // ["Resolution not configured"]
}
```

**Visual Design:**
- Compact badges for key metrics
- Warning indicators for incomplete/problematic settings
- Updates in real-time as user progresses
- Clickable badges could jump to relevant step (future enhancement)

### 4. Side Panel Advisor

Three-section panel providing real-time guidance.

#### Section 1: Educational (What these settings do)
- Plain-language explanations
- How settings affect training
- Verbosity based on experience level

#### Section 2: Recommendations (Why these values)
- Specific reasoning based on system profile
- Cause-effect relationships
- Alternative suggestions

#### Section 3: Warnings & Performance
- Validation errors (blocking)
- Warnings (non-blocking but important)
- Performance estimates:
  - Estimated VRAM usage
  - Estimated step time
  - Total training time
  - Disk space for cache/checkpoints

### 5. Smart Defaults Engine

Leverages system profile, model info, dataset analysis, and user intent.

#### Batch Size Calculation (TODO #9 Integration)

```typescript
function calculateBatchDefaults(
  profile: SystemProfile,
  resolution: number,
  intent: UserIntent
): BatchConfig {
  const vramGB = profile.gpu.isUnifiedMemory
    ? profile.memory.unifiedMemory * 0.7  // Reserve 30% for system
    : profile.gpu.vramGB;

  let initialBatch: number;
  let maxBatch: number;

  if (resolution <= 512) {
    initialBatch = Math.min(16, Math.max(4, Math.floor(vramGB / 3)));
    maxBatch = initialBatch * 2;
  } else if (resolution <= 768) {
    initialBatch = Math.min(8, Math.max(2, Math.floor(vramGB / 4)));
    maxBatch = Math.min(16, initialBatch * 2);
  } else if (resolution <= 1024) {
    initialBatch = Math.min(4, Math.max(1, Math.floor(vramGB / 6)));
    maxBatch = Math.min(8, initialBatch * 2);
  } else {
    initialBatch = Math.min(2, Math.max(1, Math.floor(vramGB / 8)));
    maxBatch = Math.min(4, initialBatch * 2);
  }

  const autoScale = intent.priority !== 'quality'; // Quality prefers fixed batch

  return {
    batch_size: initialBatch,
    auto_scale_batch_size: autoScale,
    min_batch_size: 1,
    max_batch_size: maxBatch,
    batch_size_warmup_steps: 100
  };
}
```

#### Caching Strategy (TODO #2, #3 Integration)

```typescript
function calculateCachingStrategy(
  profile: SystemProfile,
  datasetInfo: DatasetInfo
): CacheConfig {
  // Estimate cache size: ~6MB per image for latents + embeddings
  const cacheSizeGB = (datasetInfo.total_images * 6) / 1024;
  const availableRAM = profile.memory.availableRAM - 8; // Reserve for OS

  if (profile.gpu.isUnifiedMemory) {
    // Apple Silicon: Always use memory cache (no GPU transfer overhead)
    return {
      cache_latents: true,
      cache_latents_to_disk: false,
      reason: 'Unified memory - in-memory cache optimal'
    };
  }

  if (cacheSizeGB < availableRAM * 0.7) {
    // TODO #2: Shared memory cache
    return {
      cache_latents: true,
      cache_latents_to_disk: false,
      reason: `Cache fits in RAM (${cacheSizeGB.toFixed(1)}GB < ${(availableRAM * 0.7).toFixed(1)}GB available)`
    };
  } else {
    // TODO #3: Memory-mapped disk cache
    return {
      cache_latents: false,
      cache_latents_to_disk: true,
      reason: `Cache too large for RAM (${cacheSizeGB.toFixed(1)}GB), using disk with memory-mapping`
    };
  }
}
```

#### GPU Prefetching (TODO #5 Integration)

```typescript
function calculatePrefetching(
  profile: SystemProfile,
  intent: UserIntent
): number {
  if (profile.gpu.isUnifiedMemory) {
    // Less benefit with unified memory
    return 1;
  }

  const baseValue = {
    hdd: 3,
    ssd: 2,
    nvme: 1
  }[profile.storage.type];

  if (intent.priority === 'speed') {
    return baseValue + 1;
  } else if (intent.priority === 'memory_efficiency') {
    return Math.max(0, baseValue - 1);
  }

  return baseValue;
}
```

#### Worker Configuration (TODO #4 Integration)

```typescript
function calculateWorkers(profile: SystemProfile): WorkerConfig {
  const baseWorkers = Math.floor(profile.memory.totalRAM / 8);
  const numWorkers = Math.min(baseWorkers, profile.cpu.cores, 8);

  return {
    num_workers: numWorkers,
    persistent_workers: numWorkers > 0 // TODO #4
  };
}
```

#### Unified Memory Handling

```typescript
function handleUnifiedMemory(profile: SystemProfile): AdvisorMessage[] {
  if (!profile.gpu.isUnifiedMemory) return [];

  return [
    {
      type: 'info',
      title: 'Unified Memory Detected',
      message: `Your system uses unified memory (${profile.memory.unifiedMemory}GB shared between CPU and GPU). ` +
               `This means VRAM and RAM share the same pool. Batch size calculations account for this.`
    },
    {
      type: 'tip',
      title: 'Unified Memory Optimization',
      message: 'GPU prefetching provides less benefit on unified memory since there\'s no discrete GPU transfer. ' +
               'In-memory caching is always optimal.'
    }
  ];
}
```

### 6. Review Step Details

#### Editable Summary View

All configured settings displayed in organized sections with inline edit capability:
- Click on any value to edit it directly
- Changes trigger re-validation and advisor updates
- Undo/redo support

#### Validation Report

```typescript
interface ValidationResult {
  errors: ValidationMessage[];    // Must fix before submit
  warnings: ValidationMessage[];  // Should review
  suggestions: ValidationMessage[]; // Optional improvements
}

interface ValidationMessage {
  field: string;
  message: string;
  severity: 'error' | 'warning' | 'suggestion';
  autoFix?: () => void; // Optional auto-fix action
}
```

Example validations:
- ERROR: "Model path is required"
- WARNING: "Batch size 8 at 1024px may exceed your 12GB VRAM"
- SUGGESTION: "Consider enabling auto-scale batch size for better GPU utilization"

#### Comparison View

Side-by-side view of user's config vs recommended defaults:

```
Setting              Your Value    Recommended    Reason
─────────────────────────────────────────────────────────
Batch Size           8             4              Lower safer for 1024px
Learning Rate        0.0001        0.0001         ✓ Matches recommendation
Steps                1000          1500           More steps for 150 images
GPU Prefetch         0             2              Enable for NVMe storage
```

Highlights deviations and explains implications.

#### Performance Predictions

```typescript
interface PerformancePrediction {
  estimatedVRAM: string;        // "14.2 GB (of 24 GB)"
  estimatedStepTime: string;    // "1.2 seconds"
  totalTrainingTime: string;    // "~33 minutes"
  diskSpaceNeeded: string;      // "12.5 GB (cache + checkpoints)"
  memoryUsage: string;          // "28 GB RAM"
}
```

## UI/UX Considerations

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Step Progress Bar (clickable steps for completed sections)      │
├─────────────────────────────────────────────────────────────────┤
│ Config Summary Header (persistent, updates in real-time)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─── Main Content (70%) ────┐  ┌─── Advisor Panel (30%) ───┐  │
│  │                           │  │                           │  │
│  │  Step title & description │  │  📚 Educational           │  │
│  │                           │  │  💡 Recommendations        │  │
│  │  [Form fields]            │  │  ⚠️ Warnings              │  │
│  │                           │  │  📊 Performance           │  │
│  │                           │  │                           │  │
│  └───────────────────────────┘  └───────────────────────────┘  │
│                                                                 │
│  [Exit Wizard]  [← Previous]                      [Next →]      │
└─────────────────────────────────────────────────────────────────┘
```

### Responsive Design

- **Desktop:** Full three-column layout
- **Tablet:** Advisor panel collapses to expandable sidebar
- **Mobile:** Sequential sections, advisor appears as modal on demand

### Accessibility

- Keyboard navigation through all steps
- Screen reader support for advisor messages
- High contrast mode for validation indicators
- Focus management between steps

## Technical Implementation

### State Management

```typescript
interface WizardState {
  currentStep: number;
  systemProfile: SystemProfile;
  userIntent: UserIntent;
  datasetInfo: DatasetInfo | null;
  jobConfig: JobConfig;
  validationResults: ValidationResult;
  advisorMessages: AdvisorMessage[];
}
```

### API Requirements

1. **GET /api/system/profile** - Auto-detect hardware (new endpoint)
2. **POST /api/dataset/analyze** - Existing endpoint for dataset analysis
3. **POST /api/config/validate** - Validate configuration (new endpoint)
4. **POST /api/config/recommend** - Get recommendations for current config (new endpoint)
5. **GET /api/models/{arch}/info** - Get model-specific information (new endpoint)

### Component Structure

```
GuidedWizardV2/
├── PreflightModal/
│   ├── SystemDetection.tsx
│   ├── IntentQuestionnaire.tsx
│   └── index.tsx
├── WizardContainer/
│   ├── ProgressBar.tsx
│   ├── SummaryHeader.tsx
│   ├── StepContent.tsx
│   ├── AdvisorPanel.tsx
│   └── index.tsx
├── Steps/
│   ├── ModelSelection.tsx
│   ├── Quantization.tsx
│   ├── TargetConfig.tsx
│   ├── DatasetSetup.tsx
│   ├── ResolutionAugmentation.tsx
│   ├── MemoryBatch.tsx
│   ├── OptimizerLearning.tsx
│   ├── Regularization.tsx
│   ├── TrainingCore.tsx
│   ├── SamplingConfig.tsx
│   ├── SaveSettings.tsx
│   ├── ModelSpecific.tsx
│   └── ReviewSubmit.tsx
├── Advisor/
│   ├── EducationalSection.tsx
│   ├── RecommendationsSection.tsx
│   ├── WarningsSection.tsx
│   └── PerformanceSection.tsx
├── SmartDefaults/
│   ├── batchCalculations.ts
│   ├── cachingStrategy.ts
│   ├── prefetchingStrategy.ts
│   ├── workerConfiguration.ts
│   └── unifiedMemoryHandling.ts
├── Validation/
│   ├── validators.ts
│   └── performancePredictor.ts
└── hooks/
    ├── useSystemProfile.ts
    ├── useSmartDefaults.ts
    ├── useValidation.ts
    └── useAdvisor.ts
```

## Migration Strategy

1. **Phase 1:** Implement pre-flight check and smart defaults engine
2. **Phase 2:** Build wizard container with step navigation
3. **Phase 3:** Implement individual step components
4. **Phase 4:** Add advisor panel with real-time updates
5. **Phase 5:** Build review step with validation and comparison
6. **Phase 6:** Testing and refinement

Keep existing GuidedWizard as fallback option during development.

## Success Metrics

1. **Configuration completeness** - Users configure more settings than with basic wizard
2. **Error reduction** - Fewer OOM errors and misconfigurations
3. **User confidence** - Users understand their choices (survey/feedback)
4. **Time to first training** - Faster than manual YAML editing
5. **Optimal resource utilization** - GPU/memory usage closer to optimal

## Future Enhancements

1. **Configuration templates** - Save/load wizard profiles
2. **A/B testing** - Compare different configurations
3. **Historical analysis** - Learn from past trainings
4. **Cloud integration** - Detect cloud instance specs automatically
5. **Collaborative configs** - Share configurations with explanations
6. **Real-time monitoring** - Show actual vs predicted performance during training

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance prediction inaccuracy | User frustration | Start with conservative estimates, refine with telemetry |
| Overwhelming for beginners | Abandonment | Progressive disclosure, experience level affects verbosity |
| System detection failures | Wrong recommendations | Always allow manual override, fallback to safe defaults |
| Unified memory complexity | Wrong batch calculations | Extensive testing on Apple Silicon, AMD systems |

## Conclusion

This comprehensive guided config wizard transforms the configuration experience from a simple 5-step process covering ~10 settings into an intelligent, educational 12-step wizard covering all ~100+ settings. By leveraging system detection, user intent, and existing optimizations (batch scaling, prefetching, caching), it provides smart defaults while maintaining full user control. The side panel advisor ensures users understand their choices, and the review step validates the entire configuration before submission.
