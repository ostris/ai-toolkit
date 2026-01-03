# AI Toolkit - Copilot Instructions

## Project Overview

AI Toolkit is an all-in-one training suite for diffusion models (image and video), supporting FLUX.1, Stable Diffusion, and other modern models on consumer GPUs. It provides both GUI and CLI interfaces with extensive configuration through YAML files.

## Architecture

### Core Components

- **Jobs System** (`jobs/`): Job types include `train`, `extract`, `generate`, `mod`, and `extension`
  - Entry point: `run.py` parses config files and instantiates jobs via `toolkit/job.py:get_job()`
  - Each job type has a dedicated class (e.g., `TrainJob`, `ExtensionJob`) that loads and runs processes
  
- **Process Architecture** (`jobs/process/`): Hierarchical process classes handle specific training workflows
  - `BaseProcess` → `BaseTrainProcess` → `BaseSDTrainProcess` (for Stable Diffusion variants)
  - Each process implements `run()` method for execution
  - Training processes are registered in job's `process_dict` (e.g., `slider`, `lora_hack`, `vae`)

- **Extensions System** (`extensions_built_in/`): Modular training implementations
  - Extensions register via `AI_TOOLKIT_EXTENSIONS` list in `__init__.py`
  - Built-in extensions: `sd_trainer`, `flex2`, `concept_slider`, `ultimate_slider_trainer`, `diffusion_models`, etc.
  - Custom extensions can be added to `extensions/` folder following same pattern

- **Configuration** (`toolkit/config.py`): YAML-based with template substitution
  - Supports `[name]` tag replacement and `${ENV_VAR}` environment variable substitution
  - Config files searched in: `config/`, absolute paths, or current directory
  - Extensions: `.json`, `.jsonc`, `.yaml`, `.yml`

- **Data Loading** (`toolkit/data_loader.py`): Bucketing system for variable-resolution training
  - Mixins: `CaptionMixin`, `BucketsMixin`, `LatentCachingMixin`, `CLIPCachingMixin`
  - Automatic image resizing and bucketing based on config resolution list
  - Supports caption files (`.txt`) with `[trigger]` word replacement

## Development Workflows

### Training a Model

```bash
# Basic workflow
python run.py config/my_training_config.yaml

# With custom name (replaces [name] in config)
python run.py config/my_training_config.yaml --name my_experiment

# Multiple configs sequentially
python run.py config/config1.yaml config/config2.yaml --recover  # continue on failure
```

### Using the UI

```bash
cd ui
npm run build_and_start  # Available at http://localhost:8675
```

- UI requires Node.js >18, optional auth token via `AI_TOOLKIT_AUTH` env var
- UI starts/stops/monitors jobs but doesn't need to stay running

### Environment Setup

**Critical**: Install PyTorch **before** `requirements.txt`:
```bash
pip3 install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

**FLUX.1-dev**: Requires HF token in `.env` file: `HF_TOKEN=your_key_here` and accepting license at [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

## Project-Specific Conventions

### Configuration Patterns

**Network kwargs for layer targeting**:
```yaml
network:
  type: "lora"
  linear: 128
  linear_alpha: 128
  network_kwargs:
    only_if_contains:  # Train only specific layers
      - "transformer.single_transformer_blocks.7.proj_out"
    ignore_if_contains:  # Exclude layers (takes priority)
      - "transformer.single_transformer_blocks."
```

**Model-specific settings**:
- FLUX.1: Requires `is_flux: true`, `quantize: true` for 24GB VRAM
- FLUX.1-dev: Non-commercial license, requires HF token
- FLUX.1-schnell: Apache 2.0, requires `assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"`

### Dataset Structure

- Folder of images (`.jpg`, `.jpeg`, `.png`) with matching `.txt` caption files
- Caption files support `[trigger]` placeholder replaced by config's `trigger_word`
- No preprocessing needed - images auto-resized to bucket resolutions
- Set `cache_latents_to_disk: true` to avoid recomputation

### Common Patterns

**Accelerator usage**: Access via `toolkit.accelerator.get_accelerator()`, wrapped around HF Accelerate
```python
from toolkit.accelerator import get_accelerator
accelerator = get_accelerator()
if accelerator.is_main_process:
    # Main process only code
```

**Extension registration**: Create `__init__.py` with:
```python
from .my_extension import MyExtension
AI_TOOLKIT_EXTENSIONS = [MyExtension]
```

## Key Integration Points

- **Modal/Runpod**: Pre-configured templates available, use absolute paths like `/root/ai-toolkit`
- **HuggingFace**: Hub integration via `hf_transfer` (enabled in run.py), supports `push_to_hub` in save configs
- **Docker**: Official Dockerfile in `docker/`, exposed via `docker-compose.yml`
- **Notebooks**: Jupyter examples in `notebooks/` for FLUX/Slider training

## VRAM Requirements

- **FLUX.1**: Minimum 24GB VRAM, set `low_vram: true` if GPU drives monitors (slower CPU quantization)
- **LoRA training**: 16-24GB depending on model and batch size
- **Full fine-tuning**: 40GB+ for most models

## Troubleshooting

- **Corrupted checkpoint**: Avoid `Ctrl+C` during save operations (wait for completion message)
- **Windows issues**: Use [Easy Install script](https://github.com/Tavris1/AI-Toolkit-Easy-Install) or WSL
- **Import errors**: Ensure PyTorch installed before requirements.txt
- **Memory errors**: Enable `gradient_checkpointing: true` and lower batch size

## Extension Development

### Creating a Custom Extension

Extensions follow a strict pattern for lazy loading and registration:

**1. Create extension class** (`extensions/my_extension/__init__.py`):
```python
from toolkit.extension import Extension

class MyTrainerExtension(Extension):
    uid = "my_trainer"  # Must be unique, used in config: type: 'my_trainer'
    name = "My Trainer"  # Display name for logging
    
    @classmethod
    def get_process(cls):
        # Lazy import - only loaded when extension is used
        from .MyTrainer import MyTrainer
        return MyTrainer

AI_TOOLKIT_EXTENSIONS = [MyTrainerExtension]  # Must export this list
```

**2. Implement process class** (`extensions/my_extension/MyTrainer.py`):
```python
from jobs.process import BaseExtensionProcess
from collections import OrderedDict

class MyTrainer(BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        # Access config values with self.get_conf()
        self.my_param = config.get('my_param', 'default')
        
    def run(self):
        super().run()  # Always call parent run()
        # Your training logic here
```

**3. Use in config**:
```yaml
job: extension
config:
  name: "my_training_run"
  process:
    - type: 'my_trainer'  # Matches your extension's uid
      my_param: "custom_value"
```

### Built-in Extension Examples

**sd_trainer**: Generic LoRA/Dreambooth/FineTuning trainer
- Implements `BaseSDTrainProcess` (2000+ lines with adapters, guidance, EMA)
- Handles assistant adapters (`T2IAdapter`, `ControlNetModel`), IP-Adapter, CLIP vision
- Supports flow matching (`flowmatch`), SNR weighting, gradient checkpointing
- Use as reference for advanced training features

**concept_slider**: Trains concept direction sliders
- Teaches semantic concept manipulation (e.g., "more detailed" ↔ "less detailed")
- Config requires positive/negative prompt pairs

**ultimate_slider_trainer**: Advanced slider training with multiple techniques

**diffusion_models**: Universal trainer for various diffusion architectures

### Adding UI Support for Extensions

Extensions can integrate with the web UI by registering configuration options in `ui/src/app/jobs/new/options.ts`.

**Pattern for UI integration**:

1. **Add job type option** in `jobTypeOptions` array:
```typescript
export const jobTypeOptions: JobTypeOption[] = [
  {
    value: 'my_trainer',  // Matches extension uid
    label: 'My Custom Trainer',
    disableSections: ['trigger_word', 'slider'],  // Hide irrelevant UI sections
    onActivate: (config: JobConfig) => {
      // Initialize extension-specific config
      config.config.process[0].my_extension_config = { 
        custom_param: 'default_value' 
      };
      return config;
    },
    onDeactivate: (config: JobConfig) => {
      // Cleanup when user switches away
      delete config.config.process[0].my_extension_config;
      return config;
    },
  },
];
```

2. **Define TypeScript interfaces** in `ui/src/types.ts`:
```typescript
export interface MyExtensionConfig {
  custom_param?: string;
  another_param?: number;
}

export interface ProcessConfig {
  type: string;
  // ... existing fields
  my_extension_config?: MyExtensionConfig;
}
```

3. **Add UI form fields** in `ui/src/app/jobs/new/SimpleJob.tsx`:
```tsx
{jobType?.value === 'my_trainer' && (
  <Card title="My Extension Settings">
    <TextInput
      label="Custom Parameter"
      value={jobConfig.config.process[0].my_extension_config?.custom_param ?? ''}
      onChange={value => setJobConfig(value, 'config.process[0].my_extension_config.custom_param')}
      placeholder="Enter value"
    />
    <NumberInput
      label="Another Parameter"
      value={jobConfig.config.process[0].my_extension_config?.another_param ?? 10}
      onChange={value => setJobConfig(value, 'config.process[0].my_extension_config.another_param')}
    />
  </Card>
)}
```

**Example: Concept Slider UI**

The `concept_slider` extension demonstrates full UI integration:
- **Job type**: `value: 'concept_slider'` in `jobTypeOptions`
- **Config interface**: `SliderConfig` with `positive_prompt`, `negative_prompt`, `target_class`, `anchor_class`
- **UI fields**: Conditional card shown when `!disableSections.includes('slider')`
- **State management**: `onActivate` adds `defaultSliderConfig`, `onDeactivate` removes it

**DisableableSections**: Control which standard UI sections appear:
- `'trigger_word'`: Hide trigger word input
- `'slider'`: Hide slider configuration card  
- `'network.conv'`: Hide convolutional layer training options
- `'train.diff_output_preservation'`: Hide diff output preservation settings
- `'model.quantize'`: Hide quantization options

**AdditionalSections**: Show optional UI sections per model/extension:
- `'datasets.control_path'`: Show control image path input
- `'sample.ctrl_img'`: Show control image sampling options
- `'model.low_vram'`: Show low VRAM optimization toggle
- `'model.assistant_lora_path'`: Show assistant LoRA path input

## Dataset Preprocessing Tools

### SuperTagger Extension

Automated captioning and image preprocessing using LLaVA or Fuyu models.

**Key features**:
- **Batch captioning**: Long and short captions with custom prompts
- **Image preprocessing**: Contrast stretch, auto-corrections
- **Processing pipeline**: Steps system tracks completion state
- **Directory structure**: `raw/` → processing → `train/` with `.json` metadata

**Config example**:
```yaml
job: extension
config:
  process:
    - type: 'super_tagger'
      parent_dir: "/path/to/datasets"  # Or list dataset_paths
      caption_method: 'llava:default'  # or 'fuyu:default'
      caption_prompt: "Describe this image in detail"
      caption_short_prompt: "Brief description"
      steps: ['caption', 'caption_short', 'contrast_stretch']
      force_reprocess_img: false
```

**Processing flow**:
1. Images in `{dataset}/raw/` directory
2. SuperTagger creates `{dataset}/train/` with:
   - Processed images (same filename)
   - `.json` metadata files (tracks steps, version, captions)
3. Metadata includes: `caption`, `caption_short`, completed `steps`, `version`

### SyncFromCollection Extension

Downloads images from Unsplash/Pexels collections for dataset building.

**Config example**:
```yaml
job: extension
config:
  process:
    - type: 'sync_from_collection'
      min_width: 1024
      min_height: 1024
      dataset_sync:
        - host: 'unsplash'  # or 'pexels'
          collection_id: '12345'
          directory: '/path/to/dataset'
        - host: 'pexels'
          collection_id: '67890'
          directory: '/path/to/another_dataset'
```

**Directory structure**:
- `{dataset}/new/`: Downloaded images (temporary)
- `{dataset}/raw/`: Finalized raw images
- Script moves `new/` → `raw/` after download completes

### Utility Scripts

**repair_dataset_folder.py**: Fixes corrupt/missing caption files
- Located in `scripts/` directory
- Run directly: `python scripts/repair_dataset_folder.py --path /dataset`

**Dataset structure requirements**:
```
dataset/
├── image1.jpg
├── image1.txt      # Caption file (same name as image)
├── image2.png
├── image2.txt
└── ...
```

**Caption processing**:
- Extensions: `.jpg`, `.jpeg`, `.png`, `.webp` (images), `.txt` (captions)
- `[trigger]` placeholder in `.txt` files replaced with config's `trigger_word`
- `caption_dropout_rate`: Randomly drop captions during training (e.g., 0.05 = 5%)
- `shuffle_tokens`: Shuffle comma-separated caption tokens for regularization

## Reference Files

- Example configs: `config/examples/*.yaml` (start here for model-specific templates)
- FAQ: `FAQ.md` (VRAM requirements, common issues)
- Version: Check `version.py` for current release
- Dependencies: `requirements.txt` + pre-install PyTorch separately
- Extension examples: `extensions_built_in/*/` for reference implementations
