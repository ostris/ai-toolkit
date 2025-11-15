# AI Toolkit - Future Optimizations & TODOs

This document tracks potential improvements and optimizations for the AI Toolkit project.

## High Priority

### 🚀 Performance Optimizations

#### 1. Lazy Model Loading in Dataloader Workers (COMPLETED ✓)
**Impact:** Saves ~19GB memory per worker for large models

**Status:** Completed

**Implementation:**
- Added `is_worker_process()` utility function to detect worker processes
- Implemented `__getstate__` and `__setstate__` in `AiToolkitDataset` to exclude `self.sd` from pickle
- Workers receive dataset without the model, only cached data
- Added defensive checks in caching methods to fail gracefully if `self.sd` is None

**How it works:**
- During initialization in main process, `self.sd` is available for caching operations
- When dataset is pickled to worker processes, `__getstate__` sets `self.sd` to None
- Workers load cached latents/embeddings without needing the model
- Memory savings: ~19GB per worker for large models like Qwen-Image

**Files modified:**
- `toolkit/data_loader.py`: Added `is_worker_process()`, `__getstate__`, `__setstate__`
- `toolkit/dataloader_mixins.py`: Added safety checks in caching methods

**Complexity:** Medium
**Actual savings:** 19GB per worker

---

#### 2. Shared Memory for Cached Data
**Impact:** Near-zero memory duplication for cached latents/embeddings

**Current issue:**
- Each worker gets a copy of all cached latents in RAM
- With cache_latents: true, massive duplication

**Proposed solution:**
```python
# In main process
import torch.multiprocessing as mp
shared_cache = mp.Manager().dict()

# Each FileItemDTO stores key, not full tensor
file_item.latent_cache_key = f"latent_{hash(file_item.path)}"
shared_cache[file_item.latent_cache_key] = latent_tensor

# Workers access shared memory
latent = shared_cache[file_item.latent_cache_key]
```

**Alternative: Use PyTorch shared tensors:**
```python
# Store in shared memory
latent_tensor.share_memory_()  # Makes tensor accessible across processes
```

**Files to modify:**
- `toolkit/dataloader_mixins.py`: LatentCachingMixin
- `toolkit/data_loader.py`: get_dataloader_from_datasets

**Complexity:** High (requires significant refactoring)
**Estimated savings:** All cached data becomes shared (hundreds of MB to GBs)

---

#### 3. Memory-Mapped Tensor Storage
**Impact:** Near-zero memory per worker, still fast access

**Proposed solution:**
```python
# Instead of loading full tensor into RAM
latent = torch.load(latent_path)  # Current

# Use memory mapping
latent = torch.from_file(
    latent_path,
    shared=True,
    size=tensor_size,
    dtype=torch.float16
)  # OS handles paging from disk
```

**Files to modify:**
- `toolkit/dataloader_mixins.py`: LatentCachingFileItemDTOMixin.get_latent()
- `toolkit/fastsafetensors_utils.py`: Add mmap support

**Complexity:** Medium
**Estimated savings:** All cached tensors become memory-mapped

**Note:** This pairs well with fastsafetensors which already uses mmap internally

---

#### 4. Expose persistent_workers Configuration
**Impact:** Faster epoch transitions, better worker utilization

**Current issue:**
- Workers restart between epochs
- Startup overhead repeated unnecessarily

**Proposed solution:**
```yaml
train:
  datasets:
    - folder_path: "/path/to/dataset"
      num_workers: 2
      persistent_workers: true  # Keep workers alive between epochs
```

**Files to modify:**
- `toolkit/config_modules.py`: Add persistent_workers to DatasetConfig
- `toolkit/data_loader.py`: Pass to DataLoader constructor (line 662, 671)

**Complexity:** Low (just expose existing PyTorch feature)
**Estimated benefit:** Faster multi-epoch training

---

### 💾 Caching Improvements

#### 5. Intelligent Cache Warming
**Impact:** Faster training start, better GPU utilization

**Proposed solution:**
- Prefetch next N batches to GPU asynchronously
- Start loading epoch N+1 cache during epoch N
- Pipeline cache loading with training

**Files to modify:**
- `toolkit/dataloader_mixins.py`: Add prefetch logic
- New file: `toolkit/cache_prefetcher.py`

**Complexity:** High
**Estimated benefit:** Reduced training idle time

---

#### 6. Compressed Latent Cache
**Impact:** 2-4x smaller cache files

**Current format:**
- Raw float16/bfloat16 tensors
- ~4-8MB per 1024x1024 image

**Proposed solution:**
```python
# Save with compression
torch.save(
    {'latent': latent},
    latent_path,
    compression='gzip'  # or 'lz4' for faster
)
```

**Alternative: Use safetensors with quantization:**
```python
# Quantize to int8 for storage
latent_quantized = (latent * 127).to(torch.int8)
# Dequantize on load
latent = latent_quantized.to(torch.float16) / 127
```

**Files to modify:**
- `toolkit/dataloader_mixins.py`: LatentCachingMixin save/load methods

**Complexity:** Low
**Estimated savings:** 50-75% disk space

---

### 🔧 Code Quality & Maintainability

#### 7. Refactor Dataloader Mixins
**Impact:** Easier to maintain, better separation of concerns

**Current issue:**
- Large mixin classes with multiple responsibilities
- Hard to test independently

**Proposed solution:**
- Split into smaller, focused mixins
- Use composition over inheritance where possible
- Add unit tests for each mixin

**Files to modify:**
- `toolkit/dataloader_mixins.py`: Split into multiple files
- New directory: `toolkit/dataloader_mixins/`

**Complexity:** Medium
**Estimated benefit:** Better code organization

---

#### 8. Add Dataloader Memory Profiling Tool
**Impact:** Easier debugging of memory issues

**Proposed solution:**
```bash
# New diagnostic tool
python -m toolkit.diagnose_dataloader_memory config/your_config.yaml

# Output:
# Dataset memory breakdown:
#   - Model reference: 19.2 GB
#   - Cached latents: 800 MB (100 images)
#   - Text embeddings: 200 MB
#   - Estimated per worker: 20.2 GB
#
# Recommendations:
#   - Enable cache_latents_to_disk to save 800 MB per worker
#   - Consider num_workers: 1 to save 20.2 GB
```

**Files to create:**
- `toolkit/diagnose_dataloader_memory.py`

**Complexity:** Low
**Estimated benefit:** Easier troubleshooting

---

## Medium Priority

### 🎯 Features

#### 9. Smart Batch Size Scaling
**Impact:** Better GPU utilization, automatic tuning

**Proposed solution:**
- Automatically detect optimal batch size based on available memory
- Gradually increase batch size during training (warmup)
- Handle OOM gracefully and retry with smaller batch

**Files to modify:**
- New file: `toolkit/batch_size_tuner.py`
- `jobs/process/BaseSDTrainProcess.py`: Integrate tuner

**Complexity:** High
**Estimated benefit:** Easier configuration, better performance

---

#### 10. Multi-GPU Data Parallelism Optimizations
**Impact:** Better scaling across multiple GPUs

**Proposed improvements:**
- Smarter batch distribution across GPUs
- Per-GPU cache directories to avoid I/O bottleneck
- Shared cache access for multi-node training

**Files to modify:**
- `toolkit/data_loader.py`: Add DDP-aware batching
- `toolkit/dataloader_mixins.py`: Per-GPU cache paths

**Complexity:** High
**Estimated benefit:** Better multi-GPU scaling

---

#### 11. Dataset Streaming for Very Large Datasets
**Impact:** Enable training on datasets > available RAM

**Proposed solution:**
- Stream from cloud storage (S3, GCS, etc.)
- On-the-fly processing without local cache
- Intelligent prefetching and buffering

**Files to create:**
- `toolkit/streaming_dataset.py`

**Complexity:** Very High
**Estimated benefit:** Support for massive datasets

---

### 📊 Monitoring & Observability

#### 12. Training Metrics Dashboard
**Impact:** Better visibility into training progress

**Proposed features:**
- Real-time memory usage tracking
- Dataloader throughput metrics
- Cache hit/miss rates
- Worker utilization stats

**Files to create:**
- `toolkit/metrics_collector.py`
- Add to web UI

**Complexity:** Medium
**Estimated benefit:** Easier monitoring and debugging

---

#### 13. Add Timestamps to All Logging (COMPLETED ✓)
**Status:** Already implemented in commit e08d182

**Implementation:**
- Added `get_timestamp()` to `toolkit/print.py`
- All `print_acc()` calls now include timestamps
- Format: `[YYYY-MM-DD HH:MM:SS.mmm]`

---

## Low Priority

### 🧪 Testing

#### 14. Add Unit Tests for Dataloader
**Impact:** Prevent regressions, easier refactoring

**Proposed tests:**
- Test each mixin independently
- Test cache loading/saving
- Test bucket resolution logic
- Test worker memory isolation

**Files to create:**
- `tests/test_data_loader.py`
- `tests/test_dataloader_mixins.py`

**Complexity:** Medium
**Estimated benefit:** Better code quality

---

#### 15. Integration Tests for fastsafetensors
**Impact:** Catch API changes early

**Proposed tests:**
- Test loading with and without GDS
- Test sharded model loading
- Test error handling and fallback
- Performance benchmarks

**Files to create:**
- `tests/test_fastsafetensors_integration.py`

**Complexity:** Low
**Estimated benefit:** More reliable fastsafetensors integration

---

### 📝 Documentation

#### 16. Video Tutorials for Common Configurations
**Impact:** Easier onboarding for new users

**Proposed videos:**
- Setting up efficient dataloader config
- Debugging memory issues
- Optimizing for different hardware

**Complexity:** Low
**Estimated benefit:** Better user experience

---

#### 17. Interactive Config Generator
**Impact:** Easier optimal configuration

**Proposed tool:**
```bash
python -m toolkit.config_wizard

# Questions:
# - GPU model? [GB10]
# - Available RAM? [512GB]
# - Dataset size? [1000 images]
# - Training goal? [LoRA/full fine-tune]
#
# Generated optimal config saved to: config/optimized.yaml
```

**Files to create:**
- `toolkit/config_wizard.py`

**Complexity:** Low
**Estimated benefit:** Easier setup

---

## Completed

### ✅ fastsafetensors Integration
**Status:** Completed

**Implementation:**
- Added fastsafetensors support with GDS detection
- Automatic fallback to non-GDS mode
- 4-5x speedup for model/cache loading
- Comprehensive documentation

**Commits:**
- e08d182: Fix fastsafetensors API
- e0b0687: Add GDS detection and fallback
- 6a4f28b: Add technical documentation
- 7629cbd: Add worker memory guide

---

### ✅ Dataloader Worker Memory Documentation
**Status:** Completed

**Documentation:**
- Root cause analysis of worker memory usage
- Multiple solutions with examples
- Recommended configurations
- Diagnostic commands

**File:** `docs/DATALOADER_WORKER_MEMORY.md`

---

### ✅ GDS Diagnostic Tooling
**Status:** Completed

**Tools:**
- `testing/check_gds_status.py`: Check GDS availability
- `testing/diagnose_gds_error.py`: Debug error 5048
- Comprehensive troubleshooting guide

---

## Contributing

To propose a new TODO item:
1. Add it to the appropriate priority section
2. Include impact assessment
3. Estimate complexity
4. List files to modify
5. Submit PR with `[TODO]` prefix

To claim a TODO item:
1. Comment on GitHub issue or create one
2. Move item to "In Progress" section
3. Update with your progress
4. Submit PR when complete
5. Move to "Completed" section

## Legend

- **Impact:** Expected benefit (memory savings, speed improvement, UX)
- **Complexity:**
  - Low: < 1 day work
  - Medium: 1-3 days work
  - High: 1-2 weeks work
  - Very High: > 2 weeks work
- **Estimated savings/benefit:** Quantified where possible
