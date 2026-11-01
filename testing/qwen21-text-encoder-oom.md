# Qwen-Image 2.1 prompt-encoder memory fix

## Cause and scope

The failure was reported on an RTX 4070 Ti SUPER (16 GiB), with BF16,
`convrot8` on the transformer and text encoder, rank/alpha 32, resolution 512,
and layer offloading of 100% of the text encoder and 85% of the transformer.
The 125-image latent cache completed; the embedding-cache progress stopped at
63/125, before any training step.

The installed revision is `e9c72b762748e4d31837fbeb3b87d01a5b143321`, with
Transformers 5.5.3 and PyTorch 2.13.0+cu130. The original trace ends in:

```
Qwen3VLForConditionalGeneration.forward -> lm_head
OstrisLinearLayerMemoryManager._mm_forward
ConvRotInt8Quantizer.forward -> torch._int_mm(aq, qdata.t())
```

`QwenImage21PromptEncoder.encode` calls the entire conditional-generation
wrapper with `output_hidden_states=True`. Its default `logits_to_keep=0`
projects **every** token onto the vocabulary. The int8 GEMM produces an int32
intermediate of approximately `padded(B * L) * vocab_size * 4` bytes, in
addition to the staged vocabulary weights, dequantization epilogue and logits.
The trace reports a 1.30 GiB allocation failure at this operation. Offloading
reduces resident weights; it does not remove these activation allocations.

The dataloader uses `TEXT_EMBED_CACHE_BATCH=16`, independently of training
`batch_size=1`. It collects up to 64 pending captions, sorts them by character
length and encodes chunks of 16. Therefore the 63/125 progress value coincides
with a pending-cache flush; it does not establish that the 64th caption alone
is problematic. Padding increases the effective token count. The original
call also requests every layer's hidden states and defaults to retaining KV
cache, although the conditioning only uses the final hidden state.

## Targeted change

Only `extensions_built_in/diffusion_models/qwen_image_2/src/pipeline.py` changes
production behavior:

- Call `self.text_encoder.model(...)`, the same multimodal base model that the
  conditional-generation wrapper calls internally. This skips `lm_head`
  entirely. `logits_to_keep=1` would still stage/project the vocabulary weights.
- Explicitly disable hidden-state history and KV cache. Read the full
  `last_hidden_state`, with `return_dict=True` for a stable output contract.
- Preserve the existing final RMSNorm hook and its `finally` cleanup. The
  returned conditioning remains the full **pre-normalization** final decoder
  state, rather than accidentally switching to normalized embeddings.

Vision inputs, image grids, modality IDs, RoPE, left padding, system-prefix
removal, validity masks and image-slot masks follow the existing path. The
quantized modules and layer offloading hooks remain the same objects; no
weights or module forwards are globally replaced. Other model integrations
and the dataloader batch size are unchanged. The per-item cache writer already
moves saved tensors to CPU; its remaining local references are not the
vocabulary-sized allocation identified in this trace.

The Transformers base-model output and vocabulary projection contract are
documented in the [official Qwen3-VL reference for 5.5.3](https://huggingface.co/docs/transformers/v5.5.3/model_doc/qwen3_vl).

## Automated regression tests

```
python -m unittest testing.test_qwen21_prompt_encoder -v
```

Six tests use actual, randomly initialized Qwen3-VL modules, without downloads:
text and multimodal embedding equivalence, left-padding and masks, explicit
dictionary output, exception cleanup, `convrot8` CPU fallback, and CUDA
`convrot8` with 100% layer offloading (including image inputs). The reference
executes the original full LM wrapper and selects `hidden_states[-1]` under
the same RMSNorm hook. The patched path must never call `lm_head`, retain a KV
cache or return hidden-state history.

The comparison is against the **same** quantized weights and input batch,
not against an unquantized model. Required tolerance: `rtol=0, atol=0`.
CUDA coverage is explicitly skipped when no GPU is available. All six tests
passed on the installed environment above, including GPU/offloading coverage.

## Opt-in dataset/training validation

`testing/validate_qwen21_training.py` runs a separate short training job from
a read-only input configuration, using an explicitly supplied **disposable
dataset copy**. It writes generated configuration, caches, checkpoints and
measurements only to that copy and a new output directory. Do not supply a
production dataset. Keep model weights read-only and the original database
unmounted. The script clears the UI job ID and replaces the SQLite path.

```
python testing/validate_qwen21_training.py \
  --config /source-config.yaml \
  --dataset /validation/dataset \
  --output /validation/results --steps 3 --baseline-batch 4
```

Rank, alpha, image count, resolution, captions, optimizer, quantization,
gradient checkpointing and offloading settings are inherited unchanged.
Only the job name, output/dataset/SQLite paths, UI logging, checkpoint cadence
and total validation step count differ. The source configuration is not edited.

`--baseline-batch` selects which 1-based cache batch to profile (default 1).
Use a batch containing long captions to investigate the largest allocation.
When the baseline fits, the entire real batch must match exactly as well.
An optional `--processor` accepts a local copy of the same processor metadata
for offline environments, without changing the checkpoint resolver. In the
Docker validation, Transformers' AutoTokenizer requested a processor-side
`config.json` absent from the HF cache: the already-cached text-encoder config
was copied into a **disposable** local processor directory. Tokenizer files,
templates and weights were unchanged; the production cache was read-only.

The harness attempts the original batched call, records OOM or peak allocation,
compares one real prompt before/after exactly, measures patched prompt calls,
checks every image has a cache file and checks actual optimizer updates to
finite LoRA weights over at least three steps. A completed progress bar alone
is not sufficient: training can skip OOM batches, so optimizer updates are
counted separately. `metrics.json` records outcomes, even on failure.

PyTorch `max_memory_allocated` counts PyTorch-managed tensors, not all GPU
consumers. Sampled `total-free` device usage (100 ms interval) includes other
processes and allocator pools, and may miss transient peaks. These are
different measurements and must not be presented as interchangeable.

## Real GPU results (2026-11-01)

Two independent fresh embedding-cache runs completed **125/125**, followed by
**three actual AdamW8bit optimizer steps** each, on the RTX 4070 Ti SUPER above.
Rank/alpha 32, resolution 512, BF16, gradient checkpointing, `convrot8`, cache
batch 16 and the requested 100%/85% layer offloading were preserved.

The second run profiled cache batch 4, whose longest post-prefix-removal
sequence had 130 tokens. Before/after measurements on this **same batch**:

The real processor drops 14 system-prefix tokens, so the longest encoder
input is 144 tokens. Its original vocabulary GEMM needs
`16 * 144 * 151936 * 4 = 1,400,242,176` bytes (1.304 GiB) just for the int32
result, matching the failed 1.30 GiB allocation in the production trace.

| Path | Peak PyTorch allocated memory |
| --- | ---: |
| Original LM wrapper, all logits/hidden states and default KV cache | 5.11 GiB |
| Patched multimodal base model | 1.86 GiB |

This is approximately a **64% reduction** in this allocation metric. Every
embedding value and both masks matched exactly for the full 16-caption batch
(`rtol=0, atol=0`), as did a separate comparison on its longest caption.
The original path **did not OOM in these isolated comparisons**; the production
OOM is established by the supplied trace, not claimed as newly reproduced.
Available VRAM and allocator history differ between runs.

The second run's 100 ms sampled whole-device peak was 10.04 GiB during patched
embedding work and 15.19 GiB during training. The latter leaves limited VRAM
headroom; these three-step runs establish cache completion and functional LoRA
updates, not long-run stability, convergence or character fidelity.

The second run observed 192/384/384 LoRA tensors changed from initialization
after optimizer updates 1/2/3, with maximum absolute deltas approximately
`4.98e-5`, `9.92e-5`, `1.48e-4`. All checked weights were finite. The first
run's successive numbered checkpoints differed in 384 tensors; its final
save duplicates the last numbered checkpoint as expected. Optimizer-update
counts, rather than checkpoint filename counters, identify actual training
progress. Original dataset integrity was verified by SHA-256 for all 424
original files (images, captions and existing caches): no changes.

Validation ran in separate containers with no network, no mounted production
SQLite database/checkpoints and read-only model/config inputs. The live
AI-Toolkit container was not modified or restarted. Full private datasets,
captions, configurations, checkpoints and console logs are intentionally not
included in this public change; only the generic harness and aggregate
results are published.

## Apply and roll back

Review/cherry-pick the commit into a checkout matching the deployed version;
run the unit tests, then the disposable-dataset validation. For Docker, build
a separate image from the reviewed checkout and replace the service only in
an explicitly approved maintenance window. Keep the previous image tag and
compose configuration for rollback. A Docker image update alone will not
include this fix until the upstream project publishes it.

Revert the commit in the source checkout, or deploy the previous image/tag,
to roll back. No model-weight conversion or database migration is involved.
Do not edit, delete or reuse existing production checkpoints as part of this
procedure. The live service has not been patched or restarted by this work.
