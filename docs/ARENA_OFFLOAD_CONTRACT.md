# Arena offload capability contract

Arena offload does not select support by model architecture or quantization
name. An explicitly selected model is accepted when its live module graph and
storage satisfy the following contracts; otherwise setup fails at the narrowest
known boundary with the unmet contract in the error.

## Model contract

- The transformer exposes one or more repeated `ModuleList` or `Sequential`
  block containers, either by unambiguous discovery or by declarative container
  paths from the model integration.
- Selected blocks keep their ordinary installed forwards. The dispatcher does
  not reconstruct model dataflow or block arguments.
- A selected block receives at least one tensor leaf in its positional or
  keyword argument structure and returns a tensor or a nested Python structure
  containing a tensor leaf.
- Model-owned gradient checkpointing is enabled before canonical commit.
  Intentionally uncheckpointed selected blocks remain resident.
- Canonical managed leaves are frozen base weights. Trainable adapters remain
  ordinary state outside canonical storage.
- Every selected-block parameter and buffer is enumerable before commit.
  Shared, parametrized, missing, or conflicting managed state fails before the
  destructive boundary.

## Quantization contract

- Each managed linear exposes a `LayerStorageBinding` through
  `module_storage_binding()`.
- The binding enumerates every physical tensor leaf without materializing a
  dequantized weight, supplies stable execution metadata, and declares how live
  parameter or buffer targets are reconstructed from those leaves.
- Construction and dispatcher finalization both verify that every substitution
  target exists. Unknown tensor-subclass storage fails before canonical commit.
- Transfer and residency code treats declared leaves as opaque tensors. It does
  not branch on qtype or quantization backend identity.

## Maintainer validation matrix

The upstream gate is the matrix, not an allowlist. Each selected production
model and quantization row must establish the relevant mechanisms below.

| Gate | Required evidence |
| --- | --- |
| Model independence | At least two production transformer architectures use the same discovery, saved-forward dispatcher, checkpoint owner, and lifecycle APIs. |
| Quantization independence | Plain, TorchAO/Quanto tensor-subclass, and Ostris packed-buffer layouts pass declaration, canonical construction, resident/streamed execution, and teardown checks where available. |
| Training | Resident and streamed adapter training produce finite gradients; checkpoint backward performs the planned re-fetches. |
| Sampling transition | Train -> sample -> train preserves canonical storage and restores the training residency plan. |
| Structured ABI | Positional or keyword tensor inputs and tensor or nested tensor outputs preserve the original block result. |
| Sequential jobs | Arena ownership, legacy manager state, transfer runtime, device ring registry, and pin ledger return to baseline before the next job. |
| Save/resume | Supported quantized layouts dequantize/save and resume through their existing serialization path without arena-specific state. |

Production runs remain maintainer- or user-launched. Focused tests should assert
mechanism and numerical parity; they must not replace this matrix with model or
qtype name checks.
