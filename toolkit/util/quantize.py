import os
import time
from fnmatch import fnmatch
from typing import List, Optional, Union, TYPE_CHECKING
import torch

from optimum.quanto.quantize import _quantize_submodule
from optimum.quanto.tensor import Optimizer, qtype, qtypes
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
    Float8WeightOnlyConfig,
    Int8WeightOnlyConfig
)
from optimum.quanto import freeze
from tqdm import tqdm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from toolkit.print import print_acc
from toolkit.util.ostris_quant import (
    OstrisLinear,
    OstrisQuantizer,
    convert_linear_to_ostris,
    get_ostris_quantizer,
)
import os

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

# the quantize function in quanto had a bug where it was using exclude instead of include

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
    "OstrisLinear",
]

torchao_qtypes = {
    # "int4": Int4WeightOnlyConfig(),
    # uint2..uint8 are handled by the UIntXQuantizer ostris backend
    # (toolkit/util/uintx_quant.py), a bit-exact reproduction of torchao 0.10.0's
    # UIntXWeightOnlyConfig, so ARAs stay byte-identical after torchao upgrades
    "int8": Int8WeightOnlyConfig(),
    "float8": Float8WeightOnlyConfig(),
}


class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


class ostristype:
    # custom quantization backend (see toolkit/util/ostris_quant.py), e.g. orbit2/3/4
    def __init__(self, name: str, quantizer: OstrisQuantizer):
        self.name = name
        self.quantizer = quantizer


def get_qtype(qtype: Union[str, qtype]) -> qtype:
    if qtype in torchao_qtypes:
        return aotype(qtype)
    if isinstance(qtype, str):
        ostris_quantizer = get_ostris_quantizer(qtype)
        if ostris_quantizer is not None:
            return ostristype(qtype, ostris_quantizer)
        return qtypes[qtype]
    else:
        return qtype


def is_quantized_tensor(t) -> bool:
    # torchao stores quantized weights as tensor subclasses (e.g. AffineQuantizedTensor) under torchao.*
    # that still report as nn.Parameter and expose .dequantize(). (quanto is handled separately.)
    # _is_ostris_weight tags two OstrisLinear tensors: the .weight property's eager tensor
    # (already dequantized; .dequantize() is a no-op) so the merge paths route through
    # requantize_module_weight, and the lazy OstrisLazyWeight emitted by state_dict()
    # (holds no data; .dequantize() materializes) so save loops dequantize it per key.
    if getattr(t, '_is_ostris_weight', False):
        return True
    return 'torchao' in type(t).__module__ and hasattr(t, 'dequantize')


def dequantize_if_quantized(t):
    return t.dequantize() if is_quantized_tensor(t) else t


def get_torchao_config(qtype):
    # returns the requantization config for a given qtype string (a torchao config, or the
    # ostristype for custom backends), or None if the qtype supports neither
    if qtype is None:
        return None
    try:
        q = get_qtype(qtype)
    except Exception:
        return None
    if isinstance(q, aotype):
        return q.config
    if isinstance(q, ostristype):
        return q
    return None


def requantize_module_weight(module, fp_weight, orig_dtype, config) -> None:
    """Write a full precision weight back into module.weight, re-quantizing in place if a
    requantization config is provided so the module stays quantized (used by the continuous
    merge/reset method). If config is None the weight is left in full precision."""
    if isinstance(module, OstrisLinear):
        # the module's backend reuses its existing quantization state; config is not needed
        module.requantize_(fp_weight)
        return
    if isinstance(config, ostristype):
        # custom backend config but the module was never converted (e.g. skipped at
        # quantize time); leave it in full precision
        config = None
    module.weight = torch.nn.Parameter(fp_weight.to(orig_dtype), requires_grad=False)
    if config is not None:
        torchao_quantize_(module, config)


def _wrap_qlinear_ndim(qlinear: torch.nn.Module) -> None:
    """quanto QLinear forward that tolerates >3D activations by flattening
    the leading dims for the mm and restoring them after (no-op otherwise)."""
    orig_forward = qlinear.forward

    def forward(x):
        if x.ndim > 3:
            lead = x.shape[:-1]
            out = orig_forward(x.reshape(-1, x.shape[-1]))
            return out.reshape(*lead, out.shape[-1])
        return orig_forward(x)

    qlinear.forward = forward
    qlinear._aitk_ndim_wrapped = True


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
    quantize_device: Optional[torch.device] = None,
    keep_on_quantize_device: bool = False,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
        quantize_device (`Optional[torch.device]`):
            If provided, each module is moved to this device to quantize, then moved
            back to the device its weights were on initially. Lets a CPU-resident
            model (low vram) quantize layer-by-layer on the GPU.
        keep_on_quantize_device (`bool`):
            With quantize_device set: leave each layer on the quantize device after
            quantizing instead of moving it back. The gpu transient is then one
            bf16 layer at a time — never the whole unquantized remainder at once —
            for models whose final home is that device.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                # OstrisLinear may still be RE-quantized into a different
                # ostris qtype (same qtype is a per-layer no-op); every other
                # already-quantized module type is always left alone, which
                # also keeps quanto/torchao from double-quantizing
                # pre-quantized checkpoints
                if not (
                    isinstance(weights, ostristype)
                    and m.__class__.__name__ == "OstrisLinear"
                ):
                    continue
                if getattr(m.ostris_quantizer, "qtype", None) == weights.quantizer.qtype:
                    # guaranteed per-layer no-op: skip before any
                    # quantize_device round-trip
                    continue
            if isinstance(weights, ostristype) and not isinstance(m, torch.nn.Linear):
                # ostris backends only quantize nn.Linear; don't ferry norms/
                # embeddings across the bus for nothing when quantize_device
                # is set (containers fall through so children are visited)
                if quantize_device is not None and next(m.children(), None) is None:
                    continue
            if (
                isinstance(weights, aotype)
                and not isinstance(m, torch.nn.Linear)
                and (
                    quantize_device is not None
                    or include is not None
                    or exclude is not None
                )
            ):
                # torchao only quantizes nn.Linear; when a device round-trip or
                # include/exclude filtering is in play, skip containers so each
                # linear is handled individually (a container-level torchao call
                # would quantize excluded children too)
                continue
            orig_device = None
            if quantize_device is not None and next(m.children(), None) is None:
                # OstrisLinear layers being re-quantized hold buffers, not params
                param = next(m.parameters(recurse=False), None)
                if param is None:
                    param = next(m.buffers(recurse=False), None)
                if param is not None:
                    orig_device = param.device
                    m.to(quantize_device)
            try:
                if isinstance(weights, ostristype):
                    if isinstance(m, torch.nn.Linear):
                        convert_linear_to_ostris(m, weights.quantizer)
                elif isinstance(weights, aotype):
                    torchao_quantize_(m, weights.config)
                else:
                    _quantize_submodule(
                        model,
                        name,
                        m,
                        weights=weights,
                        activations=activations,
                        optimizer=optimizer,
                    )
                    # quanto's qbytes_mm only takes 2D/3D activations; video
                    # patch embeds feed their linears >3D tensors (convrot/
                    # torchao reshape internally). Flatten around the QLinear.
                    replaced = model.get_submodule(name)
                    if replaced.__class__.__name__ == "QLinear" and not getattr(
                        replaced, "_aitk_ndim_wrapped", False
                    ):
                        _wrap_qlinear_ndim(replaced)
            finally:
                if orig_device is not None and not keep_on_quantize_device:
                    # quanto replaces the module in its parent, so re-fetch by name
                    model.get_submodule(name).to(orig_device)
        except Exception as e:
            print(f"Failed to quantize {name}: {e}")
            # raise e


def _has_quantizable_linear(module: torch.nn.Module, weights, exclude=None) -> bool:
    """Whether quantizing ``module`` with ``weights`` would change anything.

    False when every non-excluded linear is already quantized with the same
    ostris qtype (or cannot be quantized at all) — the pre-quantized-checkpoint
    case, where the whole block can be skipped without the device round-trip.
    ``exclude`` patterns are matched against module names relative to
    ``module`` (use leading wildcards for patterns aimed at inner layers)."""
    if not isinstance(weights, ostristype):
        return True
    for name, m in module.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        if isinstance(m, OstrisLinear):
            if getattr(
                m.ostris_quantizer, "qtype", None
            ) != weights.quantizer.qtype and weights.quantizer.can_quantize(m):
                return True
            continue
        if m.__class__.__name__ in Q_MODULES:
            continue
        if weights.quantizer.can_quantize(m):
            return True
    return False


@torch.no_grad()
def dequantize_ostris_to_linear(module: torch.nn.Module) -> int:
    """Replace every OstrisLinear with a plain nn.Linear holding the full-
    precision weight (activation-side transforms folded), in place, layer by
    layer — the full-precision transient never exceeds one layer. Used when a
    pre-quantized checkpoint is loaded but a DIFFERENT quantization (or none,
    e.g. full finetuning) was requested. Returns the number of layers
    restored."""
    replaced = 0
    for parent in module.modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, OstrisLinear):
                continue
            weight = child.ostris_quantizer.dequantize_folded(child).to(
                child.ostris_orig_dtype
            )
            new = torch.nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device="meta",
                dtype=weight.dtype,
            )
            new.weight = torch.nn.Parameter(weight)
            if child.bias is not None:
                new.bias = torch.nn.Parameter(
                    child.bias.data.to(weight.device, weight.dtype)
                )
            setattr(parent, child_name, new)
            replaced += 1
    return replaced


@torch.no_grad()
def quantize_module(
    module: torch.nn.Module,
    qtype: str,
    device=None,
    dtype: torch.dtype = torch.bfloat16,
    block_names: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    quantize_kwargs: Optional[dict] = None,
    status_fn=print_acc,
    keep_on_device: bool = False,
):
    """Module-centric quantization: block-streamed (each repeated block moves
    to ``device`` for the math) with a whole-module pass for the extras. This
    is the core the per-model loaders call; holders' quantize_model wraps it
    with model_config plumbing.

    ``keep_on_device``: when the model's final home IS ``device`` (no layer
    offloading / low_vram), quantized blocks stay there instead of round-
    tripping back to cpu — the weights then cross the bus exactly once
    (mmap/page-cache -> gpu) instead of three times (up, back into a fresh
    host copy, up again at final placement), and the model-sized host RAM
    spike of that intermediate copy never happens. The extras pass runs on
    the gpu too. With it off (offload paths), blocks return to cpu as before
    and the extras quantize layer-by-layer on ``device`` via quantize_device
    rather than burning every cpu core."""
    from toolkit.dequantize import patch_dequantization_on_save

    patch_dequantization_on_save(module)
    quantization_type = get_qtype(qtype)
    exclude = list(exclude or [])
    quantize_kwargs = quantize_kwargs or {}
    keep_on_device = keep_on_device and device is not None

    all_blocks: List[torch.nn.Module] = []
    for name in block_names or []:
        # name may be a dotted path for models that nest their blocks
        block_list = module
        for part in name.split("."):
            block_list = getattr(block_list, part, None)
            if block_list is None:
                break
        if block_list is not None:
            all_blocks += list(block_list)
    if all_blocks:
        status_fn(f" - quantizing {len(all_blocks)} blocks")
    already_quantized = 0
    debug_phases = os.environ.get("AITK_QUANT_DEBUG") == "1"
    t_check = t_h2d = t_quant = t_d2h = 0.0
    for block in tqdm(all_blocks):
        t = time.perf_counter()
        skip = not _has_quantizable_linear(block, quantization_type, exclude)
        t_check += time.perf_counter() - t
        if skip:
            # pre-quantized checkpoint with a matching qtype: nothing in this
            # block would change — skip the dtype cast entirely so the load
            # stays byte-identical (placement still honors keep_on_device)
            already_quantized += 1
            if keep_on_device:
                block.to(device)
            continue
        t = time.perf_counter()
        if device is not None:
            block.to(device, dtype=dtype, non_blocking=True)
        t_h2d += time.perf_counter() - t
        t = time.perf_counter()
        quantize(block, weights=quantization_type, exclude=exclude, **quantize_kwargs)
        freeze(block)
        t_quant += time.perf_counter() - t
        # NOT non_blocking: an async D2H allocates the cpu destination in pinned
        # memory, which the caching host allocator keeps forever — that silently
        # retained a model-sized chunk of host ram
        t = time.perf_counter()
        if device is not None and not keep_on_device:
            block.to("cpu")
        t_d2h += time.perf_counter() - t
    if debug_phases and all_blocks:
        status_fn(
            f" - [debug] check {t_check:.1f}s h2d {t_h2d:.1f}s "
            f"quant {t_quant:.1f}s d2h {t_d2h:.1f}s"
        )
    if already_quantized:
        status_fn(
            f" - {already_quantized} blocks already quantized with a matching qtype; left untouched"
        )

    status_fn(" - quantizing extras")
    if keep_on_device:
        # blocks already live on the gpu; quantize each remaining layer there
        # one at a time and leave it — the transient is one bf16 layer, never
        # the whole unquantized remainder at once (that transient bump could
        # exceed the model's final footprint by GBs on big-extras models)
        quantize(
            module,
            weights=quantization_type,
            exclude=exclude,
            quantize_device=device,
            keep_on_quantize_device=True,
            **quantize_kwargs,
        )
        # non-quantized leftovers (norms, embeddings, buffers) follow — final
        # residency, not a transient
        module.to(device)
    else:
        # cpu-resident model: quantize each extra layer with a gpu round-trip
        quantize(
            module,
            weights=quantization_type,
            exclude=exclude,
            quantize_device=device,
            **quantize_kwargs,
        )
    freeze(module)
    return module


def quantize_model(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
):
    from toolkit.dequantize import patch_dequantization_on_save

    if not hasattr(base_model, "get_transformer_block_names"):
        raise ValueError(
            "The model to quantize must have a method `get_transformer_block_names`."
        )

    # patch the state dict method
    patch_dequantization_on_save(model_to_quantize)

    # sensitive modules to keep in full precision (fnmatch patterns)
    exclude_modules = base_model.get_quantization_exclude_modules() or []

    mc = base_model.model_config
    device = base_model.device_torch
    keep_on_device = (
        not mc.low_vram
        and not (mc.layer_offloading and mc.layer_offloading_transformer_percent > 0)
        and torch.device(device).type != "cpu"
    )

    if mc.accuracy_recovery_adapter is not None:
        attach_ara_and_quantize(
            base_model,
            model_to_quantize,
            ara_path=mc.accuracy_recovery_adapter,
            exclude=exclude_modules,
            device=device,
            keep_on_device=keep_on_device,
        )
    else:
        quantize_module(
            model_to_quantize,
            mc.qtype,
            device=device,
            dtype=base_model.torch_dtype,
            block_names=base_model.get_transformer_block_names(),
            exclude=exclude_modules,
            quantize_kwargs=mc.quantize_kwargs,
            status_fn=base_model.print_and_status_update,
            keep_on_device=keep_on_device,
        )


@torch.no_grad()
def attach_ara_and_quantize(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
    ara_path: str,
    exclude: Optional[List[str]] = None,
    device=None,
    keep_on_device: bool = False,
):
    """Load an accuracy recovery adapter as a live network on the module and
    quantize around it (adapter-hijacked linears at the configured qtype,
    everything else uint8). The network lands on
    base_model.accuracy_recovery_adapter.

    ``device``: quantize each hijacked linear there (gpu kernels) instead of
    wherever it happens to live (historically the cpu — slow). With
    ``keep_on_device`` the quantized linears stay there (final home is that
    gpu); otherwise each returns to the device it came from."""
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork

    exclude_modules = list(exclude or [])
    load_lora_path = ara_path

    if not os.path.exists(load_lora_path):
        # not local file, grab from the hub

        path_split = load_lora_path.split("/")
        if len(path_split) > 3:
            raise ValueError(
                "The accuracy recovery adapter path must be a local path or for a hf repo, 'username/repo_name/filename.safetensors'."
            )
        repo_id = f"{path_split[0]}/{path_split[1]}"
        print_acc(f"Grabbing lora from the hub: {load_lora_path}")
        new_lora_path = hf_hub_download(
            repo_id,
            filename=path_split[-1],
        )
        # replace the path
        load_lora_path = new_lora_path

    # build the lora config based on the lora weights
    lora_state_dict = load_file(load_lora_path)
        
    if hasattr(base_model, "convert_lora_weights_before_load"):
        lora_state_dict = base_model.convert_lora_weights_before_load(lora_state_dict)
        
    network_config = {
        "type": "lora",
        "network_kwargs": {"only_if_contains": []},
        "transformer_only": False,
    }
    first_key = list(lora_state_dict.keys())[0]
    first_weight = lora_state_dict[first_key]
    # if it starts with lycoris and includes lokr
    if first_key.startswith("lycoris") and any(
        "lokr" in key for key in lora_state_dict.keys()
    ):
        network_config["type"] = "lokr"
        
    network_kwargs = {}

    # find firse loraA weight
    if network_config["type"] == "lora":
        linear_dim = None
        for key, value in lora_state_dict.items():
            if "lora_A" in key:
                linear_dim = int(value.shape[0])
                break
        linear_alpha = linear_dim
        network_config["linear"] = linear_dim
        network_config["linear_alpha"] = linear_alpha

        # we build the keys to match every key
        only_if_contains = []
        for key in lora_state_dict.keys():
            contains_key = key.split(".lora_")[0]
            if contains_key not in only_if_contains:
                only_if_contains.append(contains_key)

        network_kwargs["only_if_contains"] = only_if_contains
    elif network_config["type"] == "lokr":
        # find the factor
        largest_factor = 0
        for key, value in lora_state_dict.items():
            if "lokr_w1" in key:
                factor = int(value.shape[0])
                if factor > largest_factor:
                    largest_factor = factor
        network_config["lokr_full_rank"] = True
        network_config["lokr_factor"] = largest_factor

        only_if_contains = []
        for key in lora_state_dict.keys():
            if "lokr_w1" in key:
                contains_key = key.split(".lokr_w1")[0]
                contains_key = contains_key.replace("lycoris_", "")
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)
        network_kwargs["only_if_contains"] = only_if_contains
        
    if hasattr(base_model, 'target_lora_modules'):
        network_kwargs['target_lin_modules'] = base_model.target_lora_modules

    # todo auto grab these
    # get dim and scale
    network_config = NetworkConfig(**network_config)

    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model_to_quantize,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        # conv_lora_dim=self.network_config.conv,
        # conv_alpha=self.network_config.conv_alpha,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=base_model.is_transformer,
        base_model=base_model,
        is_ara=True,
        **network_kwargs
    )
    network.apply_to(
        None, model_to_quantize, apply_text_encoder=False, apply_unet=True
    )
    network.force_to(base_model.device_torch, dtype=base_model.torch_dtype)
    network._update_torch_multiplier()
    network.load_weights(lora_state_dict)
    network.eval()
    network.is_active = True
    network.can_merge_in = False
    base_model.accuracy_recovery_adapter = network

    # quantize it
    keep_on_device = keep_on_device and device is not None
    lora_exclude_modules = []
    quantization_type = get_qtype(base_model.model_config.qtype)
    for lora_module in tqdm(network.unet_loras, desc="Attaching quantization"):
        # the lora has already hijacked the original module
        orig_module = lora_module.org_module[0]
        orig_device = None
        if device is not None:
            param = next(orig_module.parameters(), None)
            orig_device = param.device if param is not None else None
            orig_module.to(device, dtype=base_model.torch_dtype)
        else:
            orig_module.to(base_model.torch_dtype)
        # make the params not require gradients
        for param in orig_module.parameters():
            param.requires_grad = False
        quantize(orig_module, weights=quantization_type)
        freeze(orig_module)
        module_name = lora_module.lora_name.replace('$$', '.').replace('transformer.', '')
        lora_exclude_modules.append(module_name)
        if not keep_on_device and orig_device is not None:
            orig_module.to(orig_device)
        elif base_model.model_config.low_vram and device is None:
            # legacy behavior when no quantize device was given
            orig_module.to("cpu")
    # quantize additional layers
    print_acc(" - quantizing additional layers")
    quantization_type = get_qtype('uint8')
    quantize(
        model_to_quantize,
        weights=quantization_type,
        exclude=lora_exclude_modules + exclude_modules,
        quantize_device=device,
        keep_on_quantize_device=keep_on_device,
    )
    if keep_on_device:
        # non-quantized leftovers follow — final residency, not a transient
        model_to_quantize.to(device)


