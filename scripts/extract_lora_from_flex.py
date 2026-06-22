import os
from tqdm import tqdm
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Extract LoRA from Flex")
parser.add_argument("--base", type=str, default="ostris/Flex.1-alpha", help="Base model path")
parser.add_argument("--tuned", type=str, required=True, help="Tuned model path")
parser.add_argument("--output", type=str, required=True, help="Output path for lora")
parser.add_argument("--rank", type=int, default=32, help="LoRA rank for extraction")
parser.add_argument("--gpu", type=int, default=0, help="GPU to process extraction")
parser.add_argument("--full", action="store_true", help="Do a full transformer extraction, not just transformer blocks")

args = parser.parse_args()

if True:
    # set cuda environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from safetensors.torch import load_file, save_file
    from lycoris.utils import extract_linear, extract_conv, make_sparse
    from diffusers import FluxTransformer2DModel

base = args.base
tuned = args.tuned
output_path = args.output
dim = args.rank

os.makedirs(os.path.dirname(output_path), exist_ok=True)

state_dict_base = {}
state_dict_tuned = {}

output_dict = {}

@torch.no_grad()
def extract_diff(
    base_unet,
    db_unet,
    mode="fixed",
    linear_mode_param=0,
    conv_mode_param=0,
    extract_device="cpu",
    use_bias=False,
    sparsity=0.98,
    # small_conv=True,
    small_conv=False,
):
    UNET_TARGET_REPLACE_MODULE = [
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
        "LoRACompatibleLinear",
        "LoRACompatibleConv"
    ]
    LORA_PREFIX_UNET = "transformer"

    def make_state_dict(
        prefix,
        root_module: torch.nn.Module,
        target_module: torch.nn.Module,
        target_replace_modules,
    ):
        loras = {}
        temp = {}

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = module

        for name, module in tqdm(
            list((n, m) for n, m in target_module.named_modules() if n in temp)
        ):
            weights = temp[name]
            lora_name = prefix + "." + name
            # lora_name = lora_name.replace(".", "_")
            layer = module.__class__.__name__
            if 'transformer_blocks' not in lora_name and not args.full:
                continue

            if layer in {
                "Linear",
                "Conv2d",
                "LayerNorm",
                "GroupNorm",
                "GroupNorm32",
                "Embedding",
                "LoRACompatibleLinear",
                "LoRACompatibleConv"
            }:
                root_weight = module.weight
                try:
                    if torch.allclose(root_weight, weights.weight):
                        continue
                except:
                    continue
            else:
                continue
            module = module.to(extract_device, torch.float32)
            weights = weights.to(extract_device, torch.float32)

            if mode == "full":
                decompose_mode = "full"
            elif layer == "Linear":
                weight, decompose_mode = extract_linear(
                    (root_weight - weights.weight),
                    mode,
                    linear_mode_param,
                    device=extract_device,
                )
                if decompose_mode == "low rank":
                    extract_a, extract_b, diff = weight
            elif layer == "Conv2d":
                is_linear = root_weight.shape[2] == 1 and root_weight.shape[3] == 1
                weight, decompose_mode = extract_conv(
                    (root_weight - weights.weight),
                    mode,
                    linear_mode_param if is_linear else conv_mode_param,
                    device=extract_device,
                )
                if decompose_mode == "low rank":
                    extract_a, extract_b, diff = weight
                if small_conv and not is_linear and decompose_mode == "low rank":
                    dim = extract_a.size(0)
                    (extract_c, extract_a, _), _ = extract_conv(
                        extract_a.transpose(0, 1),
                        "fixed",
                        dim,
                        extract_device,
                        True,
                    )
                    extract_a = extract_a.transpose(0, 1)
                    extract_c = extract_c.transpose(0, 1)
                    loras[f"{lora_name}.lora_mid.weight"] = (
                        extract_c.detach().cpu().contiguous().half()
                    )
                    diff = (
                        (
                            root_weight
                            - torch.einsum(
                                "i j k l, j r, p i -> p r k l",
                                extract_c,
                                extract_a.flatten(1, -1),
                                extract_b.flatten(1, -1),
                            )
                        )
                        .detach()
                        .cpu()
                        .contiguous()
                    )
                    del extract_c
            else:
                module = module.to("cpu")
                weights = weights.to("cpu")
                continue

            if decompose_mode == "low rank":
                loras[f"{lora_name}.lora_A.weight"] = (
                    extract_a.detach().cpu().contiguous().half()
                )
                loras[f"{lora_name}.lora_B.weight"] = (
                    extract_b.detach().cpu().contiguous().half()
                )
                # loras[f"{lora_name}.alpha"] = torch.Tensor([extract_a.shape[0]]).half()
                if use_bias:
                    diff = diff.detach().cpu().reshape(extract_b.size(0), -1)
                    sparse_diff = make_sparse(diff, sparsity).to_sparse().coalesce()

                    indices = sparse_diff.indices().to(torch.int16)
                    values = sparse_diff.values().half()
                    loras[f"{lora_name}.bias_indices"] = indices
                    loras[f"{lora_name}.bias_values"] = values
                    loras[f"{lora_name}.bias_size"] = torch.tensor(diff.shape).to(
                        torch.int16
                    )
                del extract_a, extract_b, diff
            elif decompose_mode == "full":
                if "Norm" in layer:
                    w_key = "w_norm"
                    b_key = "b_norm"
                else:
                    w_key = "diff"
                    b_key = "diff_b"
                weight_diff = module.weight - weights.weight
                loras[f"{lora_name}.{w_key}"] = (
                    weight_diff.detach().cpu().contiguous().half()
                )
                if getattr(weights, "bias", None) is not None:
                    bias_diff = module.bias - weights.bias
                    loras[f"{lora_name}.{b_key}"] = (
                        bias_diff.detach().cpu().contiguous().half()
                    )
            else:
                raise NotImplementedError
            module = module.to("cpu", torch.bfloat16)
            weights = weights.to("cpu", torch.bfloat16)
        return loras

    all_loras = {}

    all_loras |= make_state_dict(
        LORA_PREFIX_UNET,
        base_unet,
        db_unet,
        UNET_TARGET_REPLACE_MODULE,
    )
    del base_unet, db_unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_lora_name = set()
    for k in all_loras:
        lora_name, weight = k.rsplit(".", 1)
        all_lora_name.add(lora_name)
    print(len(all_lora_name))
    return all_loras


# find all the .safetensors files and load them
print("Loading Base")
base_model = FluxTransformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=torch.bfloat16)

print("Loading Tuned")
tuned_model = FluxTransformer2DModel.from_pretrained(tuned, subfolder="transformer", torch_dtype=torch.bfloat16)

output_dict = extract_diff(
    base_model,
    tuned_model,
    mode="fixed",
    linear_mode_param=dim,
    conv_mode_param=dim,
    extract_device="cuda",
    use_bias=False,
    sparsity=0.98,
    small_conv=False,
)

meta = OrderedDict()
meta['format'] = 'pt'

save_file(output_dict, output_path, metadata=meta)

print("Done")
