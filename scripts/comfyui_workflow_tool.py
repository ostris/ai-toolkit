#!/usr/bin/env python3
"""Validate and convert raw ComfyUI API workflow JSONs for ai-toolkit remote previews.

Never modifies the input file unless --output points to the same path.
Works standalone with system python3 (stdlib + requests + Pillow).

Examples:
  # Inspect & validate a workflow locally (and optionally against Machine B):
  python3 scripts/comfyui_workflow_tool.py validate user_upload/lora_tester_qi21_4steps.json --server http://pc-b:8188

  # Convert a raw workflow (auto-rename prompt -> AITK_PROMPT, insert AITK_LORA, rewrite output -> SaveImageWebsocket):
  python3 scripts/comfyui_workflow_tool.py convert user_upload/lora_tester_qi21_4steps.json \
      -o work/lora_tester_qi21_4steps_aitk.json --server http://pc-b:8188

  # Convert a Turbo-LoRA workflow (keep Turbo LoRA intact and stack AITK_LORA after it):
  python3 scripts/comfyui_workflow_tool.py convert user_upload/lora_tester_qi21_8steps.json \
      -o work/lora_tester_qi21_8steps_aitk.json --insert-lora --server http://pc-b:8188
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load_sampler_module():
    try:
        from toolkit import remote_comfyui_sampler as mod
        return mod
    except ImportError:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mod_path = os.path.join(repo_root, "toolkit", "remote_comfyui_sampler.py")
        spec = importlib.util.spec_from_file_location("remote_comfyui_sampler", mod_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
        return mod


_MODEL_LOADER_CLASSES = (
    "UNETLoader",
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "DiffusionModelLoader",
    "UnetLoaderGGUF",
    "UNETLoaderGGUF",
)
_LORA_LOADER_CLASSES = (
    "LoraLoaderModelOnly",
    "LoraLoader",
)
_PROMPT_NODE_CLASSES = (
    "TextEncodeQwenImage21",
    "CLIPTextEncode",
    "MiniMaxH3ImageConditioning",
    "CLIPTextEncodeFlux",
    "CLIPTextEncodeSDXL",
)
_SAVE_NODE_CLASSES = ("SaveImageWebsocket", "PreviewImage", "SaveImage")


def _node_title(node: Dict[str, Any]) -> str:
    meta = node.get("_meta")
    if isinstance(meta, dict):
        return str(meta.get("title") or "").strip()
    return ""


def _set_node_title(node: Dict[str, Any], title: str) -> None:
    meta = node.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
        node["_meta"] = meta
    meta["title"] = title


def _is_link(val: Any) -> bool:
    return isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], int)


def _next_node_id(workflow: Dict[str, Any]) -> str:
    numeric = []
    for k in workflow.keys():
        try:
            numeric.append(int(k))
        except (TypeError, ValueError):
            pass
    cand = (max(numeric) + 1) if numeric else 1
    while str(cand) in workflow:
        cand += 1
    return str(cand)


def _load_api_workflow(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path!r} is not a non-empty JSON object.")
    if "nodes" in data and isinstance(data["nodes"], list):
        raise ValueError(
            f"{path!r} is a ComfyUI UI graph JSON, not API format. "
            "Export it via Workflow -> Export (API) in ComfyUI."
        )
    return data


def _detect_prompt_node(workflow: Dict[str, Any], prompt_title: str) -> Tuple[Optional[str], List[str]]:
    """Return (chosen_node_id, all_candidate_ids)."""
    # 1. Already titled
    for nid, node in workflow.items():
        if isinstance(node, dict) and _node_title(node).lower() == prompt_title.strip().lower():
            return str(nid), [str(nid)]

    candidates: List[str] = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        inputs = node.get("inputs") or {}
        if cls in _PROMPT_NODE_CLASSES or (
            isinstance(inputs, dict)
            and any(k in inputs and isinstance(inputs[k], str) for k in ("prompt", "text"))
            and "encode" in cls.lower()
        ):
            candidates.append(str(nid))

    if len(candidates) == 1:
        return candidates[0], candidates

    # Disambiguate multiple CLIPTextEncode nodes by finding which one feeds a `positive` input.
    if len(candidates) > 1:
        positive_sources: List[str] = []
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            pos = (node.get("inputs") or {}).get("positive")
            if _is_link(pos) and pos[0] in candidates and pos[0] not in positive_sources:
                positive_sources.append(pos[0])
        if len(positive_sources) == 1:
            return positive_sources[0], candidates

    return None, candidates


def _find_lora_nodes(workflow: Dict[str, Any]) -> List[str]:
    out = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        inputs = node.get("inputs") or {}
        if cls in _LORA_LOADER_CLASSES or (isinstance(inputs, dict) and "lora_name" in inputs):
            out.append(str(nid))
    return out


def _find_model_loaders(workflow: Dict[str, Any]) -> List[str]:
    out = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        if cls in _MODEL_LOADER_CLASSES:
            out.append(str(nid))
    return out


def _find_tail_model_source(workflow: Dict[str, Any]) -> Tuple[Optional[str], int]:
    """Find the best (node_id, output_slot) after which to splice a new LoraLoaderModelOnly.

    Prefers the end of an existing LoRA chain (so Turbo LoRA stays upstream),
    otherwise the single diffusion model loader (e.g. UNETLoader / CheckpointLoaderSimple).
    """
    lora_nodes = _find_lora_nodes(workflow)
    if lora_nodes:
        # Pick a LoRA node whose model output is NOT consumed by another LoRA node.
        upstream_of_lora = set()
        for nid in lora_nodes:
            m_in = (workflow[nid].get("inputs") or {}).get("model")
            if _is_link(m_in):
                upstream_of_lora.add(m_in[0])
        tails = [nid for nid in lora_nodes if nid not in upstream_of_lora]
        if len(tails) == 1:
            return tails[0], 0
        if len(lora_nodes) == 1:
            return lora_nodes[0], 0

    loaders = _find_model_loaders(workflow)
    if len(loaders) == 1:
        return loaders[0], 0
    return None, 0


def convert_workflow(
    workflow: Dict[str, Any],
    *,
    lora_title: str = "AITK_LORA",
    prompt_title: str = "AITK_PROMPT",
    prompt_node_id: Optional[str] = None,
    lora_node_id: Optional[str] = None,
    insert_lora: bool = False,
    insert_after: Optional[str] = None,
    lora_strength: float = 1.0,
    placeholder_lora: str = "placeholder.safetensors",
    to_websocket: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """Convert a raw ComfyUI API workflow into an ai-toolkit remote-preview workflow.

    Returns (converted_workflow, list_of_human_readable_changes).
    """
    mod = _load_sampler_module()
    wf = copy.deepcopy(workflow)
    changes: List[str] = []

    # 1. Prompt node -> prompt_title
    if prompt_node_id is not None:
        pid = str(prompt_node_id)
        if pid not in wf:
            raise ValueError(f"--prompt-node {pid!r} not found in workflow.")
    else:
        pid, candidates = _detect_prompt_node(wf, prompt_title)
        if pid is None:
            raise ValueError(
                f"Could not uniquely detect prompt node (candidates={candidates}). "
                "Pass --prompt-node <id> explicitly."
            )

    prev_ptitle = _node_title(wf[pid])
    if prev_ptitle.lower() != prompt_title.strip().lower():
        _set_node_title(wf[pid], prompt_title)
        changes.append(
            f"Renamed prompt node {pid} ({wf[pid].get('class_type')}) title: {prev_ptitle!r} -> {prompt_title!r}"
        )
    else:
        changes.append(f"Prompt node {pid} ({wf[pid].get('class_type')}) already titled {prompt_title!r}")

    # 2. LoRA node -> lora_title (rename existing OR insert new LoraLoaderModelOnly)
    existing_bindings = mod.find_titled_node_bindings(wf, lora_node_title=lora_title, prompt_node_title=prompt_title)
    if lora_node_id is not None and insert_lora:
        raise ValueError("Specify either --lora-node or --insert-lora, not both.")

    if lora_node_id is not None:
        lid = str(lora_node_id)
        if lid not in wf:
            raise ValueError(f"--lora-node {lid!r} not found in workflow.")
        prev_ltitle = _node_title(wf[lid])
        _set_node_title(wf[lid], lora_title)
        changes.append(
            f"Renamed LoRA node {lid} ({wf[lid].get('class_type')}) title: {prev_ltitle!r} -> {lora_title!r}"
        )
    elif existing_bindings["lora"] and not insert_lora:
        desc = ", ".join(f"{nid}.{fld}" for nid, fld in existing_bindings["lora"])
        changes.append(f"LoRA node already titled {lora_title!r} ({desc})")
    else:
        lora_nodes = _find_lora_nodes(wf)
        if lora_nodes and not insert_lora and insert_after is None:
            if len(lora_nodes) == 1:
                lid = lora_nodes[0]
                lname = (wf[lid].get("inputs") or {}).get("lora_name", "")
                if "turbo" in str(lname).lower() or "lightning" in str(lname).lower() or "lcm" in str(lname).lower():
                    # Existing LoRA looks like a speed/turbo LoRA -> stack AITK_LORA after it!
                    insert_lora = True
                    insert_after = lid
                else:
                    prev_ltitle = _node_title(wf[lid])
                    _set_node_title(wf[lid], lora_title)
                    changes.append(
                        f"Renamed existing LoRA node {lid} ({wf[lid].get('class_type')}) title: "
                        f"{prev_ltitle!r} -> {lora_title!r}"
                    )
            else:
                raise ValueError(
                    f"Multiple LoRA nodes found ({lora_nodes}). Pass --lora-node <id> to rename one "
                    "or --insert-lora [--insert-after <id>] to stack a new training LoRA node."
                )
        else:
            insert_lora = True

        if insert_lora:
            if insert_after is not None:
                upstream_id, upstream_slot = str(insert_after), 0
                if upstream_id not in wf:
                    raise ValueError(f"--insert-after {upstream_id!r} not found in workflow.")
            else:
                upstream_id, upstream_slot = _find_tail_model_source(wf)
                if upstream_id is None:
                    raise ValueError(
                        "Could not uniquely detect upstream model loader to insert LoraLoaderModelOnly after. "
                        "Pass --insert-after <node_id> explicitly."
                    )

            new_id = _next_node_id(wf)
            rewired: List[str] = []
            for nid, node in wf.items():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs")
                if not isinstance(inputs, dict):
                    continue
                for field, val in list(inputs.items()):
                    if _is_link(val) and val[0] == upstream_id and val[1] == upstream_slot and field == "model":
                        inputs[field] = [new_id, 0]
                        rewired.append(f"{nid}.{field}")

            if not rewired:
                raise ValueError(
                    f"No downstream node consumes model output [{upstream_id!r}, {upstream_slot}] "
                    "to rewire through the new AITK_LORA node."
                )

            wf[new_id] = {
                "inputs": {
                    "lora_name": placeholder_lora,
                    "strength_model": float(lora_strength),
                    "model": [upstream_id, upstream_slot],
                },
                "class_type": "LoraLoaderModelOnly",
                "_meta": {"title": lora_title},
            }
            changes.append(
                f"Inserted node {new_id} (LoraLoaderModelOnly, title={lora_title!r}, strength={lora_strength:g}) "
                f"after node {upstream_id} ({wf[upstream_id].get('class_type')}); rewired {', '.join(rewired)}"
            )

    # 3. Output node -> SaveImageWebsocket
    if to_websocket:
        before_ws = {
            str(nid): node.get("class_type")
            for nid, node in wf.items()
            if isinstance(node, dict) and node.get("class_type") in _SAVE_NODE_CLASSES
        }
        wf, ws_nodes = mod.rewrite_save_nodes_to_websocket(wf, replace_save=True)
        if not ws_nodes:
            raise ValueError("Workflow has no SaveImageWebsocket, PreviewImage, or SaveImage output node.")
        for nid in sorted(ws_nodes):
            old_cls = before_ws.get(nid, "SaveImageWebsocket")
            if old_cls != "SaveImageWebsocket":
                changes.append(f"Converted output node {nid}: {old_cls} -> SaveImageWebsocket")
            else:
                changes.append(f"Output node {nid} already SaveImageWebsocket")

    return wf, changes


def inspect_and_validate(
    workflow_path: str,
    *,
    lora_title: str = "AITK_LORA",
    prompt_title: str = "AITK_PROMPT",
    server_url: Optional[str] = None,
    local_root: Optional[str] = None,
    save_root: Optional[str] = None,
    prefix: str = "",
) -> bool:
    mod = _load_sampler_module()
    wf = _load_api_workflow(workflow_path)
    print(f"Workflow: {workflow_path} ({len(wf)} nodes)")

    model_loaders = _find_model_loaders(wf)
    lora_nodes = _find_lora_nodes(wf)
    _, prompt_candidates = _detect_prompt_node(wf, prompt_title)
    output_nodes = [
        str(nid) for nid, n in wf.items() if isinstance(n, dict) and n.get("class_type") in _SAVE_NODE_CLASSES
    ]
    print(f"  Model loaders : {[(n, wf[n].get('class_type')) for n in model_loaders]}")
    print(f"  LoRA nodes    : {[(n, _node_title(wf[n]), (wf[n].get('inputs') or {}).get('lora_name')) for n in lora_nodes]}")
    print(f"  Prompt nodes  : {[(n, wf[n].get('class_type'), _node_title(wf[n])) for n in prompt_candidates]}")
    print(f"  Output nodes  : {[(n, wf[n].get('class_type')) for n in output_nodes]}")

    if server_url:
        lroot = os.path.abspath(local_root or ".")
        sroot = os.path.abspath(save_root or lroot)
        cfg = mod.RemoteComfyUIConfig(
            enabled=True,
            server_url=server_url,
            workflow_path=workflow_path,
            remote_lora_prefix=prefix,
            local_root=lroot,
            lora_node_title=lora_title,
            prompt_node_title=prompt_title,
            preflight_check="warn",
        )
        sampler = mod.RemoteComfyUISampler(
            config=cfg,
            training_folder=lroot,
            job_name=os.path.basename(sroot) or "validate",
            save_root=sroot,
            sample_prompts_count=1,
        )
        report = sampler.preflight_report
        sampler.stop()
        return bool(report and report["ok"])

    # Local-only 2-input + 1-output check
    bindings = mod.find_titled_node_bindings(wf, lora_node_title=lora_title, prompt_node_title=prompt_title)
    _, ws_nodes = mod.rewrite_save_nodes_to_websocket(wf, replace_save=True)
    ok = True
    if bindings["lora"]:
        print(f"  [PASS] lora_input_node: {bindings['lora']}")
    else:
        print(f"  [FAIL] lora_input_node: no node titled {lora_title!r} with lora_name input")
        ok = False
    if bindings["prompt"]:
        print(f"  [PASS] prompt_input_node: {bindings['prompt']}")
    else:
        print(f"  [FAIL] prompt_input_node: no node titled {prompt_title!r} with prompt/text input")
        ok = False
    if ws_nodes:
        print(f"  [PASS] ws_output_node: {sorted(ws_nodes)}")
    else:
        print("  [FAIL] ws_output_node: no SaveImageWebsocket / PreviewImage / SaveImage node")
        ok = False
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and convert ComfyUI API workflows for ai-toolkit remote ComfyUI previews."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Inspect & validate a ComfyUI API workflow JSON")
    p_val.add_argument("workflow", help="Path to ComfyUI API-format workflow JSON")
    p_val.add_argument("--lora-title", default="AITK_LORA")
    p_val.add_argument("--prompt-title", default="AITK_PROMPT")
    p_val.add_argument("--server", default=None, help="Optional Machine B ComfyUI URL for live preflight")
    p_val.add_argument("--local-root", default=None, help="Local output root on Machine A")
    p_val.add_argument("--save-root", default=None, help="Job save_root on Machine A")
    p_val.add_argument("--prefix", default="", help="remote_lora_prefix on Machine B")

    p_conv = sub.add_parser("convert", help="Convert a raw ComfyUI API workflow JSON for ai-toolkit")
    p_conv.add_argument("input", help="Path to raw ComfyUI API workflow JSON (never modified unless same as -o)")
    p_conv.add_argument("-o", "--output", required=True, help="Output path for converted workflow JSON")
    p_conv.add_argument("--lora-title", default="AITK_LORA")
    p_conv.add_argument("--prompt-title", default="AITK_PROMPT")
    p_conv.add_argument("--prompt-node", default=None, help="Explicit prompt node ID to title AITK_PROMPT")
    p_conv.add_argument("--lora-node", default=None, help="Explicit existing LoRA node ID to title AITK_LORA")
    p_conv.add_argument("--insert-lora", action="store_true", help="Insert a new LoraLoaderModelOnly titled AITK_LORA")
    p_conv.add_argument("--insert-after", default=None, help="Upstream model node ID to insert AITK_LORA after")
    p_conv.add_argument("--lora-strength", type=float, default=1.0, help="strength_model for newly inserted AITK_LORA")
    p_conv.add_argument("--keep-preview-node", action="store_true", help="Do not rewrite PreviewImage/SaveImage")
    p_conv.add_argument("--server", default=None, help="Optional Machine B ComfyUI URL to run live preflight on output")
    p_conv.add_argument("--local-root", default=None)
    p_conv.add_argument("--save-root", default=None)
    p_conv.add_argument("--prefix", default="")

    args = parser.parse_args(argv)

    if args.cmd == "validate":
        ok = inspect_and_validate(
            args.workflow,
            lora_title=args.lora_title,
            prompt_title=args.prompt_title,
            server_url=args.server,
            local_root=args.local_root,
            save_root=args.save_root,
            prefix=args.prefix,
        )
        return 0 if ok else 1

    if args.cmd == "convert":
        raw_wf = _load_api_workflow(args.input)
        converted, changes = convert_workflow(
            raw_wf,
            lora_title=args.lora_title,
            prompt_title=args.prompt_title,
            prompt_node_id=args.prompt_node,
            lora_node_id=args.lora_node,
            insert_lora=args.insert_lora,
            insert_after=args.insert_after,
            lora_strength=args.lora_strength,
            to_websocket=not args.keep_preview_node,
        )
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Converted {args.input} -> {args.output}:")
        for c in changes:
            print(f"  - {c}")
        ok = inspect_and_validate(
            args.output,
            lora_title=args.lora_title,
            prompt_title=args.prompt_title,
            server_url=args.server,
            local_root=args.local_root,
            save_root=args.save_root,
            prefix=args.prefix,
        )
        return 0 if ok else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
