"""Inference engine end-to-end test (extensions_built_in/inference_engine).

Starts the engine in-process on an ephemeral loopback port, sends real
generate requests over HTTP, decodes the streamed frames and asserts on
them. Model/sample settings come from testing/test_model_loading.py's
MODEL_TESTS (tiny sizes: this proves the path, not quality).

Usage:
  python testing/test_inference_engine.py --arch zimage          # one arch
  python testing/test_inference_engine.py --swap zimage,wan21    # sequential
                                                                 # model switches
  python testing/test_inference_engine.py --arch zimage --cancel # cancel mid-run
  python testing/test_inference_engine.py --all                  # every arch,
                                                                 # subprocess each
  --allow-download   permit hub downloads (default HF_HUB_OFFLINE=1 -> SKIP)
  --device cuda:0
"""

import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import time

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLKIT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(TOOLKIT_ROOT, ".env"))

OUTPUT_ROOT = os.path.join(TOOLKIT_ROOT, "testing", ".engine_test_outputs")

from testing.test_model_loading import MODEL_TESTS, SKIP_MARKERS, classify_error  # noqa: E402


class Client:
    def __init__(self, host, port, token):
        self.host, self.port, self.token = host, port, token

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=3600)

    def _headers(self):
        return {"Content-Type": "application/json", "X-Engine-Token": self.token}

    def get(self, path):
        c = self._conn()
        c.request("GET", path, headers=self._headers())
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def post_json(self, path, body=None):
        c = self._conn()
        c.request("POST", path, body=json.dumps(body or {}), headers=self._headers())
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def upload(self, path, name):
        c = self._conn()
        with open(path, "rb") as f:
            data = f.read()
        c.request("POST", f"/assets?name={name}", body=data, headers={"X-Engine-Token": self.token})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def generate_stream(self, body):
        """Yields (header, payload) frames as they arrive."""
        from extensions_built_in.inference_engine.protocol import FrameReader

        c = self._conn()
        c.request("POST", "/generate", body=json.dumps(body), headers=self._headers())
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"/generate -> {r.status}: {r.read()[:400]}")

        def chunks():
            while True:
                data = r.read1(65536)  # read1 returns the available chunk instead of blocking for 64KB
                if not data:
                    return
                yield data

        reader = FrameReader(chunks())
        for frame in reader:
            yield frame
        c.close()


def start_engine(device: str, out_root: str):
    from extensions_built_in.inference_engine.engine import Engine
    from extensions_built_in.inference_engine.server import create_app, serve_in_thread

    import threading

    engine = Engine(device=device, output_folder=os.path.join(out_root, "outputs"), assets_folder=os.path.join(out_root, "assets"))
    token = "test-token"
    server, thread, host, port = serve_in_thread(create_app(engine, token))
    loop = threading.Thread(target=engine.run_forever, daemon=True, name="engine-loop")
    loop.start()
    return engine, server, Client(host, port, token)


def stop_engine(engine, server):
    engine.stop()
    server.should_exit = True
    engine.unload()


QUANTIZE_ALL = False  # --quantize: force convrot8 on transformer + TE (small GPUs)


def request_body(arch: str, entry: dict, client: Client, out_dir: str, tiny=True, stream=None):
    model = dict(entry["model"], arch=arch)
    if QUANTIZE_ALL:
        model["quantize"] = True
        model["quantize_te"] = True
    if model.get("quantize"):
        model.setdefault("qtype", "convrot8")
    if model.get("quantize_te"):
        model.setdefault("qtype_te", "convrot8")
    sample = dict(entry["sample"])
    sample["prompt"] = "a photo of a cat sitting on a wooden table"
    if tiny and not entry.get("size_locked"):
        # 384 divides every bucket size (16/32/64); video runs also keep the
        # frame count small so the swap tests fit next to a resident TE
        cap = 384 if tiny == "tiny" else 512
        sample["width"] = min(sample["width"], cap)
        sample["height"] = min(sample["height"], cap)
        if tiny == "tiny" and "num_frames" in sample:
            sample["num_frames"] = min(sample["num_frames"], 9)
    if entry.get("needs_control_image"):
        from PIL import Image

        ctrl_path = os.path.join(out_dir, "ctrl.png")
        Image.new("RGB", (sample["width"], sample["height"]), (128, 128, 128)).save(ctrl_path)
        status, asset = client.upload(ctrl_path, "ctrl.png")
        assert status == 200, asset
        sample["ctrl_img"] = asset["path"]
    return {"model": model, "sample": sample, "stream": stream or {"latents": "raw", "every_n_steps": 1, "max_frames": 2}}


def run_request(client: Client, body: dict, cancel_after: int = 0) -> dict:
    """Consume one streamed generation; returns a summary of the frames."""
    import numpy as np

    from extensions_built_in.inference_engine.protocol import payload_to_array

    seen = {"start": 0, "status": 0, "progress": 0, "latent": 0, "result": 0, "error": 0, "end": 0}
    summary = {"frames": seen, "results": [], "latent_shapes": [], "error": None, "cancelled": False, "request_id": None, "preview": None}
    for header, payload in client.generate_stream(body):
        t = header["type"]
        seen[t] = seen.get(t, 0) + 1
        summary["request_id"] = header.get("request_id")
        if t == "start":
            summary["preview"] = header.get("preview")
            summary["modality"] = header.get("modality")
        elif t == "status":
            print(f"    [status] {header.get('message')}")
        elif t == "progress":
            print(f"    [progress] {header['step']}/{header['total']} {header['elapsed']}s", flush=True)
            if cancel_after and header["step"] >= cancel_after:
                st, res = client.post_json(f"/cancel/{summary['request_id']}")
                assert st == 200 and res["ok"], res
                cancel_after = 0
        elif t == "latent":
            arr = payload_to_array(header, payload)
            assert np.isfinite(arr.astype(np.float32)).all(), "non-finite latent"
            summary["latent_shapes"].append((header["layout"], list(arr.shape)))
        elif t == "result":
            summary["results"].append(header)
            print(f"    [result] {header['kind']} {header['path']} ({header['bytes']} bytes)")
        elif t == "error":
            summary["error"] = header.get("message")
            summary["cancelled"] = bool(header.get("cancelled"))
            if header.get("traceback"):
                print(header["traceback"])
        elif t == "end":
            summary["status"] = header.get("status")
    return summary


def assert_success(summary: dict, modality: str):
    assert summary["frames"]["start"] == 1, summary
    assert summary["frames"]["end"] == 1, summary
    assert summary["error"] is None, summary["error"]
    assert summary["frames"]["progress"] >= 1, "no progress frames"
    assert summary["frames"]["latent"] >= 1, "no latent frames"
    assert summary["results"], "no result frames"
    for r in summary["results"]:
        assert os.path.isfile(r["path"]) and r["bytes"] > 1024, r
    kinds = {r["kind"] for r in summary["results"]}
    assert modality in kinds or (modality == "video" and "image" in kinds), (modality, kinds)


def run_one(arch: str, device: str, cancel: bool = False) -> dict:
    from toolkit.models.registry import get_arch_entry

    entry = MODEL_TESTS[arch]
    out_root = os.path.join(OUTPUT_ROOT, arch.replace("/", "_"))
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)
    engine, server, client = start_engine(device, out_root)
    try:
        st, health = client.get("/health")
        assert st == 200 and health["ok"], health
        st, models = client.get("/models")
        assert st == 200 and any(m["arch"] == arch for m in models["archs"]), f"{arch} missing from /models"
        modality = get_arch_entry(arch)["modality"]
        body = request_body(arch, entry, client, out_root)

        t0 = time.perf_counter()
        if cancel:
            s = run_request(client, body, cancel_after=2)
            assert s["cancelled"], f"expected a cancelled stream, got {s}"
            assert s["frames"]["end"] == 1
            print("  cancel: ok, re-running the same request")
        s = run_request(client, body)
        gen_seconds = time.perf_counter() - t0
        assert_success(s, modality)
        st, health = client.get("/health")
        assert health["active"]["arch"] == arch, health
        return {
            "arch": arch,
            "status": "PASS",
            "seconds": round(gen_seconds, 2),
            "frames": s["frames"],
            "latent_shapes": s["latent_shapes"][:2],
            "results": [r["path"] for r in s["results"]],
            "preview_format": (s["preview"] or {}).get("format"),
            "stats": engine.stats,
        }
    finally:
        stop_engine(engine, server)


def run_swap(archs, device: str) -> dict:
    out_root = os.path.join(OUTPUT_ROOT, "swap_" + "_".join(a.replace("/", "_") for a in archs))
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(out_root, exist_ok=True)
    engine, server, client = start_engine(device, out_root)
    from toolkit.models.registry import get_arch_entry

    timeline = []
    try:
        for i, arch in enumerate(archs):
            body = request_body(arch, MODEL_TESTS[arch], client, out_root, tiny="tiny")
            t0 = time.perf_counter()
            s = run_request(client, body)
            assert_success(s, get_arch_entry(arch)["modality"])
            pool = engine.pool.stats()
            resident = [(e["cls"], round(e["bytes"] / 1e9, 2), e["device"]) for e in pool["resident"]]
            entry = {
                "arch": arch,
                "seconds": round(time.perf_counter() - t0, 2),
                "holder_loads": engine.stats["holder_loads"],
                "oom_retries": engine.stats["oom_retries"],
                "pool_hits": pool["hits"],
                "pool_misses": pool["misses"],
                "pool_evictions": pool["evictions"],
                "resident_gb": round(pool["resident_bytes"] / 1e9, 2),
                "resident": resident,
            }
            timeline.append(entry)
            print(f"  {arch}: {json.dumps(entry)}")
            # after a request only this holder's components may stay resident
            assert pool["untouched"] == 0, f"stale components left resident: {pool}"
            if i > 0 and archs[i - 1] == arch:
                assert entry["holder_loads"] == timeline[i - 1]["holder_loads"], "same model should reuse the holder"
        if len(set(archs)) > 1:
            assert engine.pool.hits > 0, "expected shared components between the swapped archs"
        return {"arch": ",".join(archs), "status": "PASS", "timeline": timeline}
    finally:
        stop_engine(engine, server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--swap", type=str, default=None, help="comma-separated archs to run back to back")
    parser.add_argument("--cancel", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--json-result", type=str, default=None)
    parser.add_argument("--quantize", action="store_true", help="force convrot8 quantization of transformer + TE")
    args = parser.parse_args()
    global QUANTIZE_ALL
    QUANTIZE_ALL = args.quantize

    if args.list:
        for arch in MODEL_TESTS:
            print(arch)
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _record(result):
        if args.json_result:
            with open(args.json_result, "w") as f:
                json.dump(result, f)
        print(f"[{result['status']}] {result['arch']}" + (f" - {result.get('error', '')}" if result["status"] != "PASS" else f" {json.dumps({k: v for k, v in result.items() if k not in ('arch', 'status')})}"))
        if result["status"] == "FAIL":
            sys.exit(1)

    if args.swap:
        archs = [a.strip() for a in args.swap.split(",") if a.strip()]
        try:
            result = run_swap(archs, args.device)
        except BaseException as err:
            import traceback

            traceback.print_exc()
            result = {"arch": args.swap, "status": classify_error(err), "error": f"{type(err).__name__}: {err}"}
        _record(result)
        return

    if args.arch:
        if args.arch not in MODEL_TESTS:
            raise SystemExit(f"arch {args.arch!r} is not registered; --list shows options")
        try:
            result = run_one(args.arch, args.device, cancel=args.cancel)
        except BaseException as err:
            status = classify_error(err)
            if status == "FAIL":
                import traceback

                traceback.print_exc()
            result = {"arch": args.arch, "status": status, "error": f"{type(err).__name__}: {err}"}
        _record(result)
        return

    if not args.all:
        parser.print_help()
        return

    results = []
    total = len(MODEL_TESTS)
    t0 = time.perf_counter()
    for i, arch in enumerate(MODEL_TESTS, start=1):
        print(f"\n===== {arch} ({i}/{total}) =====", flush=True)
        result_path = os.path.join(OUTPUT_ROOT, f".{arch.replace('/', '_')}.result.json")
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--arch", arch, "--device", args.device, "--json-result", result_path]
        if args.allow_download:
            cmd.append("--allow-download")
        if args.quantize:
            cmd.append("--quantize")
        proc = subprocess.run(cmd, cwd=TOOLKIT_ROOT)
        if os.path.exists(result_path):
            with open(result_path) as f:
                results.append(json.load(f))
            os.remove(result_path)
        else:
            results.append({"arch": arch, "status": "FAIL", "error": f"subprocess died (exit {proc.returncode})"})
        tally = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for r in results:
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        elapsed = time.perf_counter() - t0
        print(f">>> progress: {i}/{total} - {tally['PASS']} pass, {tally['FAIL']} fail, {tally['SKIP']} skip - elapsed {elapsed / 60:.0f}m", flush=True)

    print("\n===== summary =====")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        line = f"[{r['status']}] {r['arch']}"
        if r["status"] != "PASS":
            line += f" - {r.get('error', '')[:160]}"
        print(line)
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    with open(os.path.join(OUTPUT_ROOT, "report.json"), "w") as f:
        json.dump(results, f, indent=2)
    if counts["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
