"""python -m toolkit.audio.melbandroformer song.flac [more files or dirs] [--out_dir DIR] [--format flac]
Writes <name>_vocals.<format> and <name>_instrumental.<format> next to each input (or in --out_dir).
Decode and encode run in worker threads so the GPU is never waiting on file IO."""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))  # MODELS_PATH, before toolkit.paths

import torch

from . import load_melbandroformer, separate, DEFAULT_WEIGHTS

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aif", ".aiff"}


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in sorted(names) if os.path.splitext(n)[1].lower() in AUDIO_EXTS]
        else:
            files.append(p)
    return files


def output_paths(path, out_dir, fmt):
    out_dir = out_dir or os.path.dirname(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return {stem: os.path.join(out_dir, f"{name}_{stem}.{fmt}") for stem in ("vocals", "instrumental")}


def decode(path):
    from torchcodec.decoders import AudioDecoder
    samples = AudioDecoder(path).get_all_samples()
    return samples.data, samples.sample_rate


def encode(audio, sample_rate, out_path):
    from torchcodec.encoders import AudioEncoder
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    AudioEncoder(audio, sample_rate=sample_rate).to_file(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="audio files or directories")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--format", default="flac", help="output container/codec by extension (flac is lossless and ~6x faster to encode than mp3)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="filename under MODELS_PATH/checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    parser.add_argument("--no_compile", action="store_true", help="skip torch.compile of the trunk (faster startup, slower per file)")
    parser.add_argument("--overwrite", action="store_true", help="redo files whose outputs already exist")
    parser.add_argument("--io_workers", type=int, default=4)
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    todo = []
    for path in files:
        outs = output_paths(path, args.out_dir, args.format)
        if args.overwrite or not all(os.path.exists(p) for p in outs.values()):
            todo.append((path, outs))
    print(f"{len(todo)} of {len(files)} files to separate")
    if not todo:
        return

    t_start = time.perf_counter()
    model = load_melbandroformer(args.weights, device=args.device, compile=not args.no_compile)
    dtype = None if args.fp32 else torch.float16
    total_audio = 0.0

    decoder_pool = ThreadPoolExecutor(max_workers=2)
    encoder_pool = ThreadPoolExecutor(max_workers=args.io_workers)
    pending_encodes = []

    # prefetch two files ahead of the GPU
    futures = [decoder_pool.submit(decode, path) for path, _ in todo[:2]]
    for idx, (path, outs) in enumerate(todo):
        if idx + 2 < len(todo):
            futures.append(decoder_pool.submit(decode, todo[idx + 2][0]))
        t_wait = time.perf_counter()
        try:
            wav, sample_rate = futures.pop(0).result()
        except Exception as e:
            print(f"skip {path}: {type(e).__name__}: {e}")
            continue

        t0 = time.perf_counter()
        wav = wav.to(args.device, non_blocking=True)
        vocals, instrumental = separate(model, wav, sample_rate, batch_size=args.batch_size, dtype=dtype)
        for stem, audio in (("vocals", vocals), ("instrumental", instrumental)):
            pending_encodes.append(encoder_pool.submit(encode, audio.cpu(), sample_rate, outs[stem]))
        seconds = wav.shape[-1] / sample_rate
        total_audio += seconds
        print(f"[{idx + 1}/{len(todo)}] {path}  {seconds:.0f}s audio in {time.perf_counter() - t0:.2f}s"
              f" (waited {t0 - t_wait:.2f}s for decode)", flush=True)

        # keep the encode queue bounded so memory does not grow with the file count
        while len(pending_encodes) > 2 * args.io_workers:
            pending_encodes.pop(0).result()

    t_drain = time.perf_counter()
    for f in pending_encodes:
        f.result()
    decoder_pool.shutdown()
    encoder_pool.shutdown()
    wall = time.perf_counter() - t_start
    print(f"done: {len(todo)} files, {total_audio:.0f}s audio in {wall:.1f}s ({total_audio / wall:.0f}x realtime, "
          f"final encode drain {time.perf_counter() - t_drain:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
