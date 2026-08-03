import os
import sys
import queue
import random
import argparse
import threading
import traceback

import torch
from tqdm import tqdm

# allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.control_generator import ControlGenerator, img_ext_list


def control_exists(img_path, control_type):
    # mirrors the lookup in ControlGenerator.get_control_path so we can skip
    # images another instance has already finished
    controls_folder = os.path.join(os.path.dirname(img_path), "_controls")
    file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
    file_name_no_ext_control = f"{file_name_no_ext}.{control_type}"
    for ext in img_ext_list:
        if os.path.exists(os.path.join(controls_folder, file_name_no_ext_control + ext)):
            return True
    return False


# sentinel pushed through the queues to tell workers to stop
_DONE = object()


def run_pipeline(control_gen, img_list, control_type, regen, n_load, n_save):
    # Three-stage pipeline so the GPU never waits on disk/CPU work:
    #   loaders (N threads)  -> read + exif + resize + preprocess to a CPU tensor
    #   gpu worker (1 thread) -> model forward only (kept single so VRAM is bounded)
    #   savers  (M threads)  -> postprocess (resize/alpha) + encode + write
    # The heavy CPU work (resize/normalize on input, resize/convert on output) is
    # pushed onto the loader/saver threads so the GPU thread does almost nothing
    # but the forward pass. Bounded queues apply backpressure so we don't load the
    # whole dataset into RAM.
    infer_q = queue.Queue(maxsize=n_load * 2)
    save_q = queue.Queue(maxsize=n_save * 2)
    path_q = queue.Queue()
    for img_path in img_list:
        path_q.put(img_path)

    # miniters=1 disables tqdm's dynamic-miniters heuristic (which otherwise
    # raises the redraw threshold after a fast burst and makes the bar look
    # frozen); mininterval keeps redraws time-based and cheap.
    pbar = tqdm(total=len(img_list), desc=f"Generating {control_type}",
                miniters=1, mininterval=0.5)
    pbar_lock = threading.Lock()
    # set on completion OR on Ctrl-C; every blocking call below uses a timeout and
    # re-checks this so the worker threads can actually be shut down.
    stop_event = threading.Event()

    def put(q, item):
        # interruptible put: blocks until there's room, but wakes periodically so
        # a stop request (or KeyboardInterrupt on the main thread) is honored.
        while not stop_event.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def loader():
        while not stop_event.is_set():
            try:
                img_path = path_q.get_nowait()
            except queue.Empty:
                break
            try:
                if not regen and control_exists(img_path, control_type):
                    # another instance (or a previous run) already did it
                    with pbar_lock:
                        pbar.update(1)
                    continue
                image = control_gen.load_image(img_path)
                payload = control_gen.preprocess(image, control_type)
                put(infer_q, (img_path, image, payload))
            except Exception:
                traceback.print_exc()
                with pbar_lock:
                    pbar.update(1)

    def saver():
        while not stop_event.is_set():
            try:
                item = save_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is _DONE:
                break
            img_path, image, result = item
            try:
                out_image = control_gen.postprocess(result, image, control_type)
                save_path = control_gen.control_save_path(img_path, control_type)
                control_gen.save_control(out_image, save_path)
            except Exception:
                traceback.print_exc()
            finally:
                with pbar_lock:
                    pbar.update(1)

    loaders = [threading.Thread(target=loader, daemon=True) for _ in range(n_load)]
    savers = [threading.Thread(target=saver, daemon=True) for _ in range(n_save)]
    for t in loaders + savers:
        t.start()

    # GPU stage runs on the main thread: pull preprocessed tensors, run the
    # forward pass, hand the raw result off to the savers. We stop once every
    # loader has exited and nothing is left queued for inference.
    interrupted = False
    try:
        while not stop_event.is_set():
            if not any(t.is_alive() for t in loaders) and infer_q.empty():
                break
            try:
                img_path, image, payload = infer_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result = control_gen.run_inference(payload, control_type)
                put(save_q, (img_path, image, result))
            except Exception:
                traceback.print_exc()
                with pbar_lock:
                    pbar.update(1)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted, shutting down...")

    if interrupted:
        # abort: tell every worker to stop; pending items are dropped
        stop_event.set()
    else:
        # normal finish: let savers drain whatever is still queued, then stop
        for _ in savers:
            save_q.put(_DONE)

    # join with a timeout so a stuck worker can never wedge shutdown; threads are
    # daemons, so anything still alive is torn down when we return.
    for t in savers:
        t.join(timeout=5)
    pbar.close()
    if interrupted:
        raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks for a dataset using the ControlGenerator"
    )
    parser.add_argument("img_dir", type=str, help="Path to image directory")
    parser.add_argument(
        "--control",
        type=str,
        default="mask",
        choices=["mask", "inpaint", "depth", "pose", "line", "sapiens2_mask"],
        help="Control type to generate (default: mask)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (default: cuda)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Regenerate controls even if they already exist",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order so multiple instances on the same dataset "
        "don't chase the same images",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=16,
        help="Number of threads for loading/resizing images (default: 4)",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=16,
        help="Number of threads for saving controls (default: 4)",
    )
    args = parser.parse_args()

    img_dir = args.img_dir
    if not os.path.isdir(img_dir):
        print(f"Error: {img_dir} is not a directory")
        sys.exit(1)

    # find images, skipping existing _controls folders and hidden files
    img_list = []
    for root, dirs, files in os.walk(img_dir):
        if "_controls" in root:
            continue
        for file in files:
            if file.startswith("."):
                continue
            if file.lower().endswith(tuple(img_ext_list)):
                img_list.append(os.path.join(root, file))

    if len(img_list) == 0:
        print(f"Error: no images found in {img_dir}")
        sys.exit(1)

    # filter out images that already have controls up front so the progress bar
    # reflects only real work (otherwise it races through thousands of skips and
    # the rate/ETA are meaningless). The loader still re-checks just before
    # processing to handle the multi-instance race.
    if not args.regen:
        total = len(img_list)
        img_list = [p for p in img_list if not control_exists(p, args.control)]
        skipped = total - len(img_list)
        if skipped:
            print(f"Skipping {skipped} images that already have '{args.control}' controls")
        if len(img_list) == 0:
            print("All images already have controls. Nothing to do.")
            return

    if args.shuffle:
        random.shuffle(img_list)

    control_gen = ControlGenerator(torch.device(args.device))
    control_gen.debug = args.debug
    control_gen.regen = args.regen

    interrupted = False
    try:
        run_pipeline(
            control_gen,
            img_list,
            args.control,
            args.regen,
            max(1, args.load_workers),
            max(1, args.save_workers),
        )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        control_gen.cleanup()

    if interrupted:
        sys.exit(130)
    print("Done")


if __name__ == "__main__":
    main()
