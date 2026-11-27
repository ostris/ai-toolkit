"""Fail-fast dataset pairing check, kept in its own file so data_loader.py only calls it (once per source folder,
before caching starts).

Stops a run when a media file's caption resolves empty (no caption file, a blank one, no default_caption) or a
caption / model sidecar file pairs with no media file: a naming slip on either side that would otherwise train
silently. On with the source's ``strict_pairing``, else the model's ``strict_dataset_pairing`` (YuE2: on)."""
import os


def check_dataset_pairing(dataset, dataset_config, dataset_path, source_items):
    # per source config: a pooled dataset mixes folders with their own caption_ext and strict_pairing
    strict = dataset_config.strict_pairing
    if strict is None:
        strict = getattr(dataset.sd, 'strict_dataset_pairing', False)
    if not strict:
        return
    folder = dataset_path if os.path.isdir(dataset_path) else None
    caption_ext = dataset_config.caption_ext
    items = list({item.path: item for item in source_items}.values())  # repeats and flips share paths
    no_caption = []
    for item in items:
        item.load_caption(dataset.caption_dict)  # the trainer's own caption resolution
        if not (item.raw_caption or '').strip():
            no_caption.append(item.path)
    unpaired = []
    if folder is not None:
        model_sidecars = getattr(dataset.sd, 'get_dataset_sidecar_paths', lambda path: [])
        expected = set()
        for item in items:
            expected.add(os.path.splitext(item.path)[0] + caption_ext)
            expected.update(model_sidecars(item.path))
        sidecar_exts = tuple({os.path.splitext(p)[1].lower() for p in expected})
        defaults = {'default.txt', 'default' + caption_ext}  # folder-wide default prompts pair with every file
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith('.')]  # same pruning as the loader's media walk
            if os.path.basename(root) == '_controls':
                continue
            for name in files:
                path = os.path.join(root, name)
                if name.lower().endswith(sidecar_exts) and not name.startswith('.') and name not in defaults and path not in expected:
                    unpaired.append(path)
    if no_caption or unpaired:
        def show(path):
            return os.path.relpath(path, folder) if folder is not None else path

        lines = [f"  no caption (and no default_caption): {show(p)}" for p in sorted(no_caption)]
        lines += [f"  pairs with no media file: {show(p)}" for p in sorted(unpaired)]
        raise ValueError(
            f"Dataset {dataset_path} has files that do not pair up:\n" + "\n".join(lines)
            + "\nRename them to match, or set strict_pairing: false on the dataset to train anyway."
        )
