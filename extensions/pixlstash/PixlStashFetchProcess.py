"""
PixlStash dataset fetch process for AI-Toolkit.

Downloads images and captions from a PixlStash server into the AI-Toolkit
datasets folder so the dataset appears automatically in the UI.

Config keys
-----------
pixlstash_url : str
    Base URL of the PixlStash server, e.g. "http://localhost:9537".
pixlstash_token : str
    Personal API token (PixlStash → Settings → API Tokens).
source_type : "character" | "picture_set"
    Whether to fetch by character or by picture set.
source_id : int
    The integer ID of the character or picture set to fetch.
caption_mode : "description" | "tags" | "both"   (default: "description")
    Which PixlStash caption source to use.
    "description" — Florence-2 natural-language caption.
    "tags"        — WD14 comma-separated tags.
    "both"        — description first, then tags.
trigger_word : str   (optional)
    Token prepended to every caption, e.g. your LoRA trigger word.
dataset_name : str   (optional)
    Name used for the output subfolder inside the AI-Toolkit datasets root.
    Defaults to the character/set name returned by PixlStash.
datasets_root : str   (optional)
    Absolute path to the AI-Toolkit datasets folder.
    Defaults to "<toolkit_root>/datasets" (matches the UI default).
overwrite : bool   (default: false)
    If false, images that already exist on disk are skipped.
    If true, every image is re-downloaded and captions are rewritten.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import TYPE_CHECKING

from tqdm import tqdm

from jobs.process import BaseExtensionProcess
from toolkit.paths import TOOLKIT_ROOT

if TYPE_CHECKING:
    from jobs import ExtensionJob


class PixlStashFetchProcess(BaseExtensionProcess):
    def __init__(
        self,
        process_id: int,
        job: "ExtensionJob",
        config: OrderedDict,
    ) -> None:
        super().__init__(process_id, job, config)

        self.pixlstash_url: str = self.get_conf("pixlstash_url", required=True)
        self.pixlstash_token: str = self.get_conf("pixlstash_token", required=True)
        self.source_type: str = self.get_conf("source_type", required=True)
        self.source_id: int = int(self.get_conf("source_id", required=True))
        self.caption_mode: str = self.get_conf("caption_mode", default="description")
        self.trigger_word: str = self.get_conf("trigger_word", default="")
        self.dataset_name: str | None = self.get_conf("dataset_name", default=None)
        self.overwrite: bool = self.get_conf("overwrite", default=False)
        self.min_score: int = int(self.get_conf("min_score", default=0))
        self.verify_ssl: bool = str(
            self.get_conf("verify_ssl", default="true")
        ).lower() not in ("false", "0", "no", "off")

        # Where to write the dataset.  Defaults to the same root the UI watches.
        default_datasets_root = os.path.join(TOOLKIT_ROOT, "datasets")
        self.datasets_root: str = self.get_conf(
            "datasets_root", default=default_datasets_root
        )

        if self.source_type not in ("character", "picture_set"):
            raise ValueError(
                f"source_type must be 'character' or 'picture_set', got '{self.source_type}'"
            )
        if self.caption_mode not in ("description", "tags", "both"):
            raise ValueError(
                f"caption_mode must be 'description', 'tags', or 'both', got '{self.caption_mode}'"
            )

    # ------------------------------------------------------------------

    def run(self) -> None:
        super().run()

        # Import here so the module is only loaded when this process runs
        from extensions.pixlstash.pixlstash_client import PixlStashClient

        print(f"\n[PixlStash] Connecting to {self.pixlstash_url} …")
        client = PixlStashClient(
            self.pixlstash_url, self.pixlstash_token, verify_ssl=self.verify_ssl
        )
        client.login()
        print("[PixlStash] Authenticated.")

        # ---- resolve source name and picture list -------------------------
        if self.source_type == "character":
            source = client.get_character(self.source_id)
            source_label = f"character '{source['name']}' (id={self.source_id})"
            pictures = client.list_pictures_for_character(self.source_id)
        else:
            source = client.get_picture_set(self.source_id)
            source_label = f"picture set '{source['name']}' (id={self.source_id})"
            pictures = client.list_pictures_for_set(self.source_id)

        total = len(pictures)
        print(
            f"[PixlStash] Fetched {source_label} — {total} picture(s) found in source.",
            flush=True,
        )

        # ---- apply score filter ------------------------------------------
        if self.min_score > 0:
            before = total
            pictures = [p for p in pictures if (p.get("score") or 0) >= self.min_score]
            total = len(pictures)
            filtered_out = before - total
            print(
                f"[PixlStash] Score filter ≥{self.min_score}★: "
                f"{total} picture(s) kept, {filtered_out} filtered out.",
                flush=True,
            )

        print(
            f"[PixlStash] Downloading {source_label} — {total} picture(s) found.",
            flush=True,
        )

        # ---- resolve output folder ----------------------------------------
        dataset_name = self.dataset_name or self._safe_folder_name(source["name"])
        output_folder = os.path.join(self.datasets_root, dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        print(f"[PixlStash] Output folder: {output_folder}")

        # ---- download loop -----------------------------------------------
        downloaded = 0
        skipped = 0
        errors = 0

        for pic in tqdm(pictures, desc="Downloading", unit="img"):
            pic_id = pic["id"]
            img_filename = f"{pic_id}.jpg"
            txt_filename = f"{pic_id}.txt"
            img_path = os.path.join(output_folder, img_filename)
            txt_path = os.path.join(output_folder, txt_filename)

            if (
                not self.overwrite
                and os.path.exists(img_path)
                and os.path.exists(txt_path)
            ):
                skipped += 1
                print(f"PROGRESS:{downloaded + skipped}/{total}", flush=True)
                continue

            try:
                # The listing/embed rows only carry scalar grid fields, so the
                # natural-language description and WD14 tags are read per-picture
                # from GET /pictures/{id}/metadata.  Fetched here, after the
                # on-disk skip check, so we never query metadata for images we
                # are about to skip.
                meta = client.get_picture_metadata(pic_id)
                fmt = meta.get("format", "jpg") or "jpg"

                # Build caption
                caption = client.build_caption(
                    meta,
                    mode=self.caption_mode,
                    trigger=self.trigger_word,
                )

                # Download image
                img_bytes = client.download_image_bytes(pic_id, fmt)

                # Write image (always save as .jpg for maximum AI-Toolkit compat)
                if fmt.lower() in ("jpg", "jpeg"):
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                else:
                    # Convert to JPEG via PIL so AI-Toolkit doesn't have to
                    import io
                    from PIL import Image

                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_img.save(img_path, format="JPEG", quality=95)

                # Write caption
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                downloaded += 1
                print(f"PROGRESS:{downloaded + skipped}/{total}", flush=True)

            except Exception as exc:
                print(
                    f"\n[PixlStash] WARNING: Failed to fetch picture id={pic_id}: {exc}",
                    flush=True,
                )
                errors += 1
                print(f"PROGRESS:{downloaded + skipped}/{total}", flush=True)

        # ---- summary -----------------------------------------------------
        print(
            f"\n[PixlStash] Done — {downloaded} downloaded, "
            f"{skipped} skipped (already on disk), {errors} errors."
        )
        print(f"[PixlStash] Dataset '{dataset_name}' is ready in the AI-Toolkit UI.")

    # ------------------------------------------------------------------

    @staticmethod
    def _safe_folder_name(name: str) -> str:
        """Convert an arbitrary string into a safe directory name."""
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        return safe.strip().replace(" ", "_")
