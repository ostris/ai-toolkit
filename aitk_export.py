#!/usr/bin/env python3
"""
Export an ai-toolkit job to a zip file (same format as the UI export).

Usage:
    python3 aitk_export.py <job_name_or_id> [--out /path/to/output.zip] [--db /path/to/aitk_db.db]
"""

import argparse
import json
import os
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def get_setting(conn, key):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def find_job(conn, name_or_id):
    row = conn.execute("SELECT * FROM job WHERE id = ?", (name_or_id,)).fetchone()
    if not row:
        row = conn.execute("SELECT * FROM job WHERE name = ?", (name_or_id,)).fetchone()
    return row


def main():
    parser = argparse.ArgumentParser(description="Export an ai-toolkit job to zip")
    parser.add_argument("job", help="Job name or ID")
    parser.add_argument("--out", help="Output zip path (default: <job_name>_export.zip)")
    parser.add_argument("--db", help="Path to aitk_db.db", default=None)
    args = parser.parse_args()

    # Find the database
    if args.db:
        db_path = args.db
    else:
        candidates = [
            Path(__file__).parent / "aitk_db.db",
            Path("/app/ai-toolkit/aitk_db.db"),
        ]
        db_path = next((str(p) for p in candidates if p.exists()), None)
        if not db_path:
            print("ERROR: aitk_db.db not found. Use --db to specify the path.", file=sys.stderr)
            sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    job = find_job(conn, args.job)
    if not job:
        print(f"ERROR: job '{args.job}' not found.", file=sys.stderr)
        sys.exit(1)

    training_folder = get_setting(conn, "TRAINING_FOLDER") or str(Path(db_path).parent / "output")
    datasets_root = get_setting(conn, "DATASETS_FOLDER") or str(Path(db_path).parent / "datasets")

    job_output_folder = Path(training_folder) / job["name"]
    has_output = job_output_folder.exists()

    # Parse dataset paths from job config
    dataset_abs_paths = []
    dataset_rel_paths = []
    try:
        job_config = json.loads(job["job_config"])
        datasets = job_config.get("config", {}).get("process", [{}])[0].get("datasets", [])
        for ds in datasets:
            abs_path = ds.get("folder_path")
            if not abs_path:
                continue
            try:
                rel = os.path.relpath(abs_path, datasets_root)
            except ValueError:
                continue
            if not rel.startswith("..") and os.path.exists(abs_path):
                dataset_abs_paths.append(abs_path)
                dataset_rel_paths.append(rel)
    except Exception:
        pass

    manifest = {
        "version": 1,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "job": {
            "name": job["name"],
            "gpu_ids": job["gpu_ids"],
            "job_config": json.loads(job["job_config"]),
            "status": "stopped",
            "step": job["step"],
            "job_type": job["job_type"],
            "job_ref": job["job_ref"] if "job_ref" in job.keys() else None,
        },
        "paths": {
            "outputFolder": job["name"] if has_output else None,
            "datasetFolders": dataset_rel_paths,
        },
    }

    out_path = args.out or f"{job['name']}_export.zip"

    print(f"Exporting job '{job['name']}' → {out_path}")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        if has_output:
            print(f"  Adding output folder: {job_output_folder}")
            for root, _, files in os.walk(job_output_folder):
                for file in files:
                    abs_file = Path(root) / file
                    arc_name = f"output/{job['name']}/{abs_file.relative_to(job_output_folder)}"
                    zf.write(abs_file, arc_name)

        for abs_ds, rel_ds in zip(dataset_abs_paths, dataset_rel_paths):
            print(f"  Adding dataset: {abs_ds}")
            for root, _, files in os.walk(abs_ds):
                for file in files:
                    abs_file = Path(root) / file
                    arc_name = f"datasets/{rel_ds}/{abs_file.relative_to(abs_ds)}"
                    zf.write(abs_file, arc_name)

    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"✓ Done ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
