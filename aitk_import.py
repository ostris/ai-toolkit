#!/usr/bin/env python3
"""
Import an ai-toolkit job from a zip file (same format as the UI import).

Usage:
    python3 aitk_import.py /path/to/export.zip
    python3 aitk_import.py http://example.com/export.zip
    python3 aitk_import.py /path/to/export.zip --db /path/to/aitk_db.db
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path


def get_setting(conn, key):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def find_db(hint=None):
    if hint:
        return hint
    candidates = [
        Path(sys.argv[0]).parent / "aitk_db.db",
        Path("/app/ai-toolkit/aitk_db.db"),
    ]
    found = next((str(p) for p in candidates if p.exists()), None)
    if not found:
        print("ERROR: aitk_db.db not found. Use --db to specify the path.", file=sys.stderr)
        sys.exit(1)
    return found


def download_zip(url):
    print(f"Scarico {url} ...")
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        with urllib.request.urlopen(url) as resp:
            total = resp.headers.get("Content-Length")
            downloaded = 0
            chunk = 1024 * 1024  # 1 MB
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                tmp.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / int(total) * 100
                    print(f"\r  {downloaded // 1024 // 1024} MB / {int(total) // 1024 // 1024} MB  ({pct:.0f}%)", end="", flush=True)
        print()
        tmp.flush()
        return tmp.name
    except Exception as e:
        os.unlink(tmp.name)
        raise


def read_manifest(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        if "manifest.json" not in zf.namelist():
            print("ERROR: manifest.json non trovato nello zip.", file=sys.stderr)
            sys.exit(1)
        return json.loads(zf.read("manifest.json"))


def extract_files(zip_path, manifest, training_folder, datasets_root):
    job_name = manifest["job"]["name"]
    output_prefix = f"output/{manifest['paths']['outputFolder']}/" if manifest["paths"].get("outputFolder") else None
    dataset_prefixes = {
        f"datasets/{rel}/": rel for rel in manifest["paths"].get("datasetFolders", [])
    }

    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.infolist():
            if entry.is_dir() or entry.filename == "manifest.json":
                continue

            dest = None

            if output_prefix and entry.filename.startswith(output_prefix):
                rel = entry.filename[len(output_prefix):]
                if rel:
                    dest = Path(training_folder) / job_name / rel

            if dest is None:
                for prefix, rel_folder in dataset_prefixes.items():
                    if entry.filename.startswith(prefix):
                        rel = entry.filename[len(prefix):]
                        if rel:
                            dest = Path(datasets_root) / rel_folder / rel
                        break

            if dest is None:
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())


def rewrite_paths(job_config, training_folder, datasets_root, dataset_rel_folders, db_path):
    cfg = json.loads(json.dumps(job_config))
    for proc in cfg.get("config", {}).get("process", []):
        if "training_folder" in proc:
            proc["training_folder"] = training_folder
        if "sqlite_db_path" in proc:
            proc["sqlite_db_path"] = str(Path(db_path).parent / "aitk_db.db")
        for ds in proc.get("datasets", []):
            if not ds.get("folder_path"):
                continue
            old = ds["folder_path"]
            for rel in dataset_rel_folders:
                if old.endswith(f"/{rel}") or old == rel:
                    ds["folder_path"] = str(Path(datasets_root) / rel)
                    break
    return cfg


def insert_job(conn, manifest, job_config_rewritten):
    job = manifest["job"]
    rows = conn.execute("SELECT MAX(queue_position) FROM job").fetchone()
    queue_pos = (rows[0] or 0) + 1000
    job_id = str(uuid.uuid4())

    conn.execute(
        """
        INSERT INTO job (id, name, gpu_ids, job_config, status, step, job_type, job_ref, queue_position)
        VALUES (?, ?, ?, ?, 'stopped', ?, ?, ?, ?)
        """,
        (
            job_id,
            job["name"],
            job.get("gpu_ids"),
            json.dumps(job_config_rewritten),
            job.get("step", 0),
            job.get("job_type", "train"),
            job.get("job_ref"),
            queue_pos,
        ),
    )
    conn.commit()
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Import an ai-toolkit job from zip")
    parser.add_argument("source", help="Percorso locale o URL http(s) del file zip")
    parser.add_argument("--db", help="Path to aitk_db.db", default=None)
    args = parser.parse_args()

    db_path = find_db(args.db)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    training_folder = get_setting(conn, "TRAINING_FOLDER") or str(Path(db_path).parent / "output")
    datasets_root = get_setting(conn, "DATASETS_FOLDER") or str(Path(db_path).parent / "datasets")

    # Download if URL
    tmp_file = None
    source = args.source
    if source.startswith("http://") or source.startswith("https://"):
        tmp_file = download_zip(source)
        zip_path = tmp_file
    else:
        zip_path = source
        if not Path(zip_path).exists():
            print(f"ERROR: file non trovato: {zip_path}", file=sys.stderr)
            sys.exit(1)

    try:
        manifest = read_manifest(zip_path)

        if not manifest.get("version") or not manifest.get("job"):
            print("ERROR: manifest.json non valido.", file=sys.stderr)
            sys.exit(1)

        job_name = manifest["job"]["name"]

        # Check duplicate
        existing = conn.execute("SELECT id FROM job WHERE name = ?", (job_name,)).fetchone()
        if existing:
            print(f"ERROR: esiste già un job con nome '{job_name}'.", file=sys.stderr)
            sys.exit(1)

        print(f"Importo job '{job_name}'")

        # Create output dir
        Path(training_folder, job_name).mkdir(parents=True, exist_ok=True)

        print("  Estraggo file...")
        extract_files(zip_path, manifest, training_folder, datasets_root)

        print("  Riscrivo percorsi nel config...")
        job_config_rewritten = rewrite_paths(
            manifest["job"]["job_config"],
            training_folder,
            datasets_root,
            manifest["paths"].get("datasetFolders", []),
            db_path,
        )

        print("  Inserisco nel database...")
        job_id = insert_job(conn, manifest, job_config_rewritten)

        print(f"✓ Job '{job_name}' importato (id: {job_id})")

    finally:
        if tmp_file and Path(tmp_file).exists():
            os.unlink(tmp_file)


if __name__ == "__main__":
    main()
