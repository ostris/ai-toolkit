"""
CLI bridge between the Next.js API routes and the Python data-source plugin system.

Commands
--------
plugins                  Print JSON array of { id, display_name, settings_schema }
                         for every registered source plugin.

browse SOURCE_TYPE_ID    Print a JSON object:
                           { groups: [...], import_fields: [...] }
                         where groups contains all browseable items for the
                         given plugin (e.g. characters + picture sets for
                         PixlStash).

thumbnail SOURCE_ID THUMBNAIL_ID THUMBNAIL_TYPE
                         Fetch a thumbnail from the plugin and print a JSON
                         object: { content_type: str, data: <base64> }

job-config SOURCE_ID     Read import params JSON from stdin, print the
                         process-config JSON dict for run.py to stdout.

Usage
-----
    python -m toolkit.dataset_sources.cli plugins
    python -m toolkit.dataset_sources.cli browse pixlstash
    python -m toolkit.dataset_sources.cli thumbnail pixlstash 42 character
    echo '{"source_type":"character","source_id":"1"}' | \\
        python -m toolkit.dataset_sources.cli job-config pixlstash
"""

from __future__ import annotations

import json
import os
import sys


def _load_extensions() -> None:
    """Auto-discover and import all extensions so they register their sources."""
    ext_dir = os.path.join(os.path.dirname(__file__), "..", "..", "extensions")
    if not os.path.isdir(ext_dir):
        return
    for name in sorted(os.listdir(ext_dir)):
        pkg = os.path.join(ext_dir, name, "__init__.py")
        if os.path.isfile(pkg):
            try:
                __import__(f"extensions.{name}")
            except Exception:
                pass  # extension may have missing optional deps — skip silently


def cmd_plugins() -> None:
    import base64
    import dataclasses
    from toolkit.dataset_sources.registry import get_all_sources, load_settings_from_db
    from toolkit.paths import TOOLKIT_ROOT

    db_path = os.path.join(TOOLKIT_ROOT, "aitk_db.db")
    settings = load_settings_from_db(db_path)

    def _icon_data_url(cls) -> str | None:
        if not cls.icon_path:
            return None
        try:
            with open(cls.icon_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(cls.icon_path)[1].lower()
            mime = {'.png': 'image/png', '.svg': 'image/svg+xml', '.jpg': 'image/jpeg'}.get(ext, 'image/png')
            return f"data:{mime};base64,{data}"
        except Exception:
            return None

    out = [
        {
            "id": cls.type_id,
            "display_name": cls.display_name,
            "icon": _icon_data_url(cls),
            "settings_schema": [
                dataclasses.asdict(f) for f in cls.get_settings_schema()
            ],
        }
        for cls in get_all_sources()
        if cls.is_configured(settings)
    ]
    print(json.dumps(out))


def cmd_browse(type_id: str) -> None:
    from toolkit.dataset_sources.registry import get_source, load_settings_from_db
    from toolkit.paths import TOOLKIT_ROOT

    SourceClass = get_source(type_id)
    if SourceClass is None:
        print(json.dumps({"error": f"Unknown plugin: {type_id}"}))
        sys.exit(1)

    db_path = os.path.join(TOOLKIT_ROOT, "aitk_db.db")
    settings = load_settings_from_db(db_path)
    source = SourceClass(settings)

    try:
        groups = source.browse()
        import_fields = SourceClass.get_import_fields()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    def _item(i):
        return {
            "id": i.id,
            "name": i.name,
            "picture_count": i.picture_count,
            "thumbnail_id": i.thumbnail_id,
            "thumbnail_type": i.thumbnail_type,
        }

    def _group(g):
        return {"id": g.id, "label": g.label, "items": [_item(i) for i in g.items]}

    def _field(f):
        return {
            "id": f.id,
            "label": f.label,
            "field_type": f.field_type,
            "options": f.options,
            "default": f.default,
            "required": f.required,
        }

    print(
        json.dumps(
            {
                "groups": [_group(g) for g in groups],
                "import_fields": [_field(f) for f in import_fields],
            }
        )
    )


def cmd_thumbnail(source_id: str, thumbnail_id: str, thumbnail_type: str) -> None:
    import base64
    from toolkit.dataset_sources.registry import get_source, load_settings_from_db
    from toolkit.paths import TOOLKIT_ROOT

    SourceClass = get_source(source_id)
    if SourceClass is None:
        print(json.dumps({"error": f"Unknown plugin: {source_id}"}))
        sys.exit(1)

    db_path = os.path.join(TOOLKIT_ROOT, "aitk_db.db")
    settings = load_settings_from_db(db_path)
    source = SourceClass(settings)

    try:
        image_bytes, content_type = source.get_thumbnail(thumbnail_id, thumbnail_type)
        print(
            json.dumps(
                {
                    "content_type": content_type,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


def cmd_job_config(source_id: str) -> None:
    from toolkit.dataset_sources.registry import get_source, load_settings_from_db
    from toolkit.paths import TOOLKIT_ROOT

    SourceClass = get_source(source_id)
    if SourceClass is None:
        print(json.dumps({"error": f"Unknown plugin: {source_id}"}))
        sys.exit(1)

    db_path = os.path.join(TOOLKIT_ROOT, "aitk_db.db")
    settings = load_settings_from_db(db_path)
    source = SourceClass(settings)

    params = json.loads(sys.stdin.read())

    try:
        config = source.build_job_config(params)
        print(json.dumps(config))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    _load_extensions()

    if len(sys.argv) < 2:
        print(
            "Usage: python -m toolkit.dataset_sources.cli <plugins|browse|thumbnail|job-config> ...",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "plugins":
        cmd_plugins()
    elif cmd == "browse":
        if len(sys.argv) < 3:
            print(
                "Usage: python -m toolkit.dataset_sources.cli browse <source_type_id>",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_browse(sys.argv[2])
    elif cmd == "thumbnail":
        if len(sys.argv) < 5:
            print(
                "Usage: python -m toolkit.dataset_sources.cli thumbnail <source_id> <thumbnail_id> <thumbnail_type>",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_thumbnail(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "job-config":
        if len(sys.argv) < 3:
            print(
                "Usage: python -m toolkit.dataset_sources.cli job-config <source_id>",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd_job_config(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
