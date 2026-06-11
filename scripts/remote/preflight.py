"""Preflight: validation and config remap for the remote-GPU training pipeline.

Catches every cheap-to-catch failure before a pod is provisioned (R1) and
emits a remote-ready derived config to runs/<run>/remote_config.yaml without
ever touching the source config (R2). All path-bearing keys are remapped to
the remote namespace and their referenced files queued for upload (R3); a
closing generic sweep fails on any string that still looks like a local
filesystem path. Training-safety invariants are enforced (R4) and dataset
upload exclusions are reported (R5; transport applies the filters).

Config loading replicates toolkit/config.py semantics (env-var substitution,
config/ dir resolution, [name] tag replacement, the float-exponent loader
fix) without importing it: toolkit/config.py needs oyaml, which is not a
laptop-side dependency of this pipeline.

Run directly is not supported; the CLI entrypoint (cli.py, U8) calls
run_preflight(). Helpers are pure where practical for testability.
"""

import copy
import fnmatch
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract, manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_EXTENSIONS = ['.json', '.jsonc', '.yaml', '.yml']
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# Inline sample-prompt flag, per the SampleItem prompt grammar in
# toolkit/config_modules.py (prompt.split('--'); content runs to next flag).
INLINE_CTRL_IMG_RE = re.compile(r"(--ctrl_img\s+)(.+?)(?=\s+--|\s*$)")

# Bare HF hub id: org/name, exactly one slash, no path-ish prefix.
HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

# Fixes yaml not loading bare exponents (1e-4) as floats — same resolver as
# toolkit/config.py, on a subclass so the global SafeLoader is not mutated.
_FLOAT_RE = re.compile(
    u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X)


class _ConfigLoader(yaml.SafeLoader):
    pass


_ConfigLoader.add_implicit_resolver(
    u'tag:yaml.org,2002:float', _FLOAT_RE, list(u'-+0123456789.'))


class PreflightError(Exception):
    pass


# ---------------------------------------------------------------------------
# Path remap table (R3) — new path-bearing keys are one-line additions.
# Patterns are relative to a process entry; '*' matches every list index.
# Kinds:
#   training_folder — replaced with the remote output root, nothing uploaded
#   dir             — local directory, uploaded under the dataset area
#   image           — local file, uploaded under the ctrl/asset area (strict:
#                     must exist locally unless already remote)
#   image_list      — image, but the value may be a list or comma-joined str
#   weights         — like image, but bare HF repo ids pass through untouched
#                     (assistant/pretrained loras are often hub references)
# ---------------------------------------------------------------------------

REMAP_TABLE = [
    (('training_folder',), 'training_folder'),
    (('datasets', '*', 'folder_path'), 'dir'),
    (('datasets', '*', 'control_path'), 'dir'),
    (('datasets', '*', 'control_path_1'), 'dir'),
    (('datasets', '*', 'control_path_2'), 'dir'),
    (('datasets', '*', 'control_path_3'), 'dir'),
    (('datasets', '*', 'mask_path'), 'dir'),
    (('datasets', '*', 'unconditional_path'), 'dir'),
    (('datasets', '*', 'clip_image_path'), 'dir'),
    (('sample', 'samples', '*', 'ctrl_img'), 'image'),
    (('sample', 'samples', '*', 'ctrl_img_1'), 'image'),
    (('sample', 'samples', '*', 'ctrl_img_2'), 'image'),
    (('sample', 'samples', '*', 'ctrl_img_3'), 'image'),
    (('adapter', 'test_img_path'), 'image_list'),
    (('model', 'lora_path'), 'weights'),
    (('model', 'pretrained_lora_path'), 'weights'),
    (('model', 'assistant_lora_path'), 'weights'),
    (('model', 'inference_lora_path'), 'weights'),
    # pretrained_lora_path actually lives under network in NetworkConfig;
    # cover both homes so the generic sweep never has to catch it.
    (('network', 'pretrained_lora_path'), 'weights'),
]

# Inline --ctrl_img flags live inside these prompt-string locations.
PROMPT_STRING_PATTERNS = [
    ('sample', 'samples', '*', 'prompt'),
    ('sample', 'prompts', '*'),  # legacy list-of-strings form
]


@dataclass
class DatasetReport:
    folder: str
    image_count: int = 0
    caption_count: int = 0
    uncaptioned: list = field(default_factory=list)   # stems missing .txt
    excluded: list = field(default_factory=list)      # rel paths transport will skip
    file_count: int = 0                               # after exclusions
    total_bytes: int = 0                              # after exclusions


@dataclass
class PreflightResult:
    run_name: str
    job_name: str
    derived_config_path: str
    config_hash: str
    upload_set: list                                  # [(local_path, remote_path)]
    warnings: list
    changes: list
    dataset_reports: list


# ---------------------------------------------------------------------------
# Config loading (toolkit/config.py semantics, minimal local implementation)
# ---------------------------------------------------------------------------

def replace_env_vars_in_string(s: str) -> str:
    """Replace ${VAR_NAME} placeholders; error on unset vars."""
    def replacer(match):
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise PreflightError(
                f"environment variable {var_name} is referenced by the config "
                "but not set"
            )
        return value
    return re.sub(r'\$\{([^}]+)\}', replacer, s)


def resolve_config_path(config_path: str, base_dir: str = ".") -> str:
    """Find the config like toolkit/config.py: config/ dir first, then the
    path as given (relative paths against base_dir, then cwd)."""
    candidates = []
    for root in (os.path.join(base_dir, 'config'), os.path.join(REPO_ROOT, 'config')):
        in_config = os.path.join(root, config_path)
        candidates.append(in_config)
        candidates.extend(in_config + ext for ext in CONFIG_EXTENSIONS)
    candidates.append(config_path)
    if not os.path.isabs(config_path):
        candidates.append(os.path.join(base_dir, config_path))
        candidates.append(os.path.join(os.getcwd(), config_path))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    raise PreflightError(f"could not find config file {config_path}")


def replace_name_tag(node, name: str):
    """Replace the [name] tag in every string value (toolkit preprocess)."""
    if isinstance(node, str):
        return node.replace('[name]', name)
    if isinstance(node, dict):
        return {k: replace_name_tag(v, name) for k, v in node.items()}
    if isinstance(node, list):
        return [replace_name_tag(v, name) for v in node]
    return node


def load_source_config(config_path: str, base_dir: str = ".") -> dict:
    """Parse the source config with toolkit/config.py semantics. Raises
    PreflightError with the parser's message on malformed input."""
    real_path = resolve_config_path(config_path, base_dir)
    with open(real_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = replace_env_vars_in_string(content)
    try:
        if real_path.endswith('.json') or real_path.endswith('.jsonc'):
            config = json.loads(content)
        elif real_path.endswith('.yaml') or real_path.endswith('.yml'):
            config = yaml.load(content, Loader=_ConfigLoader)
        else:
            raise PreflightError(
                f"config file {config_path} must be a json or yaml file")
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise PreflightError(
            f"could not parse config {real_path}: {e}") from None
    if not isinstance(config, dict):
        raise PreflightError(
            f"could not parse config {real_path}: top level is not a mapping")
    return config


def validate_structure(config: dict):
    """R1 structural checks: job key, config.name, process[0], sample block."""
    if 'job' not in config:
        raise PreflightError("config file must have a job key")
    if 'config' not in config or not isinstance(config['config'], dict):
        raise PreflightError("config file must have a config section")
    if not config['config'].get('name'):
        raise PreflightError("config file must have a config.name key")
    process = config['config'].get('process')
    if not isinstance(process, list) or len(process) == 0 or not isinstance(process[0], dict):
        raise PreflightError("config.process must be a non-empty list")
    if not isinstance(process[0].get('sample'), dict):
        raise PreflightError(
            "config.process[0] must have a sample block (the pipeline's "
            "monitoring depends on sample output)")


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

# the remote namespace root (contract.REMOTE_RUNS_ROOT lives under it)
REMOTE_PREFIX = '/workspace'


def is_remote_path(value: str) -> bool:
    return isinstance(value, str) and (
        value == REMOTE_PREFIX or value.startswith(REMOTE_PREFIX + '/'))


def looks_like_local_path(value, base_dir: str = ".") -> bool:
    """True when a string is a local-filesystem-shaped path. Bare HF repo
    ids (org/name) are NOT flagged unless they resolve to a real local file."""
    if not isinstance(value, str) or not value:
        return False
    if is_remote_path(value):
        return False
    if value.startswith(('/', './', '../')) or value == '.' or value == '..':
        return True
    if value.startswith('~'):
        return True
    if '/' in value:
        # org/name HF ids share this shape; only flag if it exists on disk
        expanded = os.path.expanduser(value)
        if os.path.exists(expanded) or os.path.exists(os.path.join(base_dir, expanded)):
            return True
    return False


def _resolve_local(value: str, base_dir: str) -> str:
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(base_dir, expanded))


# ---------------------------------------------------------------------------
# Remap engine
# ---------------------------------------------------------------------------

def _iter_pattern(node, pattern, prefix):
    """Yield (container, key_or_index, dotted_path) for every existing match
    of a key-path pattern ('*' = every list index) under node."""
    head, rest = pattern[0], pattern[1:]
    if head == '*':
        if isinstance(node, list):
            for i in range(len(node)):
                dotted = f"{prefix}[{i}]"
                if rest:
                    yield from _iter_pattern(node[i], rest, dotted)
                else:
                    yield node, i, dotted
    else:
        if isinstance(node, dict) and head in node:
            dotted = f"{prefix}.{head}" if prefix else head
            if rest:
                yield from _iter_pattern(node[head], rest, dotted)
            else:
                yield node, head, dotted


class _Remapper:
    """Collects upload pairs and changes while rewriting one derived config."""

    def __init__(self, run_name: str, base_dir: str):
        self.run_name = run_name
        self.base_dir = base_dir
        self.upload_set = []        # [(local_abs, remote)]
        self.changes = []
        self.dataset_dirs = []      # [(dotted, local_abs, is_caption_dataset)]
        self._remote_sources = {}   # remote path -> local source (collision guard)

    def _add_upload(self, dotted: str, local_abs: str, remote: str):
        prior = self._remote_sources.get(remote)
        if prior is not None and prior != local_abs:
            raise PreflightError(
                f"{dotted}: remote path collision — {local_abs} and {prior} "
                f"both map to {remote}; rename one so basenames are unique")
        if prior is None:
            self._remote_sources[remote] = local_abs
            self.upload_set.append((local_abs, remote))

    def _record(self, dotted: str, old, new):
        self.changes.append(f"{dotted}: {old!r} -> {new!r}")

    def remap_dir(self, dotted: str, value: str, is_caption_dataset: bool) -> str:
        if is_remote_path(value):
            return value
        local_abs = _resolve_local(value, self.base_dir)
        if not os.path.isdir(local_abs):
            raise PreflightError(f"{dotted}: dataset folder not found: {value}")
        basename = os.path.basename(os.path.normpath(local_abs))
        remote = f"{contract.remote_dataset_dir(self.run_name)}/{basename}"
        self._add_upload(dotted, local_abs, remote)
        self.dataset_dirs.append((dotted, local_abs, is_caption_dataset))
        self._record(dotted, value, remote)
        return remote

    def remap_file(self, dotted: str, value: str, kind: str) -> str:
        if is_remote_path(value):
            return value
        if kind == 'weights' and HF_REPO_ID_RE.match(value) \
                and not looks_like_local_path(value, self.base_dir):
            return value  # hub reference, valid on the pod as-is
        local_abs = _resolve_local(value, self.base_dir)
        if not os.path.isfile(local_abs):
            raise PreflightError(f"{dotted}: file not found: {value}")
        basename = os.path.basename(local_abs)
        remote = f"{contract.remote_ctrl_dir(self.run_name)}/{basename}"
        self._add_upload(dotted, local_abs, remote)
        self._record(dotted, value, remote)
        return remote

    def remap_prompt(self, dotted: str, prompt: str) -> str:
        def repl(m):
            value = m.group(2).strip()
            remote = self.remap_file(f"{dotted} (--ctrl_img)", value, 'image')
            return m.group(1) + remote
        return INLINE_CTRL_IMG_RE.sub(repl, prompt)

    def apply(self, process: dict, prefix: str):
        for pattern, kind in REMAP_TABLE:
            for container, key, dotted in _iter_pattern(process, pattern, prefix):
                value = container[key]
                if kind == 'training_folder':
                    remote = contract.remote_training_folder(self.run_name)
                    if value != remote:
                        self._record(dotted, value, remote)
                        container[key] = remote
                elif kind == 'dir':
                    if isinstance(value, str):
                        container[key] = self.remap_dir(
                            dotted, value, pattern[-1] == 'folder_path')
                elif kind == 'image':
                    if isinstance(value, str):
                        container[key] = self.remap_file(dotted, value, kind)
                elif kind == 'image_list':
                    if isinstance(value, str):
                        parts = [p.strip() for p in value.split(',') if p.strip()]
                        container[key] = ','.join(
                            self.remap_file(dotted, p, 'image') for p in parts)
                    elif isinstance(value, list):
                        container[key] = [
                            self.remap_file(f"{dotted}[{i}]", p, 'image')
                            if isinstance(p, str) else p
                            for i, p in enumerate(value)]
                elif kind == 'weights':
                    if isinstance(value, str):
                        container[key] = self.remap_file(dotted, value, kind)
        for pattern in PROMPT_STRING_PATTERNS:
            for container, key, dotted in _iter_pattern(process, pattern, prefix):
                if isinstance(container[key], str):
                    container[key] = self.remap_prompt(dotted, container[key])


# ---------------------------------------------------------------------------
# Invariants (R4) and warnings (R21)
# ---------------------------------------------------------------------------

# Keep-all stand-in. NEVER -1: clean_up_saves() in BaseSDTrainProcess slices
# files[:-n], so -1 deletes the OLDEST checkpoint at every save.
MAX_STEP_SAVES_TO_KEEP = 10000

PUSH_TO_HUB_WARNING = (
    "save.push_to_hub is set: the HF_TOKEN forwarded to the pod must have "
    "write scope and is visible to the provider (R21). Use a read-only "
    "token and push manually unless you accept that."
)


def enforce_invariants(process: dict, prefix: str, changes: list, warnings: list):
    logging_block = process.get('logging')
    if not isinstance(logging_block, dict):
        logging_block = {}
        process['logging'] = logging_block
        changes.append(f"{prefix}.logging: <absent> -> created")
    if logging_block.get('use_ui_logger') is not True:
        old = logging_block.get('use_ui_logger', '<absent>')
        logging_block['use_ui_logger'] = True
        changes.append(f"{prefix}.logging.use_ui_logger: {old} -> True")

    save_block = process.get('save')
    if not isinstance(save_block, dict):
        save_block = {}
        process['save'] = save_block
        changes.append(f"{prefix}.save: <absent> -> created")
    if save_block.get('max_step_saves_to_keep') != MAX_STEP_SAVES_TO_KEEP:
        old = save_block.get('max_step_saves_to_keep', '<absent>')
        save_block['max_step_saves_to_keep'] = MAX_STEP_SAVES_TO_KEEP
        changes.append(
            f"{prefix}.save.max_step_saves_to_keep: {old} -> "
            f"{MAX_STEP_SAVES_TO_KEEP}")

    if save_block.get('push_to_hub'):
        warnings.append(PUSH_TO_HUB_WARNING)


# ---------------------------------------------------------------------------
# Generic closing sweep (R3 backstop)
# ---------------------------------------------------------------------------

def sweep_local_paths(node, prefix: str = "", base_dir: str = ".") -> list:
    """Walk every string value; return [(dotted_path, value)] for anything
    that still looks like a local filesystem path."""
    offenders = []
    if isinstance(node, dict):
        for k, v in node.items():
            dotted = f"{prefix}.{k}" if prefix else str(k)
            offenders.extend(sweep_local_paths(v, dotted, base_dir))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            offenders.extend(sweep_local_paths(v, f"{prefix}[{i}]", base_dir))
    elif isinstance(node, str):
        if looks_like_local_path(node, base_dir):
            offenders.append((prefix, node))
        else:
            # backstop for inline flags hiding inside prompt strings
            for m in INLINE_CTRL_IMG_RE.finditer(node):
                value = m.group(2).strip()
                if looks_like_local_path(value, base_dir):
                    offenders.append((f"{prefix} (--ctrl_img)", value))
    return offenders


# ---------------------------------------------------------------------------
# Dataset scan (R1 caption coverage, R5 exclusion report, fingerprint)
# ---------------------------------------------------------------------------

def _is_excluded(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in contract.UPLOAD_EXCLUDES)


def scan_dataset(folder: str, check_captions: bool = True) -> DatasetReport:
    report = DatasetReport(folder=folder)
    image_stems = set()
    caption_stems = set()
    for root, dirs, files in os.walk(folder):
        rel_root = os.path.relpath(root, folder)
        kept_dirs = []
        for d in dirs:
            if _is_excluded(d):
                report.excluded.append(os.path.normpath(os.path.join(rel_root, d)) + '/')
            else:
                kept_dirs.append(d)
        dirs[:] = kept_dirs
        for f in files:
            rel = os.path.normpath(os.path.join(rel_root, f))
            if _is_excluded(f):
                report.excluded.append(rel)
                continue
            report.file_count += 1
            try:
                report.total_bytes += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
            stem, ext = os.path.splitext(rel)
            if ext.lower() in IMAGE_EXTS:
                report.image_count += 1
                image_stems.add(stem)
            elif ext.lower() == '.txt':
                report.caption_count += 1
                caption_stems.add(stem)
    if check_captions:
        report.uncaptioned = sorted(image_stems - caption_stems)
    report.excluded.sort()
    return report


# ---------------------------------------------------------------------------
# Derived config + manifest
# ---------------------------------------------------------------------------

def derived_config_path(run_name: str, base_dir: str = ".") -> str:
    return os.path.join(contract.local_run_dir(run_name, base_dir),
                        contract.DERIVED_CONFIG_FILE)


def dump_derived_config(config: dict, path: str) -> str:
    """Deterministic YAML dump (idempotent bytes); returns the sha256 hex."""
    text = yaml.safe_dump(
        config, sort_keys=False, default_flow_style=False,
        allow_unicode=True, width=10000)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _get_in(node, *keys):
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_preflight(config_path: str, run_name: str = None, base_dir: str = ".",
                  allow_uncaptioned: bool = False) -> PreflightResult:
    # R28: run names flow into root-executed shell strings — validate FIRST.
    if run_name is not None:
        contract.validate_run_name(run_name)

    config = load_source_config(config_path, base_dir)

    job_name = _get_in(config, 'config', 'name')
    if run_name is None:
        if not job_name:
            raise PreflightError("config file must have a config.name key")
        run_name = contract.validate_run_name(job_name)

    if job_name:
        config = replace_name_tag(config, job_name)
    validate_structure(config)
    job_name = config['config']['name']

    # The job name flows into remote paths, the checkpoint grammar, and
    # root-executed shell strings exactly like the run name — validate it
    # UNCONDITIONALLY (#12), even when --run-name supplied a valid override.
    try:
        contract.validate_run_name(job_name)
    except ValueError as e:
        raise PreflightError(
            f"config.name is not usable as a job name: {e}") from None

    derived = copy.deepcopy(config)
    warnings = []
    remapper = _Remapper(run_name, base_dir)
    for i, process in enumerate(derived['config']['process']):
        if not isinstance(process, dict):
            continue
        prefix = f"config.process[{i}]"
        remapper.apply(process, prefix)
        enforce_invariants(process, prefix, remapper.changes, warnings)
    changes = remapper.changes

    # closing generic sweep over the whole derived config (R3 backstop)
    offenders = sweep_local_paths(derived, "", base_dir)
    if offenders:
        listing = "\n".join(f"  {dotted}: {value!r}" for dotted, value in offenders)
        raise PreflightError(
            "derived config still contains local filesystem paths not handled "
            f"by the remap table:\n{listing}\n"
            "add the key to scripts/remote/preflight.py REMAP_TABLE or fix the config")

    # dataset scans: caption coverage + exclusion report + fingerprint
    dataset_reports = []
    for dotted, local_abs, is_caption_dataset in remapper.dataset_dirs:
        report = scan_dataset(local_abs, check_captions=is_caption_dataset)
        dataset_reports.append(report)
        if is_caption_dataset and report.image_count == 0:
            raise PreflightError(
                f"{dotted}: no images ({'/'.join(sorted(IMAGE_EXTS))}) found "
                f"in {local_abs}")
        if report.uncaptioned:
            stems = ", ".join(report.uncaptioned)
            if allow_uncaptioned:
                warnings.append(
                    f"{dotted}: {len(report.uncaptioned)} of "
                    f"{report.image_count} images have no .txt sidecar: {stems}")
            else:
                raise PreflightError(
                    f"{dotted}: {len(report.uncaptioned)} of "
                    f"{report.image_count} images have no .txt sidecar: {stems} "
                    "(pass allow_uncaptioned to proceed anyway)")
        if report.excluded:
            warnings.append(
                f"{dotted}: transport will exclude {len(report.excluded)} "
                f"file(s)/dir(s) ({', '.join(contract.UPLOAD_EXCLUDES)}): "
                f"{', '.join(report.excluded)}")

    # write the derived config (source is NEVER touched) and the manifest
    derived_path = derived_config_path(run_name, base_dir)
    config_hash = dump_derived_config(derived, derived_path)

    try:
        m = manifest.load(run_name, base_dir)
    except manifest.ManifestNotFoundError:
        m = manifest.RunManifest(run_name=run_name)
    m.state = contract.RunState.PREFLIGHTED.value
    m.job_name = job_name
    m.config_hash = config_hash
    process0 = derived['config']['process'][0]
    m.total_steps = _get_in(process0, 'train', 'steps')
    m.save_every = _get_in(process0, 'save', 'save_every')
    m.sample_every = _get_in(process0, 'sample', 'sample_every')
    sample_items = _get_in(process0, 'sample', 'samples')
    if sample_items is None:
        sample_items = _get_in(process0, 'sample', 'prompts') or []
    m.prompt_count = len(sample_items)
    m.dataset_file_count = sum(r.file_count for r in dataset_reports)
    m.dataset_total_bytes = sum(r.total_bytes for r in dataset_reports)
    manifest.save(m, base_dir)

    return PreflightResult(
        run_name=run_name,
        job_name=job_name,
        derived_config_path=derived_path,
        config_hash=config_hash,
        upload_set=remapper.upload_set,
        warnings=warnings,
        changes=changes,
        dataset_reports=dataset_reports,
    )
