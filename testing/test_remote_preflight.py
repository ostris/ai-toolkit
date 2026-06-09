"""Tests for scripts/remote/preflight.py (U2).

Run directly: python testing/test_remote_preflight.py

All fixtures are synthetic tempfile trees — no network, no torch, no real
datasets. The happy-path fixture mirrors the shape of
config/examples/train_lora_flux2_klein_9b_balfua_style_v3.yaml.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.remote import contract, manifest, preflight


def write_yaml(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_dataset(folder, n_images=4, n_captions=None, ext='.jpg'):
    """Fake dataset: tiny image files + .txt sidecars."""
    os.makedirs(folder, exist_ok=True)
    if n_captions is None:
        n_captions = n_images
    for i in range(n_images):
        with open(os.path.join(folder, f"img_{i:02d}{ext}"), 'wb') as f:
            f.write(b'fakeimagebytes')
        if i < n_captions:
            with open(os.path.join(folder, f"img_{i:02d}.txt"), 'w') as f:
                f.write(f"caption {i}")
    return folder


class PreflightTestCase(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="aitk-preflight-test-")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.base, *parts)

    def make_ctrl_img(self, name="portrait.jpg"):
        ctrl_dir = self.path("ctrl_imgs")
        os.makedirs(ctrl_dir, exist_ok=True)
        p = os.path.join(ctrl_dir, name)
        with open(p, 'wb') as f:
            f.write(b'fakectrlimage')
        return p

    def balfua_like_config(self, name="fixture_run", dataset_dir=None,
                           ctrl_img=None, **overrides):
        """Fixture modeled on train_lora_flux2_klein_9b_balfua_style_v3.yaml."""
        if dataset_dir is None:
            dataset_dir = make_dataset(self.path("data", "balfua"))
        if ctrl_img is None:
            ctrl_img = self.make_ctrl_img()
        config = {
            'job': 'extension',
            'config': {
                'name': name,
                'process': [{
                    'type': 'sd_trainer',
                    'training_folder': 'output',
                    'device': 'cuda:0',
                    'trigger_word': 'b4lf64',
                    'network': {'type': 'lora', 'linear': 32, 'linear_alpha': 32},
                    'save': {'dtype': 'float16', 'save_every': 250,
                             'max_step_saves_to_keep': 10, 'push_to_hub': False},
                    'datasets': [{
                        'folder_path': dataset_dir,
                        'caption_ext': 'txt',
                        'caption_dropout_rate': 0.1,
                        'cache_latents_to_disk': True,
                        'resolution': [512, 768, 1024],
                    }],
                    'train': {'batch_size': 1, 'steps': 2500,
                              'noise_scheduler': 'flowmatch',
                              'optimizer': 'adamw8bit', 'lr': 1.5e-4,
                              'dtype': 'bf16'},
                    'logging': {'log_every': 1, 'use_ui_logger': True},
                    'model': {'name_or_path': 'black-forest-labs/FLUX.2-klein-base-9B',
                              'arch': 'flux2_klein_9b', 'quantize': False},
                    'sample': {
                        'sampler': 'flowmatch', 'sample_every': 250,
                        'width': 1024, 'height': 1024,
                        'samples': [
                            {'prompt': 'a fox, glowing forest, dark fantasy, b4lf64'},
                            {'prompt': 'a wolf, dark fantasy, b4lf64'},
                            {'prompt': 'a fox'},
                            {'prompt': 'restyle this, glowing forest, dark fantasy, '
                                       f'b4lf64 --ctrl_img {ctrl_img}'},
                        ],
                        'neg': '', 'seed': 42, 'guidance_scale': 4,
                        'sample_steps': 20,
                    },
                }],
            },
            'meta': {'name': '[name]', 'version': '3.0'},
        }
        process = config['config']['process'][0]
        for key, value in overrides.items():
            if value is None:
                process.pop(key, None)
            else:
                process[key] = value
        return config

    def write_config(self, config, filename="fixture.yaml"):
        return write_yaml(self.path(filename), config)

    def run_pf(self, config_path, **kwargs):
        kwargs.setdefault('base_dir', self.base)
        return preflight.run_preflight(config_path, **kwargs)


class TestHappyPath(PreflightTestCase):
    def test_balfua_like_config_remaps_and_writes_derived(self):
        dataset_dir = make_dataset(self.path("data", "balfua"))
        ctrl_img = self.make_ctrl_img()
        config_path = self.write_config(
            self.balfua_like_config(dataset_dir=dataset_dir, ctrl_img=ctrl_img))
        with open(config_path, 'rb') as f:
            source_bytes_before = f.read()

        result = self.run_pf(config_path)

        # source NEVER mutated — byte-identical after
        with open(config_path, 'rb') as f:
            self.assertEqual(f.read(), source_bytes_before)

        # derived file written, hash matches its bytes
        self.assertTrue(os.path.isfile(result.derived_config_path))
        with open(result.derived_config_path, 'rb') as f:
            derived_bytes = f.read()
        self.assertEqual(result.config_hash,
                         hashlib.sha256(derived_bytes).hexdigest())

        derived = yaml.safe_load(derived_bytes)
        process = derived['config']['process'][0]
        run = result.run_name
        # the three remaps from the plan's happy path
        self.assertEqual(process['training_folder'],
                         contract.remote_training_folder(run))
        self.assertEqual(process['datasets'][0]['folder_path'],
                         f"{contract.remote_dataset_dir(run)}/balfua")
        edit_prompt = process['sample']['samples'][3]['prompt']
        self.assertIn(f"--ctrl_img {contract.remote_ctrl_dir(run)}/portrait.jpg",
                      edit_prompt)
        self.assertNotIn(self.base, edit_prompt)

        # upload set carries both the dataset dir and the ctrl image
        locals_uploaded = dict(result.upload_set)
        self.assertIn(dataset_dir, locals_uploaded)
        self.assertEqual(locals_uploaded[dataset_dir],
                         f"{contract.remote_dataset_dir(run)}/balfua")
        self.assertIn(ctrl_img, locals_uploaded)
        self.assertEqual(locals_uploaded[ctrl_img],
                         f"{contract.remote_ctrl_dir(run)}/portrait.jpg")

        # [name] tag expanded like toolkit/config.py preprocess
        self.assertEqual(derived['meta']['name'], 'fixture_run')

    def test_manifest_written_with_cadences_and_state(self):
        config_path = self.write_config(self.balfua_like_config())
        result = self.run_pf(config_path)
        m = manifest.load(result.run_name, base_dir=self.base)
        self.assertEqual(m.state, contract.RunState.PREFLIGHTED.value)
        self.assertEqual(m.job_name, 'fixture_run')
        self.assertEqual(m.config_hash, result.config_hash)
        self.assertEqual(m.total_steps, 2500)
        self.assertEqual(m.save_every, 250)
        self.assertEqual(m.sample_every, 250)
        self.assertEqual(m.prompt_count, 4)
        # fingerprint: 4 images + 4 captions in the fixture dataset
        self.assertEqual(m.dataset_file_count, 8)
        self.assertGreater(m.dataset_total_bytes, 0)

    def test_exclusions_reported_not_uploaded_in_fingerprint(self):
        dataset_dir = make_dataset(self.path("data", "balfua"))
        # AppleDouble junk + latent cache must be excluded and reported
        with open(os.path.join(dataset_dir, "._img_00.jpg"), 'wb') as f:
            f.write(b'appledouble')
        with open(os.path.join(dataset_dir, ".DS_Store"), 'wb') as f:
            f.write(b'dsstore')
        cache = os.path.join(dataset_dir, "_latent_cache")
        os.makedirs(cache)
        with open(os.path.join(cache, "x.safetensors"), 'wb') as f:
            f.write(b'latents' * 100)
        config_path = self.write_config(
            self.balfua_like_config(dataset_dir=dataset_dir))
        result = self.run_pf(config_path)
        report = result.dataset_reports[0]
        self.assertEqual(report.image_count, 4)        # ._* not counted
        self.assertEqual(report.file_count, 8)         # exclusions skipped
        self.assertIn("._img_00.jpg", report.excluded)
        self.assertIn(".DS_Store", report.excluded)
        self.assertTrue(any("_latent_cache" in e for e in report.excluded))
        self.assertTrue(any("transport will exclude" in w for w in result.warnings))


class TestFullKeyCoverage(PreflightTestCase):
    def test_every_r3_key_remapped_and_collected(self):
        run = "coverage_run"
        dirs = {}
        for key in ('folder_path', 'control_path', 'control_path_1',
                    'control_path_2', 'control_path_3', 'mask_path',
                    'unconditional_path', 'clip_image_path'):
            dirs[key] = make_dataset(self.path("data", f"d_{key}"))
        files = {}
        for key in ('ctrl_img', 'ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3',
                    'test_img_path'):
            files[key] = self.make_ctrl_img(f"{key}.png")
        for key in ('lora_path', 'pretrained_lora_path',
                    'assistant_lora_path', 'inference_lora_path'):
            p = self.path(f"{key}.safetensors")
            with open(p, 'wb') as f:
                f.write(b'fakeweights')
            files[key] = p

        config = {
            'job': 'extension',
            'config': {
                'name': run,
                'process': [{
                    'type': 'sd_trainer',
                    'training_folder': 'output',
                    'datasets': [{k: v for k, v in dirs.items()}],
                    'train': {'steps': 100},
                    'save': {'save_every': 50},
                    'adapter': {'type': 'control_net',
                                'test_img_path': files['test_img_path']},
                    'model': {'name_or_path': 'org/some-model',
                              'lora_path': files['lora_path'],
                              'pretrained_lora_path': files['pretrained_lora_path'],
                              'assistant_lora_path': files['assistant_lora_path'],
                              'inference_lora_path': files['inference_lora_path']},
                    'sample': {
                        'sample_every': 50,
                        'samples': [{
                            'prompt': 'edit it',
                            'ctrl_img': files['ctrl_img'],
                            'ctrl_img_1': files['ctrl_img_1'],
                            'ctrl_img_2': files['ctrl_img_2'],
                            'ctrl_img_3': files['ctrl_img_3'],
                        }],
                    },
                }],
            },
        }
        config_path = self.write_config(config, "coverage.yaml")
        result = self.run_pf(config_path)

        derived = read_yaml(result.derived_config_path)
        process = derived['config']['process'][0]
        dataset = process['datasets'][0]
        for key, local in dirs.items():
            base = os.path.basename(local)
            self.assertEqual(dataset[key],
                             f"{contract.remote_dataset_dir(run)}/{base}",
                             f"dataset key {key} not remapped")
        sample0 = process['sample']['samples'][0]
        for key in ('ctrl_img', 'ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'):
            self.assertEqual(sample0[key],
                             f"{contract.remote_ctrl_dir(run)}/{key}.png")
        self.assertEqual(process['adapter']['test_img_path'],
                         f"{contract.remote_ctrl_dir(run)}/test_img_path.png")
        for key in ('lora_path', 'pretrained_lora_path',
                    'assistant_lora_path', 'inference_lora_path'):
            self.assertEqual(process['model'][key],
                             f"{contract.remote_ctrl_dir(run)}/{key}.safetensors")
        # every referenced local file/dir is in the upload set
        uploaded_locals = {local for local, _ in result.upload_set}
        for local in list(dirs.values()) + list(files.values()):
            self.assertIn(local, uploaded_locals)


class TestInvariants(PreflightTestCase):
    def test_use_ui_logger_absent_and_keep_5_enforced(self):
        config = self.balfua_like_config(
            logging=None,
            save={'save_every': 250, 'max_step_saves_to_keep': 5})
        result = self.run_pf(self.write_config(config))
        derived = read_yaml(result.derived_config_path)
        process = derived['config']['process'][0]
        self.assertIs(process['logging']['use_ui_logger'], True)
        self.assertEqual(process['save']['max_step_saves_to_keep'], 10000)
        self.assertTrue(any('use_ui_logger' in c for c in result.changes))
        self.assertTrue(any('max_step_saves_to_keep: 5 -> 10000' in c
                            for c in result.changes))

    def test_keep_minus_one_enforced_to_10000(self):
        # -1 is NOT keep-all: clean_up_saves() slices files[:-n], so -1
        # deletes the oldest checkpoint at every save.
        config = self.balfua_like_config(
            save={'save_every': 250, 'max_step_saves_to_keep': -1})
        result = self.run_pf(self.write_config(config))
        derived = read_yaml(result.derived_config_path)
        self.assertEqual(
            derived['config']['process'][0]['save']['max_step_saves_to_keep'],
            10000)
        self.assertTrue(any('max_step_saves_to_keep: -1 -> 10000' in c
                            for c in result.changes))


class TestFailures(PreflightTestCase):
    def test_missing_dataset_folder_fails_naming_key(self):
        config = self.balfua_like_config()
        config['config']['process'][0]['datasets'][0]['folder_path'] = \
            self.path("does", "not", "exist")
        with self.assertRaises(preflight.PreflightError) as ctx:
            self.run_pf(self.write_config(config))
        self.assertIn("datasets[0].folder_path", str(ctx.exception))

    def test_missing_inline_ctrl_img_fails(self):
        missing = self.path("nope", "portrait.jpg")
        config = self.balfua_like_config()
        config['config']['process'][0]['sample']['samples'][3]['prompt'] = \
            f"restyle this --ctrl_img {missing}"
        with self.assertRaises(preflight.PreflightError) as ctx:
            self.run_pf(self.write_config(config))
        self.assertIn("--ctrl_img", str(ctx.exception))
        self.assertIn(missing, str(ctx.exception))

    def test_malformed_yaml_clean_error_with_parse_message(self):
        bad_path = self.path("broken.yaml")
        with open(bad_path, 'w') as f:
            f.write("job: extension\nconfig: [unclosed\n  name: oops\n")
        with self.assertRaises(preflight.PreflightError) as ctx:
            self.run_pf(bad_path)
        message = str(ctx.exception)
        self.assertIn("could not parse config", message)
        self.assertIn("broken.yaml", message)
        self.assertIn("line", message)  # the yaml parser's own message


class TestSidecarCoverage(PreflightTestCase):
    def test_uncaptioned_images_fail_and_name_stems(self):
        dataset_dir = make_dataset(self.path("data", "partial"),
                                   n_images=10, n_captions=8)
        config_path = self.write_config(
            self.balfua_like_config(dataset_dir=dataset_dir))
        with self.assertRaises(preflight.PreflightError) as ctx:
            self.run_pf(config_path)
        message = str(ctx.exception)
        self.assertIn("2 of 10", message)
        self.assertIn("img_08", message)
        self.assertIn("img_09", message)

    def test_allow_uncaptioned_downgrades_to_warning(self):
        dataset_dir = make_dataset(self.path("data", "partial"),
                                   n_images=10, n_captions=8)
        config_path = self.write_config(
            self.balfua_like_config(dataset_dir=dataset_dir))
        result = self.run_pf(config_path, allow_uncaptioned=True)
        self.assertTrue(any("img_08" in w and "img_09" in w
                            for w in result.warnings))
        report = result.dataset_reports[0]
        self.assertEqual(report.uncaptioned, ["img_08", "img_09"])


class TestIdempotency(PreflightTestCase):
    def test_two_runs_identical_bytes_and_hash(self):
        config_path = self.write_config(self.balfua_like_config())
        first = self.run_pf(config_path)
        with open(first.derived_config_path, 'rb') as f:
            first_bytes = f.read()
        second = self.run_pf(config_path)
        with open(second.derived_config_path, 'rb') as f:
            second_bytes = f.read()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first.config_hash, second.config_hash)
        # no double-remap: still exactly one /workspace prefix per path
        derived = yaml.safe_load(second_bytes)
        folder = derived['config']['process'][0]['datasets'][0]['folder_path']
        self.assertEqual(folder.count('/workspace'), 1)


class TestGenericSweep(PreflightTestCase):
    def test_unknown_key_with_local_path_fails_with_dotted_path(self):
        config = self.balfua_like_config()
        config['config']['process'][0]['custom_widget_path'] = \
            "/Users/nobody/secret/widget.png"
        with self.assertRaises(preflight.PreflightError) as ctx:
            self.run_pf(self.write_config(config))
        message = str(ctx.exception)
        self.assertIn("config.process[0].custom_widget_path", message)
        self.assertIn("/Users/nobody/secret/widget.png", message)

    def test_hf_repo_id_not_flagged(self):
        # the happy-path fixture carries name_or_path:
        # black-forest-labs/FLUX.2-klein-base-9B — it must survive the sweep
        result = self.run_pf(self.write_config(self.balfua_like_config()))
        derived = read_yaml(result.derived_config_path)
        self.assertEqual(derived['config']['process'][0]['model']['name_or_path'],
                         'black-forest-labs/FLUX.2-klein-base-9B')

    def test_looks_like_local_path_classifier(self):
        f = preflight.looks_like_local_path
        self.assertTrue(f("/Users/x/y.jpg"))
        self.assertTrue(f("/Volumes/X902/Rhizome/balfua"))
        self.assertTrue(f("~/datasets/foo"))
        self.assertTrue(f("./relative/path"))
        self.assertTrue(f("../up/one"))
        self.assertFalse(f("black-forest-labs/FLUX.2-klein-base-9B"))
        self.assertFalse(f("org/name"))
        self.assertFalse(f("a fox, glowing forest"))
        self.assertFalse(f(contract.remote_dataset_dir("r") + "/balfua"))
        self.assertFalse(f("/workspace/runs/r/output"))
        # slash-shaped relative path that DOES exist locally is flagged
        existing = make_dataset(self.path("rel", "exists"))
        rel = os.path.relpath(existing, os.getcwd())
        self.assertTrue(f(rel))


class TestRunNameGuard(PreflightTestCase):
    def test_bad_name_fails_before_other_validation(self):
        config = self.balfua_like_config(name="bad name; rm -rf")
        # dataset path is ALSO broken — the name guard must fire first
        config['config']['process'][0]['datasets'][0]['folder_path'] = \
            self.path("missing")
        with self.assertRaises(ValueError) as ctx:
            self.run_pf(self.write_config(config))
        self.assertIn("invalid run name", str(ctx.exception))

    def test_explicit_run_name_validated_before_config_load(self):
        with self.assertRaises(ValueError) as ctx:
            self.run_pf(self.path("never_even_read.yaml"),
                        run_name="bad$name")
        self.assertIn("invalid run name", str(ctx.exception))


class TestHubPushWarning(PreflightTestCase):
    def test_push_to_hub_true_warns(self):
        config = self.balfua_like_config(
            save={'save_every': 250, 'push_to_hub': True})
        result = self.run_pf(self.write_config(config))
        self.assertTrue(any('push_to_hub' in w for w in result.warnings))

    def test_push_to_hub_false_no_warning(self):
        result = self.run_pf(self.write_config(self.balfua_like_config()))
        self.assertFalse(any('push_to_hub' in w for w in result.warnings))


class TestConfigResolution(PreflightTestCase):
    def test_config_dir_resolution_and_env_substitution(self):
        dataset_dir = make_dataset(self.path("data", "balfua"))
        os.makedirs(self.path("config"), exist_ok=True)
        config = self.balfua_like_config(dataset_dir=dataset_dir)
        config['config']['process'][0]['datasets'][0]['folder_path'] = \
            "${AITK_TEST_DATASET_DIR}"
        write_yaml(self.path("config", "resolved_fixture.yaml"), config)
        os.environ['AITK_TEST_DATASET_DIR'] = dataset_dir
        try:
            # bare name resolves through <base_dir>/config/ like toolkit/config.py
            result = self.run_pf("resolved_fixture.yaml")
        finally:
            del os.environ['AITK_TEST_DATASET_DIR']
        derived = read_yaml(result.derived_config_path)
        self.assertEqual(
            derived['config']['process'][0]['datasets'][0]['folder_path'],
            f"{contract.remote_dataset_dir(result.run_name)}/balfua")

    def test_unset_env_var_fails_cleanly(self):
        config = self.balfua_like_config()
        config['config']['process'][0]['datasets'][0]['folder_path'] = \
            "${AITK_TEST_UNSET_VAR_XYZ}"
        with self.assertRaises(preflight.PreflightError) as ctx:
            self.run_pf(self.write_config(config))
        self.assertIn("AITK_TEST_UNSET_VAR_XYZ", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
