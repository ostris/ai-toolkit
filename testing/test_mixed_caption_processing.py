import ast
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Dict, List, Union
import unittest
from unittest.mock import patch

from toolkit.caption_utils import (
    drop_caption_tags,
    expand_secondary_separator,
    join_caption_sections,
    select_mixed_caption_selection,
    shuffle_caption_tags,
)


def _inject_trigger_into_prompt(
    prompt,
    trigger=None,
    to_replace_list=None,
    add_if_not_present=True,
):
    """Lightweight equivalent for source-loading the caption-only method."""
    trigger = trigger or ''
    replacements = list(to_replace_list or []) + ['[name]', '[trigger]']
    output = prompt
    for replacement in set(replacements):
        output = output.replace(replacement, trigger)
    if trigger and add_if_not_present and trigger not in output:
        output = trigger + ' ' + output
    return output


def _load_mixed_caption_processor():
    """Load the pure caption method without importing the heavy dataloader stack."""
    source_path = Path(__file__).resolve().parents[1] / 'toolkit' / 'dataloader_mixins.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'CaptionProcessingDTOMixin'
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == '_get_mixed_caption'
    )
    namespace = {
        'drop_caption_tags': drop_caption_tags,
        'expand_secondary_separator': expand_secondary_separator,
        'inject_trigger_into_prompt': _inject_trigger_into_prompt,
        'join_caption_sections': join_caption_sections,
        'random': random,
        'shuffle_caption_tags': shuffle_caption_tags,
    }
    module = ast.Module(body=[method_node], type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['_get_mixed_caption']


def _load_caption_loader():
    """Load the sidecar caption loader without importing the heavy dataloader stack."""
    source_path = Path(__file__).resolve().parents[1] / 'toolkit' / 'dataloader_mixins.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'CaptionProcessingDTOMixin'
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == 'load_caption'
    )
    namespace = {
        'Union': Union,
        'clean_caption': lambda caption: caption.strip(),
        'os': os,
        'select_mixed_caption_selection': select_mixed_caption_selection,
    }
    module = ast.Module(body=[method_node], type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['load_caption']


PROCESS_MIXED_CAPTION = _load_mixed_caption_processor()
LOAD_CAPTION = _load_caption_loader()


def _load_dataset_config_class():
    """Load DatasetConfig without importing the training dependency stack."""
    source_path = Path(__file__).resolve().parents[1] / 'toolkit' / 'config_modules.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'DatasetConfig'
    )
    namespace = {
        'ControlTypes': str,
        'Dict': Dict,
        'GuidanceType': str,
        'List': List,
        'Union': Union,
        'math': math,
        'os': os,
    }
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['DatasetConfig']


DATASET_CONFIG = _load_dataset_config_class()


class MixedCaptionProcessingTest(unittest.TestCase):
    def _make_item(self, variant):
        weights = {'tags': 40, 'nl': 30, 'tags_nl': 20, 'nl_tags': 10}
        with patch('toolkit.caption_utils.random.choices', return_value=[variant]):
            selection = select_mixed_caption_selection(
                'fixed ||| a, b;;;c, d',
                'A woman, standing outdoors.',
                weights,
                '|||',
            )
        config = SimpleNamespace(
            token_dropout_rate=0.5,
            cache_text_embeddings=False,
            keep_tokens=0,
            secondary_separator=';;;',
            random_triggers=[],
            random_triggers_max=1,
            shuffle_caption=True,
        )
        return SimpleNamespace(
            mixed_caption_selection=selection,
            dataset_config=config,
        )

    def test_shuffle_and_dropout_only_transform_tags_in_every_mixed_variant(self):
        expected = {
            'tags': 'fixed, b, c, a',
            'nl': 'fixed, A woman, standing outdoors.',
            'tags_nl': 'fixed, b, c, a, A woman, standing outdoors.',
            'nl_tags': 'fixed, A woman, standing outdoors., b, c, a',
        }

        for variant, expected_caption in expected.items():
            with self.subTest(variant=variant):
                item = self._make_item(variant)
                with patch(
                    'toolkit.caption_utils.random.random',
                    side_effect=[0.9, 0.9, 0.1],
                ), patch(
                    'toolkit.caption_utils.random.shuffle',
                    side_effect=lambda groups: groups.reverse(),
                ):
                    caption = PROCESS_MIXED_CAPTION(item)

                self.assertEqual(caption, expected_caption)

    def test_trigger_placeholder_in_nl_is_replaced_without_moving_nl(self):
        item = self._make_item('nl')
        item.dataset_config.token_dropout_rate = 1.0

        with patch('toolkit.caption_utils.random.shuffle', side_effect=lambda groups: groups.reverse()):
            caption = PROCESS_MIXED_CAPTION(
                item,
                trigger='subject',
                to_replace_list=['A woman'],
                add_if_not_present=True,
            )

        self.assertEqual(caption, 'fixed, subject, standing outdoors.')

    def test_missing_mixed_caption_still_receives_required_trigger(self):
        weights = {'tags': 40, 'nl': 30, 'tags_nl': 20, 'nl_tags': 10}
        selection = select_mixed_caption_selection('', '', weights, '|||')
        item = SimpleNamespace(
            mixed_caption_selection=selection,
            dataset_config=SimpleNamespace(
                token_dropout_rate=1.0,
                cache_text_embeddings=False,
                keep_tokens=0,
                secondary_separator=';;;',
                random_triggers=[],
                random_triggers_max=1,
                shuffle_caption=True,
            ),
        )

        caption = PROCESS_MIXED_CAPTION(
            item,
            trigger='subject',
            add_if_not_present=True,
        )

        self.assertEqual(caption, 'subject')

    def test_tags_nl_trigger_stays_before_random_tags_after_full_dropout(self):
        weights = {'tags': 0, 'nl': 0, 'tags_nl': 1, 'nl_tags': 0}
        with patch('toolkit.caption_utils.random.choices', return_value=['tags_nl']):
            selection = select_mixed_caption_selection(
                'tag one',
                'Natural language caption.',
                weights,
            )
        item = SimpleNamespace(
            mixed_caption_selection=selection,
            dataset_config=SimpleNamespace(
                token_dropout_rate=1.0,
                cache_text_embeddings=False,
                keep_tokens=0,
                secondary_separator=None,
                random_triggers=['random tag'],
                random_triggers_max=1,
                shuffle_caption=False,
            ),
        )

        caption = PROCESS_MIXED_CAPTION(
            item,
            trigger='subject',
            add_if_not_present=True,
        )

        self.assertEqual(caption, 'subject, random tag, Natural language caption.')

    def test_separator_only_tags_caption_is_semantically_empty(self):
        weights = {'tags': 40, 'nl': 30, 'tags_nl': 20, 'nl_tags': 10}

        selection = select_mixed_caption_selection('|||', '', weights, '|||')

        self.assertEqual(selection.render(), '')
        self.assertIsNone(selection.keep_tokens_separator)

    def test_mixed_weights_reject_non_finite_and_non_numeric_values(self):
        invalid_weights = (
            {'tags': float('nan')},
            {'tags': float('inf')},
            {'tags': float('-inf')},
            {'tags': 'not-a-number'},
            {
                'tags': 1e308,
                'nl': 1e308,
                'tags_nl': 1e308,
                'nl_tags': 1e308,
            },
        )

        for mixed_weights in invalid_weights:
            with self.subTest(mixed_weights=mixed_weights):
                with self.assertRaisesRegex(ValueError, 'finite numeric values'):
                    DATASET_CONFIG(caption_mode='mixed', mixed_weights=mixed_weights)

    def test_mixed_variant_is_reselected_each_time_item_is_loaded(self):
        with self.subTest('sidecar sources are reused without pinning the variant'):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as temp_dir:
                image_path = Path(temp_dir) / 'image.png'
                image_path.write_bytes(b'not needed by caption loader')
                image_path.with_suffix('.txt').write_text('tag one, tag two', encoding='utf-8')
                (Path(temp_dir) / 'image_nl.txt').write_text(
                    'A natural language caption.',
                    encoding='utf-8',
                )

                item = SimpleNamespace(
                    path=str(image_path),
                    raw_caption=None,
                    raw_caption_short=None,
                    caption=None,
                    caption_short=None,
                    mixed_caption_selection=None,
                    mixed_caption_sources=None,
                    dataset_config=SimpleNamespace(
                        caption_ext='.txt',
                        caption_mode='mixed',
                        mixed_weights={
                            'tags': 40,
                            'nl': 30,
                            'tags_nl': 20,
                            'nl_tags': 10,
                        },
                        keep_tokens_separator=None,
                        default_caption=None,
                        use_short_captions=False,
                    ),
                )
                item.get_caption = lambda short_caption=False: item.raw_caption

                with patch(
                    'toolkit.caption_utils.random.choices',
                    side_effect=[['tags'], ['nl']],
                ) as choices_mock:
                    LOAD_CAPTION(item)
                    first_caption = item.caption
                    LOAD_CAPTION(item)
                    second_caption = item.caption

                self.assertEqual(first_caption, 'tag one, tag two')
                self.assertEqual(second_caption, 'A natural language caption.')
                self.assertEqual(choices_mock.call_count, 2)


if __name__ == '__main__':
    unittest.main()
