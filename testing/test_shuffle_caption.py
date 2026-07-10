import unittest
from unittest.mock import patch

from toolkit.caption_utils import (
    drop_caption_tags,
    expand_secondary_separator,
    join_caption_sections,
    select_mixed_caption,
    shuffle_caption_tags,
    split_caption_at_separator,
)


class ShuffleCaptionTest(unittest.TestCase):
    def test_shuffle_caption_reorders_comma_separated_tags(self):
        with patch('toolkit.caption_utils.random.shuffle', side_effect=lambda tags: tags.reverse()):
            caption = shuffle_caption_tags('a, b, c, d, e')

        self.assertEqual(caption, 'e, d, c, b, a')

    def test_shuffle_caption_is_randomized_for_each_caption_load(self):
        orders = [
            ['b', 'd', 'a', 'e', 'c'],
            ['b', 'c', 'a', 'd', 'e'],
        ]

        def use_next_order(tags):
            tags[:] = orders.pop(0)

        with patch('toolkit.caption_utils.random.shuffle', side_effect=use_next_order):
            first = shuffle_caption_tags('a, b, c, d, e')
            second = shuffle_caption_tags('a, b, c, d, e')

        self.assertEqual(first, 'b, d, a, e, c')
        self.assertEqual(second, 'b, c, a, d, e')

    def test_separator_protects_prefix_from_tag_dropout(self):
        protected, processable, has_separator = split_caption_at_separator(
            'character name, portrait ||| red hair, blue eyes', '|||'
        )

        processable = drop_caption_tags(processable, dropout_rate=1.0)
        caption = join_caption_sections(protected, processable)

        self.assertTrue(has_separator)
        self.assertEqual(caption, 'character name, portrait')
        self.assertNotIn('|||', caption)

    def test_separator_protects_prefix_from_shuffle(self):
        protected, processable, _ = split_caption_at_separator(
            'character name, portrait ||| red hair, blue eyes', '|||'
        )

        with patch('toolkit.caption_utils.random.shuffle', side_effect=lambda tags: tags.reverse()):
            processable = shuffle_caption_tags(processable)
        caption = join_caption_sections(protected, processable)

        self.assertEqual(caption, 'character name, portrait, blue eyes, red hair')

    def test_separator_splits_only_at_first_occurrence(self):
        protected, processable, has_separator = split_caption_at_separator(
            'fixed ||| tag one ||| tag two', '|||'
        )

        self.assertTrue(has_separator)
        self.assertEqual(protected.strip(), 'fixed')
        self.assertEqual(processable.strip(), 'tag one ||| tag two')

    def test_secondary_separator_shuffles_group_as_one_unit(self):
        def use_group_order(groups):
            a, b, cde, f = groups
            groups[:] = [f, b, cde, a]

        with patch('toolkit.caption_utils.random.shuffle', side_effect=use_group_order):
            caption = shuffle_caption_tags('a, b, c;;;d;;;e, f', ';;;')
        caption = expand_secondary_separator(caption, ';;;')

        self.assertEqual(caption, 'f, b, c, d, e, a')

    def test_secondary_separator_uses_one_dropout_decision_per_group(self):
        with patch(
            'toolkit.caption_utils.random.random',
            side_effect=[0.5, 0.5, 0.05, 0.5],
        ) as random_mock:
            caption = drop_caption_tags(
                'a, b, c;;;d;;;e, f',
                dropout_rate=0.1,
                secondary_separator=';;;',
            )
        caption = expand_secondary_separator(caption, ';;;')

        self.assertEqual(random_mock.call_count, 4)
        self.assertEqual(caption, 'a, b, f')

    def test_keep_and_secondary_separators_work_together(self):
        protected, processable, _ = split_caption_at_separator(
            'fixed one, fixed two ||| a, b, c;;;d;;;e, f', '|||'
        )

        with patch('toolkit.caption_utils.random.shuffle', side_effect=lambda groups: groups.reverse()):
            processable = shuffle_caption_tags(processable, ';;;')
        processable = expand_secondary_separator(processable, ';;;')
        caption = join_caption_sections(protected, processable)

        self.assertEqual(caption, 'fixed one, fixed two, f, c, d, e, b, a')

    def test_mixed_caption_variants_use_configured_weights(self):
        weights = {
            'tags': 40,
            'nl': 30,
            'tags_nl': 20,
            'nl_tags': 10,
        }
        expected = {
            'tags': 'tag1, tag2, tag3',
            'nl': 'A person standing outside.',
            'tags_nl': 'tag1, tag2, tag3, A person standing outside.',
            'nl_tags': 'A person standing outside., tag1, tag2, tag3',
        }

        for variant, expected_caption in expected.items():
            with self.subTest(variant=variant):
                with patch(
                    'toolkit.caption_utils.random.choices',
                    return_value=[variant],
                ) as choices_mock:
                    caption = select_mixed_caption(
                        'tag1, tag2, tag3',
                        'A person standing outside.',
                        weights,
                    )

                self.assertEqual(caption, expected_caption)
                choices_mock.assert_called_once_with(
                    ['tags', 'nl', 'tags_nl', 'nl_tags'],
                    weights=[40, 30, 20, 10],
                    k=1,
                )

    def test_mixed_caption_falls_back_to_available_file(self):
        weights = {'tags': 0, 'nl': 100, 'tags_nl': 0, 'nl_tags': 0}

        with patch('toolkit.caption_utils.random.choices') as choices_mock:
            tags_only = select_mixed_caption('tag1, tag2', '', weights)
            nl_only = select_mixed_caption('', 'A person standing outside.', weights)

        self.assertEqual(tags_only, 'tag1, tag2')
        self.assertEqual(nl_only, 'A person standing outside.')
        choices_mock.assert_not_called()

    def test_mixed_caption_keeps_protected_tags_first_in_every_variant(self):
        tags = 'tag1, tag2, tag3 ||| tag4, tag5'
        nl = 'A person standing outside.'
        weights = {'tags': 40, 'nl': 30, 'tags_nl': 20, 'nl_tags': 10}
        expected = {
            'tags': 'tag1, tag2, tag3, tag4, tag5',
            'nl': 'tag1, tag2, tag3, A person standing outside.',
            'tags_nl': 'tag1, tag2, tag3, tag4, tag5, A person standing outside.',
            'nl_tags': 'tag1, tag2, tag3, A person standing outside., tag4, tag5',
        }

        for variant, expected_caption in expected.items():
            with self.subTest(variant=variant):
                with patch('toolkit.caption_utils.random.choices', return_value=[variant]):
                    selected = select_mixed_caption(tags, nl, weights, '|||')
                protected, processable, has_separator = split_caption_at_separator(selected, '|||')
                caption = join_caption_sections(protected, processable)

                self.assertTrue(has_separator)
                self.assertEqual(caption, expected_caption)

    def test_mixed_caption_protected_tags_survive_suffix_processing(self):
        weights = {'tags': 0, 'nl': 0, 'tags_nl': 0, 'nl_tags': 1}
        with patch('toolkit.caption_utils.random.choices', return_value=['nl_tags']):
            selected = select_mixed_caption(
                'tag1, tag2, tag3 ||| tag4, tag5',
                'A person standing outside.',
                weights,
                '|||',
            )
        protected, processable, _ = split_caption_at_separator(selected, '|||')

        dropped = drop_caption_tags(processable, dropout_rate=1.0)
        dropout_caption = join_caption_sections(protected, dropped)
        with patch('toolkit.caption_utils.random.shuffle', side_effect=lambda tags: tags.reverse()):
            shuffled = shuffle_caption_tags(processable)
        shuffle_caption = join_caption_sections(protected, shuffled)

        self.assertEqual(dropout_caption, 'tag1, tag2, tag3')
        self.assertEqual(
            shuffle_caption,
            'tag1, tag2, tag3, tag5, tag4, A person standing outside.',
        )

if __name__ == '__main__':
    unittest.main()
