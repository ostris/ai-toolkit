import unittest

from toolkit.trigger_data_split import (
    assert_no_split_leakage,
    create_data_split_manifest,
    filter_items_for_split,
    heldout_item_count,
    load_data_split_manifest,
    persist_data_split_manifest,
)


class TriggerDataSplitTest(unittest.TestCase):
    def test_fixed_seed_is_image_level_and_reproducible(self):
        item_ids = ['a/image.png', 'b/image.png', 'c/image.png', 'd/image.png']
        first = create_data_split_manifest(item_ids, seed=42, heldout_fraction=0.1)
        second = create_data_split_manifest(reversed(item_ids), seed=42, heldout_fraction=0.1)
        self.assertEqual(first, second)
        self.assertEqual(len(first.heldout_item_ids), 1)
        self.assertEqual(set(first.train_item_ids) | set(first.heldout_item_ids), set(item_ids))

    def test_caption_pairs_follow_image_side(self):
        manifest = create_data_split_manifest(['one.png', 'two.png', 'three.png'], seed=5)
        items = [
            {'item_id': item_id}
            for item_id in manifest.train_item_ids + manifest.heldout_item_ids
        ]
        train = filter_items_for_split(items, manifest, 'train')
        heldout = filter_items_for_split(items, manifest, 'heldout')
        self.assertTrue(set(item['item_id'] for item in train).isdisjoint(
            item['item_id'] for item in heldout
        ))

    def test_manifest_round_trip_and_stale_dataset_failure(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'split.json')
            manifest = create_data_split_manifest(['a.png', 'b.png', 'c.png'], seed=11)
            persist_data_split_manifest(manifest, path)
            self.assertEqual(load_data_split_manifest(path, item_ids=['a.png', 'b.png', 'c.png']), manifest)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                load_data_split_manifest(path, item_ids=['a.png', 'b.png', 'changed.png'])

    def test_small_dataset_rounding_keeps_both_sides(self):
        self.assertEqual(heldout_item_count(2, 0.1), 1)
        self.assertEqual(heldout_item_count(15, 0.1), 2)
        with self.assertRaisesRegex(ValueError, 'leakage'):
            assert_no_split_leakage(['same.png'], ['same.png'])


if __name__ == '__main__':
    unittest.main()
