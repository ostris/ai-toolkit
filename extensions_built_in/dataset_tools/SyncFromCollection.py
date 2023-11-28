import os
import shutil
from collections import OrderedDict
import gc
from typing import List

import torch
from tqdm import tqdm

from .tools.dataset_tools_config_modules import DatasetSyncCollectionConfig, RAW_DIR, NEW_DIR
from .tools.sync_tools import get_unsplash_images, get_pexels_images, get_local_image_file_names, download_image, \
    get_img_paths
from jobs.process import BaseExtensionProcess


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class SyncFromCollection(BaseExtensionProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)

        self.min_width = config.get('min_width', 1024)
        self.min_height = config.get('min_height', 1024)

        # add our min_width and min_height to each dataset config if they don't exist
        for dataset_config in config.get('dataset_sync', []):
            if 'min_width' not in dataset_config:
                dataset_config['min_width'] = self.min_width
            if 'min_height' not in dataset_config:
                dataset_config['min_height'] = self.min_height

        self.dataset_configs: List[DatasetSyncCollectionConfig] = [
            DatasetSyncCollectionConfig(**dataset_config)
            for dataset_config in config.get('dataset_sync', [])
        ]
        print(f"Found {len(self.dataset_configs)} dataset configs")

    def move_new_images(self, root_dir: str):
        raw_dir = os.path.join(root_dir, RAW_DIR)
        new_dir = os.path.join(root_dir, NEW_DIR)
        new_images = get_img_paths(new_dir)

        for img_path in new_images:
            # move to raw
            new_path = os.path.join(raw_dir, os.path.basename(img_path))
            shutil.move(img_path, new_path)

        # remove new dir
        shutil.rmtree(new_dir)

    def sync_dataset(self, config: DatasetSyncCollectionConfig):
        if config.host == 'unsplash':
            get_images = get_unsplash_images
        elif config.host == 'pexels':
            get_images = get_pexels_images
        else:
            raise ValueError(f"Unknown host: {config.host}")

        results = {
            'num_downloaded': 0,
            'num_skipped': 0,
            'bad': 0,
            'total': 0,
        }

        photos = get_images(config)
        raw_dir = os.path.join(config.directory, RAW_DIR)
        new_dir = os.path.join(config.directory, NEW_DIR)
        raw_images = get_local_image_file_names(raw_dir)
        new_images = get_local_image_file_names(new_dir)

        for photo in tqdm(photos, desc=f"{config.host}-{config.collection_id}"):
            try:
                if photo.filename not in raw_images and photo.filename not in new_images:
                    download_image(photo, new_dir, min_width=self.min_width, min_height=self.min_height)
                    results['num_downloaded'] += 1
                else:
                    results['num_skipped'] += 1
            except Exception as e:
                print(f" - BAD({photo.id}): {e}")
                results['bad'] += 1
                continue
            results['total'] += 1

        return results

    def print_results(self, results):
        print(
            f" - new:{results['num_downloaded']}, old:{results['num_skipped']}, bad:{results['bad']} total:{results['total']}")

    def run(self):
        super().run()
        print(f"Syncing {len(self.dataset_configs)} datasets")
        all_results = None
        failed_datasets = []
        for dataset_config in tqdm(self.dataset_configs, desc="Syncing datasets", leave=True):
            try:
                results = self.sync_dataset(dataset_config)
                if all_results is None:
                    all_results = {**results}
                else:
                    for key, value in results.items():
                        all_results[key] += value

                self.print_results(results)
            except Exception as e:
                print(f" - FAILED: {e}")
                if 'response' in e.__dict__:
                    error = f"{e.response.status_code}: {e.response.text}"
                    print(f"   - {error}")
                    failed_datasets.append({'dataset': dataset_config, 'error': error})
                else:
                    failed_datasets.append({'dataset': dataset_config, 'error': str(e)})
                continue

        print("Moving new images to raw")
        for dataset_config in self.dataset_configs:
            self.move_new_images(dataset_config.directory)

        print("Done syncing datasets")
        self.print_results(all_results)

        if len(failed_datasets) > 0:
            print(f"Failed to sync {len(failed_datasets)} datasets")
            for failed in failed_datasets:
                print(f" - {failed['dataset'].host}-{failed['dataset'].collection_id}")
                print(f"   - ERR: {failed['error']}")
