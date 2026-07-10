import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest


def load_dataset_helpers():
    """Load only the lightweight dataset helpers, without importing Gradio."""
    source_path = Path(__file__).resolve().parents[1] / 'flux_train_ui.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    helper_names = {'_is_text_file', '_is_image_file', 'create_dataset'}
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    namespace = {
        'json': json,
        'os': os,
        'shutil': shutil,
        'uuid': SimpleNamespace(uuid4=lambda: 'test-dataset'),
    }
    module = ast.Module(body=helper_nodes, type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace['create_dataset']


class FluxTrainUiDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.uploads = self.root / 'uploads'
        self.workdir = self.root / 'workdir'
        self.uploads.mkdir()
        self.workdir.mkdir()
        self.create_dataset = load_dataset_helpers()

    def create_from_workdir(self, files, *captions):
        previous_cwd = os.getcwd()
        try:
            os.chdir(self.workdir)
            relative_destination = self.create_dataset(files, *captions)
        finally:
            os.chdir(previous_cwd)
        return self.workdir / relative_destination

    def test_writes_edited_sidecars_and_preserves_uploaded_mixed_caption(self):
        first_image = self.uploads / 'first.bmp'
        second_image = self.uploads / 'second.gif'
        uploaded_base = self.uploads / 'first.txt'
        uploaded_nl = self.uploads / 'first_nl.txt'
        first_image.write_bytes(b'first image')
        second_image.write_bytes(b'second image')
        uploaded_base.write_text('uploaded base caption', encoding='utf-8')
        uploaded_nl.write_text('업로드된 자연어 캡션', encoding='utf-8')

        destination = self.create_from_workdir(
            [first_image, uploaded_base, uploaded_nl, second_image],
            '편집된 태그 ||| red hair;;;blue eyes',
            'AI generated caption',
        )

        self.assertEqual(
            (destination / 'first.txt').read_text(encoding='utf-8'),
            '편집된 태그 ||| red hair;;;blue eyes',
        )
        self.assertEqual(
            (destination / 'first_nl.txt').read_text(encoding='utf-8'),
            '업로드된 자연어 캡션',
        )
        self.assertEqual(
            (destination / 'second.txt').read_text(encoding='utf-8'),
            'AI generated caption',
        )

        metadata = [
            json.loads(line)
            for line in (destination / 'metadata.jsonl').read_text(encoding='utf-8').splitlines()
        ]
        self.assertEqual(
            metadata,
            [
                {
                    'file_name': 'first.bmp',
                    'prompt': '편집된 태그 ||| red hair;;;blue eyes',
                },
                {'file_name': 'second.gif', 'prompt': 'AI generated caption'},
            ],
        )

    def test_empty_textbox_overwrites_base_caption_but_not_nl_caption(self):
        image = self.uploads / 'image.tiff'
        uploaded_base = self.uploads / 'image.txt'
        uploaded_nl = self.uploads / 'image_nl.txt'
        image.write_bytes(b'image')
        uploaded_base.write_text('base caption', encoding='utf-8')
        uploaded_nl.write_text('natural language caption', encoding='utf-8')

        destination = self.create_from_workdir(
            [image, uploaded_base, uploaded_nl],
            '',
        )

        self.assertEqual((destination / 'image.txt').read_text(encoding='utf-8'), '')
        self.assertEqual(
            (destination / 'image_nl.txt').read_text(encoding='utf-8'),
            'natural language caption',
        )


if __name__ == '__main__':
    unittest.main()
