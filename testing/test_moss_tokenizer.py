import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from transformers import Qwen2Tokenizer

from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel


class MossTokenizerTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            model_config=SimpleNamespace(name_or_path="OpenMOSS-Team/MOSS-Image-SFT")
        )
        self.offline = mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"})
        self.offline.start()
        self.addCleanup(self.offline.stop)

    @staticmethod
    def valid_tokenizer():
        tokenizer = Qwen2Tokenizer(
            vocab={"<|endoftext|>": 0, "t": 1, "e": 2, "s": 3}, merges=[]
        )
        tokenizer.chat_template = "{{ messages[0]['content'] }}"
        return tokenizer

    def test_missing_vocabulary_retries_online(self):
        incomplete = Qwen2Tokenizer()
        incomplete.chat_template = "{{ messages[0]['content'] }}"
        # This is the Transformers 5 failure: formatted text yields no IDs.
        self.assertEqual(incomplete.encode("test", add_special_tokens=False), [])
        valid = self.valid_tokenizer()
        with mock.patch.object(
            Qwen2Tokenizer, "from_pretrained", side_effect=[incomplete, valid]
        ) as loader:
            result = MossImageModel._load_tokenizer(self.model)
        self.assertIs(result, valid)
        self.assertEqual(loader.call_count, 2)
        self.assertTrue(loader.call_args_list[0].kwargs["local_files_only"])
        self.assertNotIn("local_files_only", loader.call_args_list[1].kwargs)

    def test_valid_cached_tokenizer_does_not_retry(self):
        valid = self.valid_tokenizer()
        with mock.patch.object(Qwen2Tokenizer, "from_pretrained", return_value=valid) as loader:
            self.assertIs(MossImageModel._load_tokenizer(self.model), valid)
        loader.assert_called_once()

    def test_missing_template_is_loaded_separately(self):
        tokenizer = self.valid_tokenizer()
        template = tokenizer.chat_template
        tokenizer.chat_template = None
        with tempfile.TemporaryDirectory(
            prefix=".moss-tokenizer-", dir=Path(__file__).resolve().parent
        ) as folder:
            path = Path(folder) / "chat_template.jinja"
            path.write_text(template, encoding="utf-8")
            with mock.patch.object(Qwen2Tokenizer, "from_pretrained", return_value=tokenizer):
                with mock.patch(
                    "extensions_built_in.diffusion_models.moss_image.moss_image.cached_file",
                    return_value=str(path),
                ):
                    result = MossImageModel._load_tokenizer(self.model)
        self.assertEqual(result.chat_template, template)

    def test_invalid_vocabulary_fails_at_load_in_offline_mode(self):
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with mock.patch.object(Qwen2Tokenizer, "from_pretrained", return_value=Qwen2Tokenizer()):
                with self.assertRaisesRegex(ValueError, "vocabulary is missing or invalid"):
                    MossImageModel._load_tokenizer(self.model)


if __name__ == "__main__":
    unittest.main()
