import os
import unittest

from toolkit.ltx_only import validate_ltx_only_config


class LtxOnlyModeTests(unittest.TestCase):
    def setUp(self):
        self._old_ltx_only = os.environ.get("AITK_LTX_ONLY_MODE")
        self._old_allow_non_ltx = os.environ.get("AITK_ALLOW_NON_LTX")

    def tearDown(self):
        if self._old_ltx_only is None:
            os.environ.pop("AITK_LTX_ONLY_MODE", None)
        else:
            os.environ["AITK_LTX_ONLY_MODE"] = self._old_ltx_only
        if self._old_allow_non_ltx is None:
            os.environ.pop("AITK_ALLOW_NON_LTX", None)
        else:
            os.environ["AITK_ALLOW_NON_LTX"] = self._old_allow_non_ltx

    def test_blocks_non_ltx_training_when_enabled(self):
        os.environ["AITK_LTX_ONLY_MODE"] = "1"
        os.environ["AITK_ALLOW_NON_LTX"] = "0"
        config = {
            "job": "extension",
            "config": {
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "model": {"arch": "flux", "name_or_path": "black-forest-labs/FLUX.1-dev"},
                    }
                ]
            },
        }
        with self.assertRaises(ValueError):
            validate_ltx_only_config(config)

    def test_allows_non_ltx_when_override_enabled(self):
        os.environ["AITK_LTX_ONLY_MODE"] = "1"
        os.environ["AITK_ALLOW_NON_LTX"] = "1"
        config = {
            "job": "extension",
            "config": {
                "process": [
                    {
                        "type": "diffusion_trainer",
                        "model": {"arch": "flux", "name_or_path": "black-forest-labs/FLUX.1-dev"},
                    }
                ]
            },
        }
        validate_ltx_only_config(config)


if __name__ == "__main__":
    unittest.main()
