from collections import OrderedDict
from toolkit.lycoris_utils import extract_diff
from .BaseExtractProcess import BaseExtractProcess


mode_dict = {
    'fixed': {
        'dim': 64,
        'type': int
    },
    'threshold': {
        'dim': 0,
        'type': float
    },
    'ratio': {
        'dim': 0.5,
        'type': float
    },
    'quantile': {
        'dim': 0.5,
        'type': float
    }
}

CLAMP_QUANTILE = 0.99
MIN_DIFF = 1e-6


class ExtractLoraProcess(BaseExtractProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.mode = self.get_conf('mode', 'fixed')

        # set modes
        if self.mode not in list(mode_dict.keys()):
            raise ValueError(f"Unknown mode: {self.mode}")
        self.dim = self.get_conf('dim', mode_dict[self.mode]['dim'], as_type=mode_dict[self.mode]['type'])
        self.use_sparse_bias = self.get_conf('use_sparse_bias', False)
        self.sparsity = self.get_conf('sparsity', 0.98)

    def run(self):
        super().run()
        print(f"Running process: {self.mode}, dim: {self.dim}")

        state_dict, extract_diff_meta = extract_diff(
            self.job.model_base,
            self.job.model_extract,
            self.mode,
            self.dim,
            0,
            self.job.device,
            self.use_sparse_bias,
            self.sparsity,
            small_conv=False,
            linear_only=True,
            extract_unet=self.extract_unet,
            extract_text_encoder=self.extract_text_encoder
        )

        self.add_meta(extract_diff_meta)
        self.save(state_dict)

    def get_output_path(self, prefix=None, suffix=None):
        if suffix is None:
            suffix = f"_{self.dim}"
        return super().get_output_path(prefix, suffix)
