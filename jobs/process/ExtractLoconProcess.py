from collections import OrderedDict
from toolkit.lycoris_utils import extract_diff
from .BaseExtractProcess import BaseExtractProcess

mode_dict = {
    'fixed': {
        'linear': 64,
        'conv': 32,
        'type': int
    },
    'threshold': {
        'linear': 0,
        'conv': 0,
        'type': float
    },
    'ratio': {
        'linear': 0.5,
        'conv': 0.5,
        'type': float
    },
    'quantile': {
        'linear': 0.5,
        'conv': 0.5,
        'type': float
    }
}


class ExtractLoconProcess(BaseExtractProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.mode = self.get_conf('mode', 'fixed')
        self.use_sparse_bias = self.get_conf('use_sparse_bias', False)
        self.sparsity = self.get_conf('sparsity', 0.98)
        self.disable_cp = self.get_conf('disable_cp', False)

        # set modes
        if self.mode not in list(mode_dict.keys()):
            raise ValueError(f"Unknown mode: {self.mode}")
        self.linear_param = self.get_conf('linear', mode_dict[self.mode]['linear'], as_type=mode_dict[self.mode]['type'])
        self.conv_param = self.get_conf('conv', mode_dict[self.mode]['conv'], as_type=mode_dict[self.mode]['type'])

    def run(self):
        super().run()
        print(f"Running process: {self.mode}, lin: {self.linear_param}, conv: {self.conv_param}")

        state_dict, extract_diff_meta = extract_diff(
            self.job.model_base,
            self.job.model_extract,
            self.mode,
            self.linear_param,
            self.conv_param,
            self.job.device,
            self.use_sparse_bias,
            self.sparsity,
            not self.disable_cp,
            extract_unet=self.extract_unet,
            extract_text_encoder=self.extract_text_encoder
        )

        self.add_meta(extract_diff_meta)
        self.save(state_dict)

    def get_output_path(self, prefix=None, suffix=None):
        if suffix is None:
            suffix = f"_{self.mode}_{self.linear_param}_{self.conv_param}"
        return super().get_output_path(prefix, suffix)

