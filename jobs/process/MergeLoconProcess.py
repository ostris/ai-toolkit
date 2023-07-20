from collections import OrderedDict
from toolkit.lycoris_utils import extract_diff
from .BaseExtractProcess import BaseExtractProcess


class MergeLoconProcess(BaseExtractProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)

    def run(self):
        super().run()
        new_state_dict = {}
        raise NotImplementedError("This is not implemented yet")


    def get_output_path(self, prefix=None, suffix=None):
        if suffix is None:
            suffix = f"_{self.mode}_{self.linear_param}_{self.conv_param}"
        return super().get_output_path(prefix, suffix)

