import copy
import json
from collections import OrderedDict
from typing import ForwardRef


class BaseProcess:
    meta: OrderedDict

    def __init__(
            self,
            process_id: int,
            job: 'BaseJob',
            config: OrderedDict
    ):
        self.process_id = process_id
        self.job = job
        self.config = config
        self.meta = copy.deepcopy(self.job.meta)

    def get_conf(self, key, default=None, required=False, as_type=None):
        if key in self.config:
            value = self.config[key]
            if as_type is not None:
                value = as_type(value)
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.process[{self.process_id}].{key}" key')
        else:
            if as_type is not None:
                return as_type(default)
            return default

    def run(self):
        # implement in child class
        # be sure to call super().run() first incase something is added here
        pass

    def add_meta(self, additional_meta: OrderedDict):
        self.meta.update(additional_meta)


from jobs import BaseJob
