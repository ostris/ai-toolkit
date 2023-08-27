import importlib
from collections import OrderedDict
from typing import List

from jobs.process import BaseProcess


class BaseJob:

    def __init__(self, config: OrderedDict):
        if not config:
            raise ValueError('config is required')
        self.process: List[BaseProcess]

        self.config = config['config']
        self.raw_config = config
        self.job = config['job']
        self.torch_profiler = self.get_conf('torch_profiler', False)
        self.name = self.get_conf('name', required=True)
        if 'meta' in config:
            self.meta = config['meta']
        else:
            self.meta = OrderedDict()

    def get_conf(self, key, default=None, required=False):
        if key in self.config:
            return self.config[key]
        elif required:
            raise ValueError(f'config file error. Missing "config.{key}" key')
        else:
            return default

    def run(self):
        print("")
        print(f"#############################################")
        print(f"# Running job: {self.name}")
        print(f"#############################################")
        print("")
        # implement in child class
        # be sure to call super().run() first
        pass

    def load_processes(self, process_dict: dict):
        # only call if you have processes in this job type
        if 'process' not in self.config:
            raise ValueError('config file is invalid. Missing "config.process" key')
        if len(self.config['process']) == 0:
            raise ValueError('config file is invalid. "config.process" must be a list of processes')

        module = importlib.import_module('jobs.process')

        # add the processes
        self.process = []
        for i, process in enumerate(self.config['process']):
            if 'type' not in process:
                raise ValueError(f'config file is invalid. Missing "config.process[{i}].type" key')

            # check if dict key is process type
            if process['type'] in process_dict:
                if isinstance(process_dict[process['type']], str):
                    ProcessClass = getattr(module, process_dict[process['type']])
                else:
                    # it is the class
                    ProcessClass = process_dict[process['type']]
                self.process.append(ProcessClass(i, self, process))
            else:
                raise ValueError(f'config file is invalid. Unknown process type: {process["type"]}')

    def cleanup(self):
        # if you implement this in child clas,
        # be sure to call super().cleanup() LAST
        del self
