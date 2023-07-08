from collections import OrderedDict


class BaseJob:
    config: OrderedDict
    job: str
    name: str
    meta: OrderedDict

    def __init__(self, config: OrderedDict):
        if not config:
            raise ValueError('config is required')

        self.config = config['config']
        self.job = config['job']
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

    def cleanup(self):
        # if you implement this in child clas,
        # be sure to call super().cleanup() LAST
        del self
