from typing import OrderedDict
from PIL import Image

from toolkit.config_modules import LoggingConfig


class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def commit(self):
        pass

    def add_log_image(self, *args, **kwargs):
        return None
    
    def finish(self):
        pass


class WandbLogger(EmptyLogger):
    image_stack = {}

    def __init__(self, project: str, run_name: str | None, config: OrderedDict) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config
    
    def start(self):
        import wandb
        run = wandb.init(project=self.project, name=self.run_name, config=self.config)
        self.run = run
        self.wandb = wandb

    def log(self, *args, **kwargs):
        self.wandb.log(*args, **kwargs, commit=False)
    
    def commit(self):
        if len(self.image_stack) > 0:
            self.wandb.log(self.image_stack, commit=False)
        self.image_stack = {}
        self.wandb.log(commit=True)

    def add_log_image(self, image: Image, id, caption: str | None = None, *args, **kwargs):
        self.image_stack[f"sample_{id}"] = self.wandb.Image(image=image, caption=caption, *args, **kwargs)
    
    def finish(self):
        self.run.finish()



def create_logger(logging_config: LoggingConfig, all_config: OrderedDict):
    if logging_config.use_wandb:
        project_name = logging_config.project_name
        run_name = logging_config.run_name
        return WandbLogger(project=project_name, run_name=run_name, config=all_config)
    else:
        return EmptyLogger()
        


