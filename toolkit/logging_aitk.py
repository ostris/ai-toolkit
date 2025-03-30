from typing import OrderedDict, Optional
from PIL import Image

from toolkit.config_modules import LoggingConfig

# Base logger class
# This class does nothing, it's just a placeholder
class EmptyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass

    # start logging the training
    def start(self):
        pass
    
    # collect the log to send
    def log(self, *args, **kwargs):
        pass
    
    # send the log
    def commit(self, step: Optional[int] = None):
        pass

    # log image
    def log_image(self, *args, **kwargs):
        pass

    # finish logging
    def finish(self):
        pass

# Wandb logger class
# This class logs the data to wandb
class WandbLogger(EmptyLogger):
    def __init__(self, project: str, run_name: str | None, config: OrderedDict) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config

    def start(self):
        try:
            import wandb
        except ImportError:
            raise ImportError("Failed to import wandb. Please install wandb by running `pip install wandb`")
        
        # send the whole config to wandb
        run = wandb.init(project=self.project, name=self.run_name, config=self.config)
        self.run = run
        self._log = wandb.log # log function
        self._image = wandb.Image # image object

    def log(self, *args, **kwargs):
        # when commit is False, wandb increments the step,
        # but we don't want that to happen, so we set commit=False
        self._log(*args, **kwargs, commit=False)

    def commit(self, step: Optional[int] = None):
        # after overall one step is done, we commit the log
        # by log empty object with commit=True
        self._log({}, step=step, commit=True)

    def log_image(
        self,
        image: Image,
        id,  # sample index
        caption: str | None = None,  # positive prompt
        *args,
        **kwargs,
    ):
        # create a wandb image object and log it
        image = self._image(image, caption=caption, *args, **kwargs)
        self._log({f"sample_{id}": image}, commit=False)

    def finish(self):
        self.run.finish()

# create logger based on the logging config
def create_logger(logging_config: LoggingConfig, all_config: OrderedDict):
    if logging_config.use_wandb:
        project_name = logging_config.project_name
        run_name = logging_config.run_name
        return WandbLogger(project=project_name, run_name=run_name, config=all_config)
    else:
        return EmptyLogger()
