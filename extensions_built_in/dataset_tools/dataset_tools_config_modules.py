import json
from typing import Literal, Type, TYPE_CHECKING

Host: Type = Literal['unsplash', 'pexels']

RAW_DIR = "raw"
NEW_DIR = "_tmp"
TRAIN_DIR = "train"
DEPTH_DIR = "depth"

from .tools.image_tools import Step, img_manipulation_steps


class DatasetSyncCollectionConfig:
    def __init__(self, **kwargs):
        self.host: Host = kwargs.get('host', None)
        self.collection_id: str = kwargs.get('collection_id', None)
        self.directory: str = kwargs.get('directory', None)
        self.api_key: str = kwargs.get('api_key', None)
        self.min_width: int = kwargs.get('min_width', 1024)
        self.min_height: int = kwargs.get('min_height', 1024)

        if self.host is None:
            raise ValueError("host is required")
        if self.collection_id is None:
            raise ValueError("collection_id is required")
        if self.directory is None:
            raise ValueError("directory is required")
        if self.api_key is None:
            raise ValueError(f"api_key is required: {self.host}:{self.collection_id}")


class ImageState:
    def __init__(self, **kwargs):
        self.steps_complete: list[Step] = kwargs.get('steps_complete', [])
        self.steps_to_complete: list[Step] = kwargs.get('steps_to_complete', [])

    def to_dict(self):
        return {
            'steps_complete': self.steps_complete
        }


class Rect:
    def __init__(self, **kwargs):
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.width = kwargs.get('width', 0)
        self.height = kwargs.get('height', 0)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }


class ImgInfo:
    def __init__(self, **kwargs):
        self.version: int = kwargs.get('version', None)
        self.caption: str = kwargs.get('caption', None)
        self.caption_short: str = kwargs.get('caption_short', None)
        self.poi = [Rect(**poi) for poi in kwargs.get('poi', [])]
        self.state = ImageState(**kwargs.get('state', {}))
        self._upgrade_state()
        self.force_image_process: bool = False

        self.is_dirty: bool = False

    def _upgrade_state(self):
        # upgrades older states
        if self.caption is not None and 'caption' not in self.state.steps_complete:
            self.mark_step_complete('caption')
            self.is_dirty = True
        if self.caption_short is not None and 'caption_short' not in self.state.steps_complete:
            self.mark_step_complete('caption_short')
            self.is_dirty = True

    def to_dict(self):
        return {
            'version': self.version,
            'caption': self.caption,
            'caption_short': self.caption_short,
            'poi': [poi.to_dict() for poi in self.poi],
            'state': self.state.to_dict()
        }

    def mark_step_complete(self, step: Step):
        if step not in self.state.steps_complete:
            self.state.steps_complete.append(step)
        if step in self.state.steps_to_complete:
            self.state.steps_to_complete.remove(step)
        self.is_dirty = True

    def add_step(self, step: Step):
        if step not in self.state.steps_to_complete and step not in self.state.steps_complete:
            self.state.steps_to_complete.append(step)

    def trigger_image_reprocess(self, steps):
        # remove all image manipulationf from steps_to_complete
        for step in img_manipulation_steps:
            if step in self.state.steps_to_complete:
                self.state.steps_to_complete.remove(step)
            if step in self.state.steps_complete:
                self.state.steps_complete.remove(step)
        self.force_image_process = True
        self.is_dirty = True
        # we want to keep the order passed in process file
        for step in steps:
            if step in img_manipulation_steps:
                self.add_step(step)


    def add_steps(self, steps: list[Step]):
        for stage in steps:
            self.add_step(stage)

        # update steps if we have any img processes not complete, we have to reprocess them all
        # if any steps_to_complete are in img_manipulation_steps
        # TODO check if they are in a new order now ands trigger a redo

        is_manipulating_image = any([step in img_manipulation_steps for step in self.state.steps_to_complete])
        order_has_changed = False

        if not is_manipulating_image:
            # check to see if order has changed. No need to if already redoing it. Will detect if ones are removed
            target_img_manipulation_order = [step for step in steps if step in img_manipulation_steps]
            current_img_manipulation_order = [step for step in self.state.steps_complete if
                                              step in img_manipulation_steps]
            if target_img_manipulation_order != current_img_manipulation_order:
                order_has_changed = True

        if is_manipulating_image or order_has_changed:
            self.trigger_image_reprocess(steps)


    def to_json(self):
        return json.dumps(self.to_dict())

    def set_version(self, version: int):
        if self.version != version:
            self.is_dirty = True
        self.version = version
