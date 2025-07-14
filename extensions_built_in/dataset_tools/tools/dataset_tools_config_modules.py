import json
from typing import Literal, Type, TYPE_CHECKING

Host: Type = Literal['unsplash', 'pexels']

RAW_DIR = "raw"
NEW_DIR = "_tmp"
TRAIN_DIR = "train"
DEPTH_DIR = "depth"

from .image_tools import Step, img_manipulation_steps
from .caption import caption_manipulation_steps


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
        self.caption_method = kwargs.get('caption_method', None)
        self.other_captions = kwargs.get('other_captions', {})
        self._upgrade_state()
        self.force_image_process: bool = False
        self._requested_steps: list[Step] = []

        self.is_dirty: bool = False

    def _upgrade_state(self):
        # upgrades older states
        if self.caption is not None and 'caption' not in self.state.steps_complete:
            self.mark_step_complete('caption')
            self.is_dirty = True
        if self.caption_short is not None and 'caption_short' not in self.state.steps_complete:
            self.mark_step_complete('caption_short')
            self.is_dirty = True
        if self.caption_method is None and self.caption is not None:
            # added caption method in version 2. Was all llava before that
            self.caption_method = 'llava:default'
            self.is_dirty = True

    def to_dict(self):
        return {
            'version': self.version,
            'caption_method': self.caption_method,
            'caption': self.caption,
            'caption_short': self.caption_short,
            'poi': [poi.to_dict() for poi in self.poi],
            'state': self.state.to_dict(),
            'other_captions': self.other_captions
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

    def trigger_image_reprocess(self):
        if self._requested_steps is None:
            raise Exception("Must call add_steps before trigger_image_reprocess")
        steps = self._requested_steps
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
        self._requested_steps = [step for step in steps]
        for stage in steps:
            self.add_step(stage)

        # update steps if we have any img processes not complete, we have to reprocess them all
        # if any steps_to_complete are in img_manipulation_steps

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
            self.trigger_image_reprocess()

    def set_caption_method(self, method: str):
        if self._requested_steps is None:
            raise Exception("Must call add_steps before set_caption_method")
        if self.caption_method != method:
            self.is_dirty = True
            # move previous caption method to other_captions
            if self.caption_method is not None and self.caption is not None or self.caption_short is not None:
                self.other_captions[self.caption_method] = {
                    'caption': self.caption,
                    'caption_short': self.caption_short,
                }
            self.caption_method = method
            self.caption = None
            self.caption_short = None
            # see if we have a caption from the new method
            if method in self.other_captions:
                self.caption = self.other_captions[method].get('caption', None)
                self.caption_short = self.other_captions[method].get('caption_short', None)
            else:
                self.trigger_new_caption()

    def trigger_new_caption(self):
        self.caption = None
        self.caption_short = None
        self.is_dirty = True
        # check to see if we have any steps in the complete list and move them to the to_complete list
        for step in self.state.steps_complete:
            if step in caption_manipulation_steps:
                self.state.steps_complete.remove(step)
                self.state.steps_to_complete.append(step)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def set_version(self, version: int):
        if self.version != version:
            self.is_dirty = True
        self.version = version
