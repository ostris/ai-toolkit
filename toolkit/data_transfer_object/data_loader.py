import os
from typing import TYPE_CHECKING, List, Union
import cv2
import torch

from PIL import Image
from PIL.ImageOps import exif_transpose

from toolkit import image_utils
from toolkit.basic import get_quick_signature_string
from toolkit.dataloader_mixins import (
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    LatentCachingFileItemDTOMixin,
    ControlFileItemDTOMixin,
    ArgBreakMixin,
    PoiFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    TextEmbeddingFileItemDTOMixin,
)
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds

if TYPE_CHECKING:
    from toolkit.config_modules import DatasetConfig

printed_messages = []


def print_once(msg):
    global printed_messages
    if msg not in printed_messages:
        print(msg)
        printed_messages.append(msg)


def _pin_nested(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.device.type == "cpu":
            return value.pin_memory()
        return value
    if isinstance(value, PromptEmbeds):
        value.text_embeds = _pin_nested(value.text_embeds)
        value.pooled_embeds = _pin_nested(value.pooled_embeds)
        value.attention_mask = _pin_nested(value.attention_mask)
        return value
    if isinstance(value, list):
        return [_pin_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pin_nested(item) for item in value)
    if isinstance(value, dict):
        return {k: _pin_nested(v) for k, v in value.items()}
    return value


def _to_device_nested(value, device, non_blocking=False):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.device == device:
            return value
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, PromptEmbeds):
        value.text_embeds = _to_device_nested(value.text_embeds, device, non_blocking=non_blocking)
        value.pooled_embeds = _to_device_nested(value.pooled_embeds, device, non_blocking=non_blocking)
        value.attention_mask = _to_device_nested(value.attention_mask, device, non_blocking=non_blocking)
        return value
    if isinstance(value, list):
        return [_to_device_nested(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device_nested(item, device, non_blocking=non_blocking) for item in value)
    if isinstance(value, dict):
        return {k: _to_device_nested(v, device, non_blocking=non_blocking) for k, v in value.items()}
    return value


class FileItemDTO(
    LatentCachingFileItemDTOMixin,
    TextEmbeddingFileItemDTOMixin,
    CaptionProcessingDTOMixin,
    ImageProcessingDTOMixin,
    ControlFileItemDTOMixin,
    InpaintControlFileItemDTOMixin,
    ClipImageFileItemDTOMixin,
    MaskFileItemDTOMixin,
    AugmentationFileItemDTOMixin,
    UnconditionalFileItemDTOMixin,
    PoiFileItemDTOMixin,
    ArgBreakMixin,
):
    def __init__(self, *args, **kwargs):
        self.path = kwargs.get("path", "")
        self.dataset_config: "DatasetConfig" = kwargs.get("dataset_config", None)
        self.is_video = self.dataset_config.num_frames > 1
        size_database = kwargs.get("size_database", {})
        dataset_root = kwargs.get("dataset_root", None)
        self.encode_control_in_text_embeddings = kwargs.get(
            "encode_control_in_text_embeddings", False
        )
        self.te_padding_side = kwargs.get("te_padding_side", "right")
        self.latent_space_version = kwargs.get("latent_space_version", "sd1")
        self.text_embedding_space_version = kwargs.get("text_embedding_space_version", "sd1")
        if dataset_root is not None:
            # remove dataset root from path
            file_key = self.path.replace(dataset_root, "")
        else:
            file_key = os.path.basename(self.path)

        file_signature = get_quick_signature_string(self.path)
        if file_signature is None:
            raise Exception("Error: Could not get file signature for {self.path}")

        use_db_entry = False
        if file_key in size_database:
            db_entry = size_database[file_key]
            if (
                db_entry is not None
                and len(db_entry) >= 3
                and db_entry[2] == file_signature
            ):
                use_db_entry = True

        if use_db_entry:
            w, h, _ = size_database[file_key]
        elif self.is_video:
            # Open the video file
            video = cv2.VideoCapture(self.path)

            # Check if video opened successfully
            if not video.isOpened():
                raise Exception(f"Error: Could not open video file {self.path}")

            # Get width and height
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w, h = width, height

            # Release the video capture object immediately
            video.release()
            size_database[file_key] = (width, height, file_signature)
        else:
            if self.dataset_config.fast_image_size:
                # original method is significantly faster, but some images are read sideways. Not sure why. Do slow method by default.
                try:
                    w, h = image_utils.get_image_size(self.path)
                except image_utils.UnknownImageFormat:
                    print_once(
                        f"Warning: Some images in the dataset cannot be fast read. "
                        + f"This process is faster for png, jpeg"
                    )
                    img = exif_transpose(Image.open(self.path))
                    w, h = img.size
            else:
                img = exif_transpose(Image.open(self.path))
                w, h = img.size
            size_database[file_key] = (w, h, file_signature)
        self.width: int = w
        self.height: int = h
        self.dataloader_transforms = kwargs.get("dataloader_transforms", None)
        super().__init__(*args, **kwargs)

        # self.caption_path: str = kwargs.get('caption_path', None)
        self.raw_caption: str = kwargs.get("raw_caption", None)
        # we scale first, then crop
        self.scale_to_width: int = kwargs.get(
            "scale_to_width", int(self.width * self.dataset_config.scale)
        )
        self.scale_to_height: int = kwargs.get(
            "scale_to_height", int(self.height * self.dataset_config.scale)
        )
        # crop values are from scaled size
        self.crop_x: int = kwargs.get("crop_x", 0)
        self.crop_y: int = kwargs.get("crop_y", 0)
        self.crop_width: int = kwargs.get("crop_width", self.scale_to_width)
        self.crop_height: int = kwargs.get("crop_height", self.scale_to_height)
        self.flip_x: bool = kwargs.get("flip_x", False)
        self.flip_y: bool = kwargs.get("flip_x", False)
        self.augments: List[str] = self.dataset_config.augments
        self.loss_multiplier: float = self.dataset_config.loss_multiplier

        self.network_weight: float = self.dataset_config.network_weight
        self.is_reg = self.dataset_config.is_reg
        self.prior_reg = self.dataset_config.prior_reg
        self.tensor: Union[torch.Tensor, None] = None
        self.audio_data = None
        self.audio_tensor = None

    def cleanup(self):
        self.tensor = None
        self.audio_data = None
        self.audio_tensor = None
        self.cleanup_latent()
        self.cleanup_text_embedding()
        self.cleanup_control()
        self.cleanup_inpaint()
        self.cleanup_clip_image()
        self.cleanup_mask()
        self.cleanup_unconditional()


class DataLoaderBatchDTO:
    def __init__(self, **kwargs):
        try:
            self.file_items: List["FileItemDTO"] = kwargs.get("file_items", None)
            is_latents_cached = self.file_items[0].is_latent_cached
            self.tensor: Union[torch.Tensor, None] = None
            self.latents: Union[torch.Tensor, None] = None
            self.control_tensor: Union[torch.Tensor, None] = None
            self.control_tensor_list: Union[List[List[torch.Tensor]], None] = None
            self.clip_image_tensor: Union[torch.Tensor, None] = None
            self.mask_tensor: Union[torch.Tensor, None] = None
            self.unaugmented_tensor: Union[torch.Tensor, None] = None
            self.unconditional_tensor: Union[torch.Tensor, None] = None
            self.unconditional_latents: Union[torch.Tensor, None] = None
            self.clip_image_embeds: Union[List[dict], None] = None
            self.clip_image_embeds_unconditional: Union[List[dict], None] = None
            self.sigmas: Union[torch.Tensor, None] = (
                None  # can be added elseware and passed along training code
            )
            self.extra_values: Union[torch.Tensor, None] = (
                torch.tensor([x.extra_values for x in self.file_items])
                if len(self.file_items[0].extra_values) > 0
                else None
            )
            if any([x.audio_data is not None for x in self.file_items]):
                # Keep per-item audio alignment across mixed batches. Missing audio stays
                # as None so model code can decide whether to synthesize a fallback.
                self.audio_data: Union[List, None] = [x.audio_data for x in self.file_items]
            else:
                self.audio_data = None
            self.audio_tensor: Union[torch.Tensor, None] = None
            self.first_frame_latents: Union[torch.Tensor, None] = None
            self.audio_latents: Union[torch.Tensor, None] = None

            # just for holding noise and preds during training
            self.audio_target: Union[torch.Tensor, None] = None
            self.audio_pred: Union[torch.Tensor, None] = None
            self.audio_loss: Union[torch.Tensor, None] = None

            if not is_latents_cached:
                # only return a tensor if latents are not cached
                self.tensor: torch.Tensor = torch.cat(
                    [x.tensor.unsqueeze(0) for x in self.file_items]
                )
            # if we have encoded latents, we concatenate them
            self.latents: Union[torch.Tensor, None] = None
            if is_latents_cached:
                # this get_latent call with trigger loading all cached items from the disk
                self.latents = torch.cat(
                    [x.get_latent().unsqueeze(0) for x in self.file_items]
                )
                if any(
                    [x._cached_first_frame_latent is not None for x in self.file_items]
                ):
                    self.first_frame_latents = torch.cat(
                        [
                            x._cached_first_frame_latent.unsqueeze(0)
                            if x._cached_first_frame_latent is not None
                            else torch.zeros_like(
                                self.file_items[0]._cached_first_frame_latent
                            ).unsqueeze(0)
                            for x in self.file_items
                        ]
                    )
                if any([x._cached_audio_latent is not None for x in self.file_items]):
                    self.audio_latents = torch.cat(
                        [
                            x._cached_audio_latent.unsqueeze(0)
                            if x._cached_audio_latent is not None
                            else torch.zeros_like(
                                self.file_items[0]._cached_audio_latent
                            ).unsqueeze(0)
                            for x in self.file_items
                        ]
                    )

            self.prompt_embeds: Union[PromptEmbeds, None] = None
            # if self.file_items[0].control_tensor is not None:
            # if any have a control tensor, we concatenate them
            if any([x.control_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_control_tensor = None
                for x in self.file_items:
                    if x.control_tensor is not None:
                        base_control_tensor = x.control_tensor
                        break
                control_tensors = []
                for x in self.file_items:
                    if x.control_tensor is None:
                        control_tensors.append(torch.zeros_like(base_control_tensor))
                    else:
                        control_tensors.append(x.control_tensor)
                self.control_tensor = torch.cat(
                    [x.unsqueeze(0) for x in control_tensors]
                )

            # handle control tensor list
            if any([x.control_tensor_list is not None for x in self.file_items]):
                self.control_tensor_list = []
                for x in self.file_items:
                    if x.control_tensor_list is not None:
                        self.control_tensor_list.append(x.control_tensor_list)
                    else:
                        raise Exception(
                            f"Could not find control tensors for all file items, missing for {x.path}"
                        )

            self.inpaint_tensor: Union[torch.Tensor, None] = None
            if any([x.inpaint_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_inpaint_tensor = None
                for x in self.file_items:
                    if x.inpaint_tensor is not None:
                        base_inpaint_tensor = x.inpaint_tensor
                        break
                inpaint_tensors = []
                for x in self.file_items:
                    if x.inpaint_tensor is None:
                        inpaint_tensors.append(torch.zeros_like(base_inpaint_tensor))
                    else:
                        inpaint_tensors.append(x.inpaint_tensor)
                self.inpaint_tensor = torch.cat(
                    [x.unsqueeze(0) for x in inpaint_tensors]
                )

            self.loss_multiplier_list: List[float] = [
                x.loss_multiplier for x in self.file_items
            ]

            if any([x.clip_image_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_clip_image_tensor = None
                for x in self.file_items:
                    if x.clip_image_tensor is not None:
                        base_clip_image_tensor = x.clip_image_tensor
                        break
                clip_image_tensors = []
                for x in self.file_items:
                    if x.clip_image_tensor is None:
                        clip_image_tensors.append(
                            torch.zeros_like(base_clip_image_tensor)
                        )
                    else:
                        clip_image_tensors.append(x.clip_image_tensor)
                self.clip_image_tensor = torch.cat(
                    [x.unsqueeze(0) for x in clip_image_tensors]
                )

            if any([x.mask_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_mask_tensor = None
                for x in self.file_items:
                    if x.mask_tensor is not None:
                        base_mask_tensor = x.mask_tensor
                        break
                mask_tensors = []
                for x in self.file_items:
                    if x.mask_tensor is None:
                        mask_tensors.append(torch.zeros_like(base_mask_tensor))
                    else:
                        mask_tensors.append(x.mask_tensor)
                self.mask_tensor = torch.cat([x.unsqueeze(0) for x in mask_tensors])

            # add unaugmented tensors for ones with augments
            if any([x.unaugmented_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_unaugmented_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unaugmented_tensor = x.unaugmented_tensor
                        break
                unaugmented_tensor = []
                for x in self.file_items:
                    if x.unaugmented_tensor is None:
                        unaugmented_tensor.append(
                            torch.zeros_like(base_unaugmented_tensor)
                        )
                    else:
                        unaugmented_tensor.append(x.unaugmented_tensor)
                self.unaugmented_tensor = torch.cat(
                    [x.unsqueeze(0) for x in unaugmented_tensor]
                )

            # add unconditional tensors
            if any([x.unconditional_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_unconditional_tensor = None
                for x in self.file_items:
                    if x.unaugmented_tensor is not None:
                        base_unconditional_tensor = x.unconditional_tensor
                        break
                unconditional_tensor = []
                for x in self.file_items:
                    if x.unconditional_tensor is None:
                        unconditional_tensor.append(
                            torch.zeros_like(base_unconditional_tensor)
                        )
                    else:
                        unconditional_tensor.append(x.unconditional_tensor)
                self.unconditional_tensor = torch.cat(
                    [x.unsqueeze(0) for x in unconditional_tensor]
                )

            if any([x.clip_image_embeds is not None for x in self.file_items]):
                self.clip_image_embeds = []
                for x in self.file_items:
                    if x.clip_image_embeds is not None:
                        self.clip_image_embeds.append(x.clip_image_embeds)
                    else:
                        raise Exception("clip_image_embeds is None for some file items")

            if any(
                [x.clip_image_embeds_unconditional is not None for x in self.file_items]
            ):
                self.clip_image_embeds_unconditional = []
                for x in self.file_items:
                    if x.clip_image_embeds_unconditional is not None:
                        self.clip_image_embeds_unconditional.append(
                            x.clip_image_embeds_unconditional
                        )
                    else:
                        raise Exception(
                            "clip_image_embeds_unconditional is None for some file items"
                        )

            if any([x.prompt_embeds is not None for x in self.file_items]):
                # find one to use as a base
                base_prompt_embeds = None
                for x in self.file_items:
                    if x.prompt_embeds is not None:
                        base_prompt_embeds = x.prompt_embeds
                        break
                prompt_embeds_list = []
                for x in self.file_items:
                    if x.prompt_embeds is None:
                        y = base_prompt_embeds
                    else:
                        y = x.prompt_embeds
                    if x.text_embedding_space_version == "zimage":
                        # z image needs to be a list if it is not already
                        if not isinstance(y.text_embeds, list):
                            y.text_embeds = [y.text_embeds]
                    prompt_embeds_list.append(y)
                padding_side = self.file_items[0].te_padding_side
                
                self.prompt_embeds = concat_prompt_embeds(prompt_embeds_list, padding_side=padding_side)

            if any([x.audio_tensor is not None for x in self.file_items]):
                # find one to use as a base
                base_audio_tensor = None
                for x in self.file_items:
                    if x.audio_tensor is not None:
                        base_audio_tensor = x.audio_tensor
                        break
                audio_tensors = []
                for x in self.file_items:
                    if x.audio_tensor is None:
                        audio_tensors.append(torch.zeros_like(base_audio_tensor))
                    else:
                        audio_tensors.append(x.audio_tensor)
                self.audio_tensor = torch.cat([x.unsqueeze(0) for x in audio_tensors])

        except Exception as e:
            print(e)
            raise e

    def get_is_reg_list(self):
        return [x.is_reg for x in self.file_items]

    def get_network_weight_list(self):
        return [x.network_weight for x in self.file_items]

    def get_caption_list(
        self, trigger=None, to_replace_list=None, add_if_not_present=True
    ):
        return [x.caption for x in self.file_items]

    def get_caption_short_list(
        self, trigger=None, to_replace_list=None, add_if_not_present=True
    ):
        return [x.caption_short for x in self.file_items]

    def cleanup(self):
        del self.latents
        del self.tensor
        del self.control_tensor
        del self.audio_tensor
        del self.audio_data
        del self.audio_target
        del self.audio_pred
        del self.audio_loss
        del self.first_frame_latents
        del self.audio_latents
        for file_item in self.file_items:
            file_item.cleanup()

    def pin_memory(self):
        # Support DataLoader(pin_memory=True) for this custom batch object.
        self.tensor = _pin_nested(self.tensor)
        self.latents = _pin_nested(self.latents)
        self.control_tensor = _pin_nested(self.control_tensor)
        self.control_tensor_list = _pin_nested(self.control_tensor_list)
        self.clip_image_tensor = _pin_nested(self.clip_image_tensor)
        self.mask_tensor = _pin_nested(self.mask_tensor)
        self.unaugmented_tensor = _pin_nested(self.unaugmented_tensor)
        self.unconditional_tensor = _pin_nested(self.unconditional_tensor)
        self.unconditional_latents = _pin_nested(self.unconditional_latents)
        self.extra_values = _pin_nested(self.extra_values)
        self.audio_tensor = _pin_nested(self.audio_tensor)
        self.first_frame_latents = _pin_nested(self.first_frame_latents)
        self.audio_latents = _pin_nested(self.audio_latents)
        self.audio_target = _pin_nested(self.audio_target)
        self.audio_pred = _pin_nested(self.audio_pred)
        self.audio_loss = _pin_nested(self.audio_loss)
        self.prompt_embeds = _pin_nested(self.prompt_embeds)
        self.clip_image_embeds = _pin_nested(self.clip_image_embeds)
        self.clip_image_embeds_unconditional = _pin_nested(self.clip_image_embeds_unconditional)
        self.sigmas = _pin_nested(self.sigmas)
        return self

    def to_device(self, device: torch.device, non_blocking: bool = False):
        self.tensor = _to_device_nested(self.tensor, device, non_blocking=non_blocking)
        self.latents = _to_device_nested(self.latents, device, non_blocking=non_blocking)
        self.control_tensor = _to_device_nested(self.control_tensor, device, non_blocking=non_blocking)
        self.control_tensor_list = _to_device_nested(self.control_tensor_list, device, non_blocking=non_blocking)
        self.clip_image_tensor = _to_device_nested(self.clip_image_tensor, device, non_blocking=non_blocking)
        self.mask_tensor = _to_device_nested(self.mask_tensor, device, non_blocking=non_blocking)
        self.unaugmented_tensor = _to_device_nested(self.unaugmented_tensor, device, non_blocking=non_blocking)
        self.unconditional_tensor = _to_device_nested(self.unconditional_tensor, device, non_blocking=non_blocking)
        self.unconditional_latents = _to_device_nested(self.unconditional_latents, device, non_blocking=non_blocking)
        self.extra_values = _to_device_nested(self.extra_values, device, non_blocking=non_blocking)
        self.audio_tensor = _to_device_nested(self.audio_tensor, device, non_blocking=non_blocking)
        self.first_frame_latents = _to_device_nested(self.first_frame_latents, device, non_blocking=non_blocking)
        self.audio_latents = _to_device_nested(self.audio_latents, device, non_blocking=non_blocking)
        self.audio_target = _to_device_nested(self.audio_target, device, non_blocking=non_blocking)
        self.audio_pred = _to_device_nested(self.audio_pred, device, non_blocking=non_blocking)
        self.audio_loss = _to_device_nested(self.audio_loss, device, non_blocking=non_blocking)
        self.prompt_embeds = _to_device_nested(self.prompt_embeds, device, non_blocking=non_blocking)
        self.clip_image_embeds = _to_device_nested(self.clip_image_embeds, device, non_blocking=non_blocking)
        self.clip_image_embeds_unconditional = _to_device_nested(
            self.clip_image_embeds_unconditional, device, non_blocking=non_blocking
        )
        self.sigmas = _to_device_nested(self.sigmas, device, non_blocking=non_blocking)
        return self

    @property
    def dataset_config(self) -> "DatasetConfig":
        if len(self.file_items) > 0:
            return self.file_items[0].dataset_config
        else:
            return None
