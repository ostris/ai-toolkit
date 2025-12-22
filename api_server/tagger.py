import logging
import os.path
from enum import Enum
from time import monotonic_ns
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as rt  # type: ignore
import pandas as pd
import torch
from PIL import Image

from api_server.tagging_models import get_model_and_labels

logger = logging.getLogger(__name__)

MODEL_ID = "wd14-vit.v1"

SUPPORTED_VIDEO_FORMATS = [".mp4", ".webm", ".gifv", ".gif"]
NANOS_PER_MS = 1_000_000


class MediaType(str, Enum):
    image = "image"
    video = "video"


class PredictionRequest:
    media_path: str
    media_type: MediaType
    general_threshold: float
    character_threshold: float

    def __init__(
        self,
        media_path: str,
        media_type: MediaType,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ):
        self.media_path = media_path
        self.media_type = media_type
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold

    @classmethod
    def new(
        cls,
        media_path: str,
        media_type: Optional[MediaType] = None,
        frame_interval: float = 0.25,
        max_frame_count: int = 50,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ):
        if media_type is None:
            _, ext = os.path.splitext(media_path)
            media_type = MediaType.video if ext in SUPPORTED_VIDEO_FORMATS else MediaType.image

        if media_type == MediaType.image:
            return cls(media_path, media_type, general_threshold, character_threshold)
        if media_type == MediaType.video:
            return VideoPredictionRequest(
                media_path,
                frame_interval,
                max_frame_count,
                general_threshold,
                character_threshold,
            )
        raise ValueError(f"Unsupported media type: {media_type}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls.new(
            data["media_path"],
            MediaType(data["media_type"]) if "media_type" in data else None,
            data.get("frame_interval", 0.25),
            data.get("max_frame_count", 50),
            data.get("general_threshold", 0.35),
            data.get("character_threshold", 0.85),
        )


class VideoPredictionRequest(PredictionRequest):
    def __init__(
        self,
        media_path: str,
        frame_interval: float,
        max_frame_count: int,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ):
        super().__init__(media_path, MediaType.video, general_threshold, character_threshold)
        self.frame_interval = frame_interval
        self.max_frame_count = max_frame_count


class PredictionResult:
    def __init__(
        self,
        name: str,
        success: bool,
        images: Optional[List[Image.Image]] = None,
        tags: Optional[Dict[str, float]] = None,
        characters: Optional[Dict[str, float]] = None,
        rating: Optional[Dict[str, float]] = None,
        message: Optional[str] = None,
    ):
        self.name = name
        self.success = success
        self.images = images
        self.tags = tags
        self.characters = characters
        self.rating = rating
        self.message = message


def monotonic_ms_since(last: int) -> Tuple[int, int]:
    now = monotonic_ns()
    return (now, (now - last) // NANOS_PER_MS)


def load_labels(labels_path: str):
    df = pd.read_csv(labels_path)

    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def resolve_device() -> torch.device:
    if torch.cuda.is_available() and "CUDAExecutionProvider" in rt.get_available_providers():
        return torch.device("cuda")
    return torch.device("cpu")


def load(
    model_uri: str = MODEL_ID,
    device: Optional[torch.device] = None,
) -> Tuple[rt.InferenceSession, Tuple[Any, list]]:
    model_path, labels_path = get_model_and_labels(model_uri)
    resolved_device = device or resolve_device()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if resolved_device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = rt.InferenceSession(model_path, providers=providers)
    labels = load_labels(labels_path)

    return session, labels


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def tag_image(
    base_image: Image.Image,
    model: rt.InferenceSession,
    labels: tuple[Any, list, list, list],
    general_threshold: float = 0.35,
    character_threshold: float = 0.85,
) -> Dict[str, Dict[str, Any]]:
    tag_names, rating_indexes, general_indexes, character_indexes = labels
    _, height, width, _ = model.get_inputs()[0].shape

    # Alpha to white
    image = base_image.convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    np_image = np.asarray(image)

    # PIL RGB to OpenCV BGR
    np_image = np_image[:, :, ::-1]

    np_image = make_square(np_image, height)
    np_image = smart_resize(np_image, height)
    np_image = np_image.astype(np.float32)
    np_image = np.expand_dims(np_image, 0)
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_name: np_image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))

    # First 4 labels are actually ratings: pick one with argmax
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # Then we have general tags: pick anywhere prediction confidence > threshold
    general_names = [labels[i] for i in general_indexes]
    general_res = {
        k.replace("_", " "): v
        for k, v in [x for x in general_names if x[1] > general_threshold]
    }
    general_res = dict(sorted(general_res.items(), key=lambda x: x[1], reverse=True))

    # Everything else is characters: pick anywhere prediction confidence > threshold
    character_names = [labels[i] for i in character_indexes]
    character_res = {
        k.replace("_", " "): v
        for k, v in [x for x in character_names if x[1] > character_threshold]
    }

    return {
        "tags": convert_numpy_types(general_res),
        "characters": convert_numpy_types(character_res),
        "rating": convert_numpy_types(rating),
    }


def extract_frames(
    video_path: str,
    interval: float = 0.25,
    max_frame_count: int = 50,
) -> Iterator[Image.Image]:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    # Get the video's frames per second
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get or calculate the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not total_frames:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    logger.info("Total frames in video: %s", total_frames)

    # Calculate the frame step based on the desired interval
    frame_step = int(fps * interval) or 1
    if max_frame_count and total_frames / frame_step > max_frame_count:
        frame_step = int(total_frames / max_frame_count)
        logger.info(
            "Frame step adjusted to %s because there would be too many frames",
            frame_step,
        )

    extracted_frame_count = 0
    # Iterate over the frames by frame_step, setting the CAP_PROP_POS_FRAMES to frame_index
    for frame_index in range(0, total_frames, frame_step):
        start = monotonic_ns()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        last, duration = monotonic_ms_since(start)
        logger.debug("Seeking to frame %s took %s ms", frame_index, duration)
        ret, frame = cap.read()
        last, duration = monotonic_ms_since(last)
        logger.debug("Reading frame %s took %s ms", frame_index, duration)
        if not ret:
            logger.error("Error: Could not read frame.")
            break

        extracted_frame_count += 1
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last, duration = monotonic_ms_since(last)
        logger.debug("Converting frame %s took %s ms", frame_index, duration)
        yield img

        if max_frame_count and extracted_frame_count >= max_frame_count:
            break

    logger.info("Extracted %s frames from video", extracted_frame_count)
    cap.release()


def combine_results(all_results):
    if len(all_results) == 1:
        # shortcut
        return all_results[0]

    # Merge results into single dict
    results = {"embeddings": []}
    for result in all_results:
        for key in result:
            if key not in results:
                results[key] = {}
            for tag in result[key]:
                if tag not in results[key]:
                    results[key][tag] = 0
                if result[key][tag] > results[key][tag]:
                    results[key][tag] = result[key][tag]

    # Reorder results[key] by confidence
    for key in results:
        if key == "embeddings":
            continue
        results[key] = dict(sorted(results[key].items(), key=lambda x: x[1], reverse=True))
    return results


def images_from_data(request: PredictionRequest) -> Iterator[Image.Image]:
    """
    Extract images from data, which can be a single image or a video.
    """
    if isinstance(request, VideoPredictionRequest):
        # Extract frames from video into memory so we can delete the file safely
        yield from extract_frames(
            request.media_path,
            request.frame_interval,
            request.max_frame_count,
        )
    else:
        yield Image.open(request.media_path)


def predict(
    request: PredictionRequest,
    model: rt.InferenceSession,
    labels: tuple[Any, list, list, list],
    device: torch.device,
) -> Generator[PredictionResult, Any, Any]:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    model.set_providers(providers)
    for i, image in enumerate(images_from_data(request)):
        if i and i % 10 == 0:
            logger.info("Processing image %s", i)
        result = tag_image(
            image,
            model,
            labels,
            general_threshold=request.general_threshold,
            character_threshold=request.character_threshold,
        )

        yield PredictionResult(
            request.media_path,
            success=True,
            images=[image],
            tags=result["tags"],
            characters=result["characters"],
            rating=result["rating"],
        )


def merge_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Merge scores with the same key by taking the maximum.
    """
    result: Dict[str, float] = {}
    for score in scores:
        if score is None:
            continue
        for key, value in score.items():
            result[key] = max(result.get(key, 0), value)
    return result


class TaggerService:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[torch.device] = None):
        self.model_id = model_id
        self.device = device or resolve_device()
        self.session, self.labels = load(model_id, device=self.device)
        logger.info("Tagger Device: %s", self.device)

    def predict_request(self, request: PredictionRequest) -> List[PredictionResult]:
        return list(predict(request, self.session, self.labels, self.device))

    def predict_inputs(self, requests: List[PredictionRequest]) -> Dict[str, Dict[str, Dict[str, float]]]:
        predictions: Dict[str, List[PredictionResult]] = {}
        for request in requests:
            start = monotonic_ns()
            result = self.predict_request(request)
            _, duration = monotonic_ms_since(start)
            logger.info(
                "Prediction of %s (%s item(s)) took %sms",
                request.media_path,
                len(result),
                duration,
            )
            predictions[request.media_path] = result

        return {
            media_path: {
                "rating": merge_scores([prediction_result.rating for prediction_result in result]),
                "tags": merge_scores([prediction_result.tags for prediction_result in result]),
            }
            for media_path, result in predictions.items()
        }

    def offload(self):
        self.session.set_providers(["CPUExecutionProvider"])
        self.device = torch.device("cpu")
        logger.info("Tagger model '%s' offloaded to CPU", self.model_id)
