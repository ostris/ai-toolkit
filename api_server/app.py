from typing import Any, Dict, Optional, Tuple
import os
import tempfile
import requests
from io import BytesIO

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from api_server.audio_caption import DEFAULT_AUDIO_MODEL_ID, DEFAULT_AUDIO_PROMPT, AudioCaptioner
from api_server.manager import TrainingSessionManager

app = FastAPI(title='AI Toolkit Training API')

manager = TrainingSessionManager()

_caption_processor = None
_tagger_service = None
_tagger_model_path = None
_audio_captioner = None
_audio_model_path = None

JOYCAPTION_LEGACY_PROMPT = "A descriptive caption for this image:"
JOYCAPTION_LEGACY_MAX_NEW_TOKENS = 300
JOYCAPTION_LEGACY_TEMPERATURE = 0.5
JOYCAPTION_LEGACY_TOP_K = 10
FLORENCE2_DEFAULT_MAX_NEW_TOKENS = 1024


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, alias='sessionId')
    max_steps: Optional[int] = Field(default=None, alias='maxSteps')
    config: Dict[str, Any]

    class Config:
        allow_population_by_field_name = True


class AllocateStepsRequest(BaseModel):
    steps: int = Field(..., gt=0)
    timeout: Optional[float] = None


class CaptionRequest(BaseModel):
    image_path: Optional[str] = Field(default=None, description="Path to local image file")
    image_url: Optional[str] = Field(default=None, description="URL to image file")
    video_path: Optional[str] = Field(default=None, description="Path to local video file")
    video_url: Optional[str] = Field(default=None, description="URL to video file")
    model_type: Optional[str] = Field(default="florence2", description="Model type: 'florence2' or 'joycaption'")
    model_path: Optional[str] = Field(default=None, description="Custom model path (default varies by model_type)")
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate (default: 1024 for Florence2, 300 for JoyCaption)",
        gt=0,
    )
    num_beams: Optional[int] = Field(default=3, description="Number of beams for generation (Florence2 only)")
    task: Optional[str] = Field(default="<DETAILED_CAPTION>", description="Florence2 task type (Florence2 only)")
    prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for JoyCaption (default: old service style 'A descriptive caption for this image:')",
    )
    temperature: Optional[float] = Field(default=None, description="Temperature for sampling (JoyCaption only)")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter (JoyCaption only)")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter (JoyCaption only)", gt=0)
    do_sample: Optional[bool] = Field(default=True, description="Whether to sample during generation (JoyCaption only)")
    clean_output: Optional[bool] = Field(
        default=False,
        description="Apply dataset-style lowercase/comma cleaning to JoyCaption output (JoyCaption only)",
    )
    num_frames: Optional[int] = Field(default=8, description="Number of frames to extract from video")
    sample_method: Optional[str] = Field(default="uniform", description="Frame sampling method: 'uniform' or 'first'")
    combine_method: Optional[str] = Field(default="first", description="Caption combining method: 'first', 'longest', or 'combined'")
    do_audio: Optional[bool] = Field(default=False, description="Also caption audio when using video inputs")
    audio_model_path: Optional[str] = Field(
        default=None,
        description=f"Custom audio model path (default: '{DEFAULT_AUDIO_MODEL_ID}')",
    )
    audio_prompt: Optional[str] = Field(
        default=DEFAULT_AUDIO_PROMPT,
        description="Prompt for audio captioning",
    )
    audio_max_new_tokens: Optional[int] = Field(default=256, description="Maximum tokens for audio captioning", gt=0)
    audio_temperature: Optional[float] = Field(default=0.2, description="Audio captioning temperature")
    audio_top_p: Optional[float] = Field(default=0.9, description="Audio captioning top-p")
    audio_num_beams: Optional[int] = Field(default=1, description="Audio captioning num_beams", gt=0)
    audio_do_sample: Optional[bool] = Field(default=True, description="Audio captioning do_sample")
    audio_repetition_penalty: Optional[float] = Field(default=None, description="Audio captioning repetition penalty")
    audio_target_sample_rate: Optional[int] = Field(
        default=16000,
        description="Sample rate when extracting audio from video",
        gt=0,
    )
    audio_max_audio_seconds: Optional[float] = Field(
        default=None,
        description="Max duration (seconds) for extracted audio",
        gt=0,
    )

    class Config:
        allow_population_by_field_name = True


class AudioCaptionRequest(BaseModel):
    audio_path: Optional[str] = Field(default=None, description="Path to local audio file")
    audio_url: Optional[str] = Field(default=None, description="URL to audio file")
    video_path: Optional[str] = Field(default=None, description="Path to local video file (audio extracted)")
    video_url: Optional[str] = Field(default=None, description="URL to video file (audio extracted)")
    model_path: Optional[str] = Field(
        default=None,
        description=f"Custom model path (default: '{DEFAULT_AUDIO_MODEL_ID}')",
    )
    prompt: Optional[str] = Field(
        default=DEFAULT_AUDIO_PROMPT,
        description="Prompt for captioning",
    )
    max_new_tokens: Optional[int] = Field(default=256, description="Maximum tokens to generate", gt=0)
    temperature: Optional[float] = Field(default=0.2, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling parameter")
    num_beams: Optional[int] = Field(default=1, description="Number of beams for generation", gt=0)
    do_sample: Optional[bool] = Field(default=True, description="Whether to sample during generation")
    repetition_penalty: Optional[float] = Field(default=None, description="Repetition penalty")
    target_sample_rate: Optional[int] = Field(
        default=16000,
        description="Sample rate for extracted audio",
        gt=0,
    )
    max_audio_seconds: Optional[float] = Field(
        default=None,
        description="Max audio duration to process in seconds",
        gt=0,
    )

    class Config:
        allow_population_by_field_name = True


@app.post('/sessions', status_code=status.HTTP_201_CREATED)
def create_session(request: SessionCreateRequest):
    try:
        session = manager.create_session(
            request.config,
            session_id=request.session_id,
            max_steps=request.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return session.get_status()


@app.get('/sessions')
def list_sessions():
    return manager.list_status()


@app.get('/sessions/{session_id}')
def get_session(session_id: str):
    try:
        session = manager.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return session.get_status()


@app.post('/sessions/{session_id}/steps')
def allocate_steps(session_id: str, request: AllocateStepsRequest = Body(...)):
    try:
        result = manager.allocate_steps(session_id, steps=request.steps, timeout=request.timeout)
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='session not found')
    return {
        'result': result,
        'status': session.get_status(),
    }


@app.post('/sessions/{session_id}/epochs')
def run_epochs_legacy(session_id: str):
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail='Epoch-based control has been replaced. Use POST /sessions/{session_id}/steps instead.',
    )


@app.post('/sessions/{session_id}/abort', status_code=status.HTTP_202_ACCEPTED)
def abort_session(session_id: str):
    try:
        manager.abort_session(session_id)
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='session not found')
    return session.get_status()


@app.delete('/sessions/{session_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str):
    try:
        manager.delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='session not found')


@app.post('/free')
def free_vram():
    result = manager.free_all_vram()
    return {
        'status': 'success',
        'freed_sessions': result['freed_sessions'],
        'failed_sessions': result['failed_sessions'],
    }


@app.post('/vram/free')
def free_all_vram():
    global _caption_processor, _tagger_service, _tagger_model_path, _audio_captioner, _audio_model_path

    import gc
    import torch

    memory_before = _get_cuda_used_memory() / (1024 * 1024)
    result = manager.free_all_vram()

    caption_unloaded = False
    tagger_offloaded = False
    audio_unloaded = False

    if _caption_processor is not None:
        _caption_processor.unload_model()
        _caption_processor = None
        caption_unloaded = True

    if _tagger_service is not None:
        _tagger_service.offload()
        _tagger_service = None
        _tagger_model_path = None
        tagger_offloaded = True

    if _audio_captioner is not None:
        _audio_captioner.unload_model()
        _audio_captioner = None
        _audio_model_path = None
        audio_unloaded = True

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    memory_after = _get_cuda_used_memory() / (1024 * 1024)

    return {
        'status': 'success',
        'freed_sessions': result['freed_sessions'],
        'failed_sessions': result['failed_sessions'],
        'caption_unloaded': caption_unloaded,
        'tagger_offloaded': tagger_offloaded,
        'audio_unloaded': audio_unloaded,
        'memory_before': memory_before,
        'memory_after': memory_after,
        'freed': max(0.0, memory_before - memory_after),
    }


@app.get('/sessions/{session_id}/logs')
def get_logs(session_id: str, limit: Optional[int] = None):
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='session not found')
    return {
        'session_id': session_id,
        'logs': session.get_logs(limit=limit),
    }


@app.get('/sessions/{session_id}/logs/stream')
def stream_logs(session_id: str):
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='session not found')

    def iterator():
        for line in session.log_stream():
            yield line + '\n'

    return StreamingResponse(iterator(), media_type='text/plain')


def _get_caption_processor(model_type: str = "florence2", model_path: Optional[str] = None):
    global _caption_processor
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == "florence2":
        from extensions_built_in.dataset_tools.tools.florence2_utils import Florence2ImageProcessor
        ProcessorClass = Florence2ImageProcessor
        default_model_path = "microsoft/Florence-2-large-ft"
    elif model_type == "joycaption":
        from extensions_built_in.dataset_tools.tools.joycaption_utils import JoyCaptionImageProcessor
        ProcessorClass = JoyCaptionImageProcessor
        default_model_path = "fancyfeast/llama-joycaption-beta-one-hf-llava"
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'florence2' or 'joycaption'")

    target_model_path = model_path or default_model_path

    need_reload = False
    if _caption_processor is None:
        need_reload = True
    elif not isinstance(_caption_processor, ProcessorClass):
        print(f"Switching from {type(_caption_processor).__name__} to {ProcessorClass.__name__}")
        _caption_processor.unload_model()
        need_reload = True
    elif _caption_processor.model_path != target_model_path:
        print(f"Switching model from {_caption_processor.model_path} to {target_model_path}")
        _caption_processor.unload_model()
        need_reload = True

    if need_reload:
        _caption_processor = ProcessorClass(device=device, model_path=target_model_path)
        _caption_processor.load_model()

    return _caption_processor


def _get_tagger_service(model_path: Optional[str]):
    global _tagger_service, _tagger_model_path
    if _tagger_service is None or _tagger_model_path != model_path:
        if _tagger_service is not None:
            _tagger_service.offload()
        from api_server.tagger import TaggerService
        _tagger_service = TaggerService(model_path=model_path)
        _tagger_model_path = model_path
    return _tagger_service


def _get_audio_captioner(model_path: Optional[str]):
    global _audio_captioner, _audio_model_path

    target_model_path = model_path or DEFAULT_AUDIO_MODEL_ID
    if _audio_captioner is None or _audio_model_path != target_model_path:
        if _audio_captioner is not None:
            _audio_captioner.unload_model()
        _audio_captioner = AudioCaptioner(model_path=target_model_path)
        _audio_captioner.load_model()
        _audio_model_path = target_model_path

    return _audio_captioner


def _get_cuda_used_memory() -> int:
    import torch

    if not torch.cuda.is_available():
        return 0
    try:
        free, total = torch.cuda.mem_get_info()
        return total - free
    except Exception:
        return torch.cuda.memory_reserved()


def _is_video_file(file_path: str) -> bool:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    return any(file_path.lower().endswith(ext) for ext in video_extensions)


def _load_image_from_source(image_path: Optional[str], image_url: Optional[str]) -> Image:
    if image_path and image_url:
        raise ValueError("Provide either image_path or image_url, not both")

    if not image_path and not image_url:
        raise ValueError("Must provide either image_path or image_url")

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert('RGB')

    if image_url:
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")


def _load_video_from_source(video_path: Optional[str], video_url: Optional[str]) -> str:
    if video_path and video_url:
        raise ValueError("Provide either video_path or video_url, not both")

    if not video_path and not video_url:
        raise ValueError("Must provide either video_path or video_url")

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path

    if video_url:
        try:
            response = requests.get(video_url, timeout=120, stream=True)
            response.raise_for_status()

            ext = os.path.splitext(video_url.split('?')[0])[-1] or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                temp_path = tmp_file.name

            return temp_path
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download video from URL: {str(e)}")


def _load_audio_from_source(audio_path: Optional[str], audio_url: Optional[str]) -> Tuple[str, bool]:
    if audio_path and audio_url:
        raise ValueError("Provide either audio_path or audio_url, not both")

    if not audio_path and not audio_url:
        raise ValueError("Must provide either audio_path or audio_url")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return audio_path, False

    temp_path = _download_media_from_url(audio_url, "audio")
    return temp_path, True


def _extract_audio_from_video(
    video_path: str,
    target_sample_rate: Optional[int] = 16000,
    max_audio_seconds: Optional[float] = None,
) -> str:
    import av
    import numpy as np
    from scipy.io import wavfile

    container = av.open(video_path)
    try:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in video: {video_path}")

        target_rate = target_sample_rate or audio_stream.rate or 16000
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_rate)

        audio_chunks = []
        max_samples = int(target_rate * max_audio_seconds) if max_audio_seconds else None
        total_samples = 0

        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue

                resampled_frames = resampled if isinstance(resampled, list) else [resampled]
                for resampled_frame in resampled_frames:
                    frame_array = resampled_frame.to_ndarray()
                    if frame_array.ndim > 1:
                        frame_array = frame_array.reshape(-1)

                    if max_samples is not None:
                        remaining = max_samples - total_samples
                        if remaining <= 0:
                            break
                        if frame_array.size > remaining:
                            frame_array = frame_array[:remaining]

                    audio_chunks.append(frame_array)
                    total_samples += frame_array.size

            if max_samples is not None and total_samples >= max_samples:
                break

        if not audio_chunks:
            raise ValueError(f"No audio could be extracted from video: {video_path}")

        audio_data = np.concatenate(audio_chunks)
    finally:
        container.close()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        wavfile.write(tmp_file.name, target_rate, audio_data)
        return tmp_file.name


def _download_media_from_url(media_url: str, media_type: Optional[str]) -> str:
    try:
        response = requests.get(media_url, timeout=120, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download media from URL: {str(e)}")

    url_path = media_url.split('?')[0]
    ext = os.path.splitext(url_path)[-1].lower()
    if not ext:
        if media_type == "video":
            ext = ".mp4"
        elif media_type == "image":
            ext = ".png"
        elif media_type == "audio":
            ext = ".wav"
        else:
            content_type = response.headers.get("Content-Type", "")
            if "video" in content_type:
                ext = ".mp4"
            elif "image" in content_type:
                ext = ".png"
            elif "audio" in content_type:
                ext = ".wav"
            else:
                ext = ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        temp_path = tmp_file.name

    return temp_path


@app.post('/caption')
def generate_caption(request: CaptionRequest):
    global _caption_processor, _audio_captioner, _audio_model_path
    temp_paths = []
    try:
        is_video = request.video_path is not None or request.video_url is not None
        is_image = request.image_path is not None or request.image_url is not None

        if is_video and is_image:
            raise ValueError("Provide either image or video, not both")

        if not is_video and not is_image:
            raise ValueError("Must provide either image_path, image_url, video_path, or video_url")

        if request.do_audio and not is_video:
            raise ValueError("do_audio can only be used with video inputs")

        processor = _get_caption_processor(request.model_type, request.model_path)

        if is_video:
            video_path = _load_video_from_source(request.video_path, request.video_url)
            if request.video_url:
                temp_paths.append(video_path)

            if request.model_type == "joycaption":
                joy_prompt = request.prompt if request.prompt is not None else JOYCAPTION_LEGACY_PROMPT
                joy_max_new_tokens = (
                    request.max_new_tokens
                    if request.max_new_tokens is not None
                    else JOYCAPTION_LEGACY_MAX_NEW_TOKENS
                )
                joy_temperature = (
                    request.temperature
                    if request.temperature is not None
                    else JOYCAPTION_LEGACY_TEMPERATURE
                )
                joy_top_k = request.top_k if request.top_k is not None else JOYCAPTION_LEGACY_TOP_K
                joy_do_sample = request.do_sample if request.do_sample is not None else True
                if request.clean_output is not None:
                    joy_clean_output = request.clean_output
                else:
                    # Combined mode expects comma-separated elements, so default to cleaned captions.
                    joy_clean_output = request.combine_method == "combined"
                caption = processor.generate_video_caption(
                    video_path=video_path,
                    num_frames=request.num_frames,
                    sample_method=request.sample_method,
                    max_new_tokens=joy_max_new_tokens,
                    temperature=joy_temperature,
                    top_p=request.top_p,
                    top_k=joy_top_k,
                    do_sample=joy_do_sample,
                    clean_output=joy_clean_output,
                    combine_method=request.combine_method,
                    prompt=joy_prompt
                )
            else: 
                florence_max_new_tokens = (
                    request.max_new_tokens
                    if request.max_new_tokens is not None
                    else FLORENCE2_DEFAULT_MAX_NEW_TOKENS
                )
                caption = processor.generate_video_caption(
                    video_path=video_path,
                    num_frames=request.num_frames,
                    sample_method=request.sample_method,
                    max_new_tokens=florence_max_new_tokens,
                    num_beams=request.num_beams,
                    task=request.task,
                    combine_method=request.combine_method
                )

            if request.do_audio:
                if _caption_processor is not None:
                    try:
                        _caption_processor.unload_model()
                    except Exception:
                        pass
                    _caption_processor = None
                    try:
                        import gc
                        import torch

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                audio_path = _extract_audio_from_video(
                    video_path,
                    target_sample_rate=request.audio_target_sample_rate,
                    max_audio_seconds=request.audio_max_audio_seconds,
                )
                temp_paths.append(audio_path)

                captioner = _get_audio_captioner(request.audio_model_path)
                try:
                    audio_caption = captioner.generate_caption(
                        audio_path=audio_path,
                        prompt=request.audio_prompt,
                        max_new_tokens=request.audio_max_new_tokens or 256,
                        temperature=request.audio_temperature if request.audio_temperature is not None else 0.2,
                        top_p=request.audio_top_p if request.audio_top_p is not None else 0.9,
                        num_beams=request.audio_num_beams or 1,
                        do_sample=request.audio_do_sample if request.audio_do_sample is not None else True,
                        repetition_penalty=request.audio_repetition_penalty,
                    )
                finally:
                    if _audio_captioner is not None:
                        try:
                            _audio_captioner.unload_model()
                        except Exception:
                            pass
                        _audio_captioner = None
                        _audio_model_path = None
                        try:
                            import gc
                            import torch

                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                caption = f"{caption}\n\n{audio_caption}"

            source = request.video_path or request.video_url
            media_type = 'video'

        else:
            image = _load_image_from_source(request.image_path, request.image_url)

            if request.model_type == "joycaption":
                joy_prompt = request.prompt if request.prompt is not None else JOYCAPTION_LEGACY_PROMPT
                joy_max_new_tokens = (
                    request.max_new_tokens
                    if request.max_new_tokens is not None
                    else JOYCAPTION_LEGACY_MAX_NEW_TOKENS
                )
                joy_temperature = (
                    request.temperature
                    if request.temperature is not None
                    else JOYCAPTION_LEGACY_TEMPERATURE
                )
                joy_top_k = request.top_k if request.top_k is not None else JOYCAPTION_LEGACY_TOP_K
                joy_do_sample = request.do_sample if request.do_sample is not None else True
                joy_clean_output = request.clean_output if request.clean_output is not None else False
                caption = processor.generate_caption(
                    image=image,
                    prompt=joy_prompt,
                    max_new_tokens=joy_max_new_tokens,
                    temperature=joy_temperature,
                    top_p=request.top_p,
                    top_k=joy_top_k,
                    do_sample=joy_do_sample,
                    clean_output=joy_clean_output,
                )
            else:  
                florence_max_new_tokens = (
                    request.max_new_tokens
                    if request.max_new_tokens is not None
                    else FLORENCE2_DEFAULT_MAX_NEW_TOKENS
                )
                caption = processor.generate_caption(
                    image=image,
                    max_new_tokens=florence_max_new_tokens,
                    num_beams=request.num_beams,
                    task=request.task
                )

            source = request.image_path or request.image_url
            media_type = 'image'

        return {
            'caption': caption,
            'source': source,
            'model': request.model_type,
            'media_type': media_type
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating caption: {str(e)}"
        )
    finally:
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


@app.post('/audio/caption')
def generate_audio_caption(request: AudioCaptionRequest):
    temp_paths = []
    try:
        is_audio = request.audio_path is not None or request.audio_url is not None
        is_video = request.video_path is not None or request.video_url is not None

        if is_audio and is_video:
            raise ValueError("Provide either audio or video, not both")

        if not is_audio and not is_video:
            raise ValueError("Must provide audio_path, audio_url, video_path, or video_url")

        if is_audio:
            audio_path, is_temp = _load_audio_from_source(request.audio_path, request.audio_url)
            if is_temp:
                temp_paths.append(audio_path)
            source = request.audio_path or request.audio_url
            media_type = "audio"
        else:
            video_path = _load_video_from_source(request.video_path, request.video_url)
            if request.video_url:
                temp_paths.append(video_path)

            target_rate = request.target_sample_rate or 16000
            audio_path = _extract_audio_from_video(
                video_path,
                target_sample_rate=target_rate,
                max_audio_seconds=request.max_audio_seconds,
            )
            temp_paths.append(audio_path)
            source = request.video_path or request.video_url
            media_type = "video"

        captioner = _get_audio_captioner(request.model_path)
        caption = captioner.generate_caption(
            audio_path=audio_path,
            prompt=request.prompt or DEFAULT_AUDIO_PROMPT,
            max_new_tokens=request.max_new_tokens or 256,
            temperature=request.temperature if request.temperature is not None else 0.2,
            top_p=request.top_p if request.top_p is not None else 0.9,
            num_beams=request.num_beams or 1,
            do_sample=request.do_sample if request.do_sample is not None else True,
            repetition_penalty=request.repetition_penalty,
        )

        return {
            "caption": caption,
            "source": source,
            "model": request.model_path or DEFAULT_AUDIO_MODEL_ID,
            "media_type": media_type,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating audio caption: {str(e)}",
        )
    finally:
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


@app.post('/tag')
def tag_media(request: Dict[str, Any] = Body(...)):
    temp_paths = []
    input_data = request.get("input")
    model_path = request.get("model_path")
    if not input_data or not isinstance(input_data, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'input' should be provided and be a list.",
        )
    if model_path is not None and not isinstance(model_path, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'model_path' must be a string when provided.",
        )

    from api_server.tagger import PredictionRequest

    try:
        prediction_requests = []
        for idx, inp in enumerate(input_data):
            if isinstance(inp, str):
                prediction_requests.append(PredictionRequest.new(inp))
                continue
            if not isinstance(inp, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid input format at index {idx}",
                )

            media_path = inp.get("media_path")
            media_url = inp.get("media_url")
            if media_path and media_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Provide either media_path or media_url at index {idx}, not both",
                )
            if not media_path and not media_url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"media_path or media_url must be provided at index {idx}",
                )

            if media_url:
                temp_path = _download_media_from_url(media_url, inp.get("media_type"))
                temp_paths.append(temp_path)
                resolved = dict(inp)
                resolved["media_path"] = temp_path
                resolved.pop("media_url", None)
                prediction_requests.append(PredictionRequest.from_dict(resolved))
            else:
                prediction_requests.append(PredictionRequest.from_dict(inp))

        tagger = _get_tagger_service(model_path)
        result = tagger.predict_inputs(prediction_requests)
        return {"result": result}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating tags: {str(e)}",
        )
    finally:
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


@app.get('/tag/free')
def free_tagger_vram():
    global _tagger_service
    if _tagger_service is None:
        return {
            'status': 'no_model_loaded',
            'message': 'No tagger model is currently loaded',
        }

    import torch

    memory_before = _get_cuda_used_memory() / (1024 * 1024)
    _tagger_service.offload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = _get_cuda_used_memory() / (1024 * 1024)
    else:
        memory_after = 0

    return {
        'status': 'success',
        'memory_before': memory_before,
        'memory_after': memory_after,
        'freed': max(0.0, memory_before - memory_after),
    }


@app.post('/caption/unload')
def unload_caption_model():

    global _caption_processor

    if _caption_processor is None:
        return {
            'status': 'no_model_loaded',
            'message': 'No model is currently loaded'
        }

    try:
        _caption_processor.unload_model()
        _caption_processor = None

        return {
            'status': 'success',
            'message': 'Model unloaded and GPU memory freed'
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading model: {str(e)}"
        )


@app.post('/audio/unload')
def unload_audio_model():
    global _audio_captioner, _audio_model_path

    if _audio_captioner is None:
        return {
            'status': 'no_model_loaded',
            'message': 'No audio model is currently loaded',
        }

    try:
        _audio_captioner.unload_model()
        _audio_captioner = None
        _audio_model_path = None

        return {
            'status': 'success',
            'message': 'Audio model unloaded and GPU memory freed',
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading audio model: {str(e)}",
        )


@app.on_event('shutdown')
def _shutdown():
    global _caption_processor, _tagger_service, _tagger_model_path, _audio_captioner, _audio_model_path
    if _caption_processor is not None:
        _caption_processor.unload_model()
    if _tagger_service is not None:
        _tagger_service.offload()
        _tagger_service = None
        _tagger_model_path = None
    if _audio_captioner is not None:
        _audio_captioner.unload_model()
        _audio_captioner = None
        _audio_model_path = None
    manager.dispose_all()
