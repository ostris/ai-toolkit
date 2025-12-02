from typing import Any, Dict, Optional
import os
import tempfile
import requests
from io import BytesIO

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from api_server.manager import TrainingSessionManager

app = FastAPI(title='AI Toolkit Training API')

manager = TrainingSessionManager()

_caption_processor = None


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
    max_new_tokens: Optional[int] = Field(default=1024, description="Maximum tokens to generate")
    num_beams: Optional[int] = Field(default=3, description="Number of beams for generation (Florence2 only)")
    task: Optional[str] = Field(default="<DETAILED_CAPTION>", description="Florence2 task type (Florence2 only)")
    prompt: Optional[str] = Field(default=None, description="Custom prompt for JoyCaption (default: 'Write a long descriptive caption for this image in a formal tone.')")
    temperature: Optional[float] = Field(default=0.6, description="Temperature for sampling (JoyCaption only)")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling parameter (JoyCaption only)")
    num_frames: Optional[int] = Field(default=8, description="Number of frames to extract from video")
    sample_method: Optional[str] = Field(default="uniform", description="Frame sampling method: 'uniform' or 'first'")
    combine_method: Optional[str] = Field(default="first", description="Caption combining method: 'first', 'longest', or 'combined'")

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


@app.post('/caption')
def generate_caption(request: CaptionRequest):

    temp_video_path = None
    try:
        is_video = request.video_path is not None or request.video_url is not None
        is_image = request.image_path is not None or request.image_url is not None

        if is_video and is_image:
            raise ValueError("Provide either image or video, not both")

        if not is_video and not is_image:
            raise ValueError("Must provide either image_path, image_url, video_path, or video_url")

        processor = _get_caption_processor(request.model_type, request.model_path)

        if is_video:
            video_path = _load_video_from_source(request.video_path, request.video_url)
            if request.video_url:
                temp_video_path = video_path 

            if request.model_type == "joycaption":
                caption = processor.generate_video_caption(
                    video_path=video_path,
                    num_frames=request.num_frames,
                    sample_method=request.sample_method,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    combine_method=request.combine_method,
                    prompt=request.prompt
                )
            else: 
                caption = processor.generate_video_caption(
                    video_path=video_path,
                    num_frames=request.num_frames,
                    sample_method=request.sample_method,
                    max_new_tokens=request.max_new_tokens,
                    num_beams=request.num_beams,
                    task=request.task,
                    combine_method=request.combine_method
                )

            source = request.video_path or request.video_url
            media_type = 'video'

        else:
            image = _load_image_from_source(request.image_path, request.image_url)

            if request.model_type == "joycaption":
                caption = processor.generate_caption(
                    image=image,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
            else:  
                caption = processor.generate_caption(
                    image=image,
                    max_new_tokens=request.max_new_tokens,
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
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass


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


@app.on_event('shutdown')
def _shutdown():
    global _caption_processor
    if _caption_processor is not None:
        _caption_processor.unload_model()
    manager.dispose_all()
