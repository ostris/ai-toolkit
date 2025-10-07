from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api_server.manager import TrainingSessionManager

app = FastAPI(title='AI Toolkit Training API')

manager = TrainingSessionManager()


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, alias='sessionId')
    max_steps: Optional[int] = Field(default=None, alias='maxSteps')
    config: Dict[str, Any]

    class Config:
        allow_population_by_field_name = True


class AllocateStepsRequest(BaseModel):
    steps: int = Field(..., gt=0)
    timeout: Optional[float] = None


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


@app.on_event('shutdown')
def _shutdown():
    manager.dispose_all()
