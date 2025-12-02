import gc
import threading
import uuid
from typing import Any, Dict, List, Optional

import torch

from api_server.training_session import TrainingSession


class TrainingSessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, TrainingSession] = {}

    def create_session(
        self,
        config: dict,
        *,
        session_id: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> TrainingSession:
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            if sid in self._sessions:
                raise ValueError(f'session "{sid}" already exists')
            session = TrainingSession(sid, config, max_steps=max_steps)
            self._sessions[sid] = session
            return session

    def get_session(self, session_id: str) -> TrainingSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f'unknown session "{session_id}"')
            return self._sessions[session_id]

    def list_status(self) -> Dict[str, dict]:
        with self._lock:
            return {sid: session.get_status() for sid, session in self._sessions.items()}

    def allocate_steps(self, session_id: str, steps: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        session = self.get_session(session_id)
        return session.allocate_steps(steps, timeout=timeout)

    def abort_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        session.abort()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise KeyError(f'unknown session "{session_id}"')
        session.dispose()

    def dispose_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()
        for _, session in sessions:
            session.dispose()

    def free_all_vram(self) -> Dict[str, Any]:
        freed_sessions: List[str] = []
        failed_sessions: List[str] = []

        with self._lock:
            sessions = list(self._sessions.items())

        for sid, session in sessions:
            try:
                if session.free_vram():
                    freed_sessions.append(sid)
                else:
                    failed_sessions.append(sid)
            except Exception:
                failed_sessions.append(sid)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return {
            'freed_sessions': freed_sessions,
            'failed_sessions': failed_sessions,
        }
