"""HTTP surface of the engine (FastAPI). Runs on its own thread; every call
hands work to the Engine and returns, except /generate which streams the
job's frames for as long as the generation runs."""

import asyncio
import os
import re
import socket
import threading
import time
import uuid
from typing import Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from toolkit.models.registry import describe_archs

from .engine import END, Engine

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def create_app(engine: Engine, token: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="AI Toolkit Inference Engine", docs_url=None, redoc_url=None)

    def check_auth(request: Request):
        if not token:
            return
        supplied = request.headers.get("x-engine-token")
        if supplied is None:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                supplied = auth[7:]
        if supplied != token:
            raise HTTPException(status_code=401, detail="bad engine token")

    @app.get("/health")
    async def health():
        return engine.health()

    @app.get("/models")
    async def models(request: Request):
        check_auth(request)
        return {"archs": await asyncio.to_thread(describe_archs)}

    @app.get("/queue")
    async def queue_(request: Request):
        check_auth(request)
        return {"queued": engine.queued(), "recent": engine.recent()}

    @app.post("/cancel/{request_id}")
    async def cancel(request: Request, request_id: str):
        check_auth(request)
        return {"ok": engine.cancel(request_id)}

    @app.post("/unload")
    async def unload(request: Request):
        check_auth(request)
        if engine.current is not None:
            raise HTTPException(status_code=409, detail="busy")
        await asyncio.to_thread(engine.unload)
        return {"ok": True}

    @app.post("/assets")
    async def upload_asset(request: Request, name: str = "asset.png"):
        """Raw-body upload (no multipart dependency). Returns the stored path,
        usable as ctrl_img in a generate request."""
        check_auth(request)
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty body")
        base = _SAFE_NAME.sub("_", os.path.basename(name)) or "asset"
        stem, ext = os.path.splitext(base)
        asset_id = f"{stem}_{uuid.uuid4().hex[:8]}{ext or '.bin'}"
        path = os.path.join(engine.assets_folder, asset_id)
        with open(path, "wb") as f:
            f.write(body)
        return {"id": asset_id, "path": path, "bytes": len(body)}

    @app.get("/outputs/{relpath:path}")
    async def get_output(request: Request, relpath: str):
        check_auth(request)
        root = os.path.realpath(engine.output_folder)
        path = os.path.realpath(os.path.join(root, relpath))
        if not path.startswith(root + os.sep) or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.post("/generate")
    async def generate(request: Request):
        check_auth(request)
        body = await request.json()
        model = body.get("model")
        sample = body.get("sample") or {}
        if not isinstance(model, dict) or not model.get("arch"):
            raise HTTPException(status_code=400, detail="model.arch is required")
        if body.get("wait"):
            # plain JSON mode for scripts: block until the job is done, drop frames
            job = engine.submit(model, sample, {"latents": "none"})
            q = job.subscribe(replay=False)
            while True:
                frame = await asyncio.to_thread(q.get)
                if frame is END:
                    break
            return JSONResponse(job.info(), status_code=200 if job.status == "done" else 500)

        job = engine.submit(model, sample, body.get("stream"))
        # a dropped connection (page reload) must not cancel: the client
        # reattaches via /stream/{id}; cancelling is explicit (/cancel)
        return _stream_job(job, cancel_on_disconnect=False)

    @app.get("/stream/{request_id}")
    async def stream(request: Request, request_id: str):
        """(Re)attach to a request's frame stream: replays everything so far
        (status/progress/result frames and the newest latent) then follows
        live until the end frame. Disconnecting does not cancel the job."""
        check_auth(request)
        job = engine.get_job(request_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown request")
        return _stream_job(job, cancel_on_disconnect=False)

    def _stream_job(job, cancel_on_disconnect: bool):
        q = job.subscribe(replay=True)

        async def frames():
            try:
                while True:
                    frame = await asyncio.to_thread(q.get)
                    if frame is END:
                        break
                    yield frame
            finally:
                if cancel_on_disconnect and job.status in ("queued", "running"):
                    engine.cancel(job.request_id)

        return StreamingResponse(
            frames(),
            media_type="application/octet-stream",
            headers={"X-Request-Id": job.request_id, "Cache-Control": "no-store"},
        )

    return app


def serve_in_thread(app, host: str = "127.0.0.1", port: int = 0, timeout: float = 30.0) -> Tuple[object, threading.Thread, str, int]:
    """Bind the socket ourselves (port 0 = ephemeral) and run uvicorn on a
    daemon thread. Returns (server, thread, host, port); the caller stops it
    with server.should_exit = True. uvicorn only installs signal handlers on
    the main thread, so the process keeps its own."""
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(64)
    host, port = sock.getsockname()[:2]
    config = uvicorn.Config(app, log_level="warning", access_log=False, timeout_keep_alive=600)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True, name="engine-http")
    thread.start()
    deadline = time.time() + timeout
    while not server.started and time.time() < deadline:
        if not thread.is_alive():
            raise RuntimeError("engine http server died on startup")
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("engine http server failed to start")
    return server, thread, host, port
