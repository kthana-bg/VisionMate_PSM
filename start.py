"""
start.py
--------
Single entry-point for Render (and local dev).

Architecture
────────────
  Browser → HTTPS :$PORT → Starlette reverse proxy
                              ├── /api/*  → FastAPI  (uvicorn :8001 internal)
                              └── /*      → Streamlit (subprocess :8002 internal)

Only ONE port ($PORT) is exposed externally, which is exactly what Render expects.
FastAPI and Streamlit each run on private localhost ports.
No STUN. No WebRTC. No TURN.
"""

import os
import sys
import subprocess
import threading
import time
import signal
import asyncio
import httpx
from starlette.applications  import Starlette
from starlette.requests       import Request
from starlette.responses      import Response, StreamingResponse
from starlette.routing        import Route, Mount
import uvicorn

# ── Ports ──────────────────────────────────────────────────
PUBLIC_PORT      = int(os.environ.get("PORT", 10000))
FASTAPI_PORT     = 8001   # internal, never exposed
STREAMLIT_PORT   = 8002   # internal, never exposed


# ── Async HTTP clients ─────────────────────────────────────
# Created once in the event loop; reused for all proxy calls.
_api_client: httpx.AsyncClient = None
_st_client:  httpx.AsyncClient = None


async def _get_api_client():
    global _api_client
    if _api_client is None:
        _api_client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{FASTAPI_PORT}",
            timeout=30.0,
        )
    return _api_client


async def _get_st_client():
    global _st_client
    if _st_client is None:
        _st_client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{STREAMLIT_PORT}",
            timeout=60.0,
        )
    return _st_client


# ── Proxy handlers ─────────────────────────────────────────

async def _proxy(client: httpx.AsyncClient, request: Request, path: str) -> Response:
    url     = httpx.URL(path=path, query=request.url.query.encode())
    body    = await request.body()
    headers = dict(request.headers)
    # Remove hop-by-hop headers that confuse the upstream
    for h in ("host", "content-length", "transfer-encoding", "connection"):
        headers.pop(h, None)

    try:
        rsp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        return Response(
            content=rsp.content,
            status_code=rsp.status_code,
            headers=dict(rsp.headers),
        )
    except httpx.ConnectError:
        return Response(b"Service starting up, please wait...", status_code=503)


async def api_proxy(request: Request) -> Response:
    # Strip the /api prefix before forwarding
    path = "/" + request.url.path.lstrip("/")[4:].lstrip("/")   # /api/frame → /frame
    client = await _get_api_client()
    return await _proxy(client, request, path)


async def streamlit_proxy(request: Request) -> Response:
    path   = request.url.path or "/"
    client = await _get_st_client()
    return await _proxy(client, request, path)


# ── Starlette proxy app ────────────────────────────────────

proxy_app = Starlette(
    routes=[
        # Anything under /api/ → FastAPI
        Mount("/api", app=api_proxy),
        # Everything else → Streamlit
        Route("/{path:path}", endpoint=streamlit_proxy, methods=["GET","POST","PUT","DELETE","PATCH","OPTIONS","HEAD"]),
    ]
)


# ── Background processes ───────────────────────────────────

def _run_fastapi():
    """FastAPI inference server — internal port."""
    uvicorn.run(
        "inference_server:app",
        host="127.0.0.1",
        port=FASTAPI_PORT,
        log_level="warning",
        workers=1,
    )


def _run_streamlit():
    """Streamlit UI — internal port."""
    time.sleep(4)   # Wait for FastAPI to finish loading models
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port",                 str(STREAMLIT_PORT),
        "--server.address",              "127.0.0.1",
        "--server.headless",             "true",
        "--server.enableCORS",           "false",
        "--server.enableXsrfProtection", "false",
        "--server.baseUrlPath",          "",
    ]
    proc = subprocess.Popen(cmd)
    proc.wait()


# ── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[start] Public port  : {PUBLIC_PORT}")
    print(f"[start] FastAPI port : {FASTAPI_PORT} (internal)")
    print(f"[start] Streamlit port: {STREAMLIT_PORT} (internal)")

    # FastAPI in a daemon thread
    api_thread = threading.Thread(target=_run_fastapi, daemon=True)
    api_thread.start()

    # Streamlit in a daemon thread
    st_thread = threading.Thread(target=_run_streamlit, daemon=True)
    st_thread.start()

    # Reverse proxy on the public port (main thread — keeps the process alive)
    uvicorn.run(
        proxy_app,
        host="0.0.0.0",
        port=PUBLIC_PORT,
        log_level="info",
    )
