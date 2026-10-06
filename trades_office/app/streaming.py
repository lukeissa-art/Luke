"""Streamed chat replies: the assistant's words show up in the browser while the model writes them.

A chat page asks for a stream with `Accept: text/event-stream`. The reply then arrives as
server-sent events:
    data: {"type": "text", "text": "Sure, I can "}      (zero or more, in order)
    data: {"type": "done", "status": 200, "say": ..., "action": ..., ...}
The "done" event carries the same JSON the plain endpoint returns, including the final `say`,
which the page shows in place of the streamed text (it can differ, e.g. after an AI failure).
Without that header the endpoints return plain JSON as before.
"""

import asyncio
import json
import logging
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

log = logging.getLogger(__name__)


def wants_stream(request: Request) -> bool:
    return "text/event-stream" in request.headers.get("accept", "")


def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def stream_reply(work: Callable[[Callable[[str], None]], JSONResponse]) -> StreamingResponse:
    """Run work(on_text) in a worker thread and stream the text it reports, then its JSON result.

    Must be called from an async endpoint. If the visitor closes the page mid-reply, the turn
    still finishes and is saved; only the stream stops.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_text(delta: str) -> None:  # called from the worker thread
        if delta:
            loop.call_soon_threadsafe(queue.put_nowait, _event({"type": "text", "text": delta}))

    async def run() -> None:
        try:
            response = await run_in_threadpool(work, on_text)
            done = {"type": "done", "status": response.status_code, **json.loads(response.body)}
        except Exception:
            log.exception("Streamed reply failed")
            done = {"type": "done", "status": 500, "error": "Something went wrong. Please try again."}
        await queue.put(_event(done))
        await queue.put(None)

    async def events():
        task = asyncio.create_task(run())
        while (item := await queue.get()) is not None:
            yield item
        await task

    return StreamingResponse(events(), media_type="text/event-stream",
                             # Ask proxies not to hold the reply back until it's complete.
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
