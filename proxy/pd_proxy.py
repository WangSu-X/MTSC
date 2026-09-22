"""Simple concurrent Prefill/Decode proxy for MTSC.

The proxy deliberately has no retry or attempt-id machinery. Backend failures
are surfaced to the client and recorded in the request JSONL file.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("mtsc.proxy")
TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def _iso8601(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _duration_ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000, 3)


@dataclass(frozen=True)
class PrefillEndpoint:
    api_url: str
    engine_id: str
    bootstrap_addr: str
    dp_rank: int | None = None


@dataclass(frozen=True)
class DecodeEndpoint:
    api_url: str


@dataclass
class RequestMetrics:
    request_id: str
    client_request_id: str | None
    transfer_id: str
    endpoint: str
    prefill: PrefillEndpoint
    decode: DecodeEndpoint
    model: str | None = None
    stream: bool | None = None
    max_tokens: int | None = None
    prompt_tokens: int | None = None
    request_received_at: float = field(default_factory=time.time)
    request_received_mono: float = field(default_factory=time.monotonic)
    prefill_start_at: float | None = None
    prefill_start_mono: float | None = None
    prefill_end_at: float | None = None
    prefill_end_mono: float | None = None
    decode_start_at: float | None = None
    decode_start_mono: float | None = None
    decode_headers_at: float | None = None
    decode_headers_mono: float | None = None
    decode_first_chunk_at: float | None = None
    decode_first_chunk_mono: float | None = None
    decode_end_at: float | None = None
    decode_end_mono: float | None = None
    request_end_at: float | None = None
    request_end_mono: float | None = None
    prefill_http_status: int | None = None
    decode_http_status: int | None = None
    outcome: str = "running"
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def set_body(self, body: dict[str, Any]) -> None:
        model = body.get("model")
        self.model = str(model) if model is not None else None
        self.stream = bool(body.get("stream", False))
        limit = body.get("max_tokens", body.get("max_completion_tokens"))
        self.max_tokens = int(limit) if isinstance(limit, int) else None
        prompt = body.get("prompt")
        if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
            self.prompt_tokens = len(prompt)

    def to_record(self) -> dict[str, Any]:
        p_url, d_url = urlsplit(self.prefill.api_url), urlsplit(self.decode.api_url)
        return {
            "schema_version": 1,
            "record_type": "mtsc_proxy_request",
            "request_id": self.request_id,
            "client_request_id": self.client_request_id,
            "transfer_id": self.transfer_id,
            "endpoint": self.endpoint,
            "model": self.model,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "prompt_tokens": self.prompt_tokens,
            "prefill_host": p_url.hostname,
            "prefill_port": p_url.port,
            "prefill_engine_id": self.prefill.engine_id,
            "prefill_dp_rank": self.prefill.dp_rank or 0,
            "decode_host": d_url.hostname,
            "decode_port": d_url.port,
            "request_received_at": _iso8601(self.request_received_at),
            "prefill_start_at": _iso8601(self.prefill_start_at),
            "prefill_end_at": _iso8601(self.prefill_end_at),
            "decode_start_at": _iso8601(self.decode_start_at),
            "decode_headers_at": _iso8601(self.decode_headers_at),
            "decode_first_chunk_at": _iso8601(self.decode_first_chunk_at),
            "decode_end_at": _iso8601(self.decode_end_at),
            "request_end_at": _iso8601(self.request_end_at),
            "prefill_duration_ms": _duration_ms(
                self.prefill_start_mono, self.prefill_end_mono
            ),
            "decode_headers_latency_ms": _duration_ms(
                self.decode_start_mono, self.decode_headers_mono
            ),
            "decode_ttft_ms": _duration_ms(
                self.decode_start_mono, self.decode_first_chunk_mono
            ),
            "decode_duration_ms": _duration_ms(
                self.decode_start_mono, self.decode_end_mono
            ),
            "total_duration_ms": _duration_ms(
                self.request_received_mono, self.request_end_mono
            ),
            "prefill_http_status": self.prefill_http_status,
            "decode_http_status": self.decode_http_status,
            "outcome": self.outcome,
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class TransferSession:
    request_id: str
    transfer_id: str
    prefill: PrefillEndpoint
    decode: DecodeEndpoint
    metrics: RequestMetrics
    prefill_task: asyncio.Task[None] | None = None


class JsonlMetricsWriter:
    def __init__(self, path: Path | None, queue_size: int = 4096) -> None:
        self.path = path
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=max(1, queue_size)
        )
        self._task: asyncio.Task[None] | None = None
        self.write_failures = 0

    @staticmethod
    def _append(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.write("\n")

    async def _run(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                if record is None:
                    return
                assert self.path is not None
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                try:
                    await asyncio.to_thread(self._append, self.path, line)
                except Exception:
                    self.write_failures += 1
                    logger.exception("MTSC metrics write failed")
            finally:
                self._queue.task_done()

    async def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="mtsc-metrics-writer")
        await self._queue.put(record)

    async def close(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task
        self._task = None


class PDProxy:
    def __init__(
        self,
        prefill: list[PrefillEndpoint],
        decode: list[DecodeEndpoint],
        metrics_file: Path | None,
        backend_token: str | None,
        metrics_queue_size: int = 4096,
        first_token_timeout: float = 180.0,
    ) -> None:
        if not prefill or not decode:
            raise ValueError(
                "At least one Prefill and one Decode endpoint are required"
            )
        self.prefill = prefill
        self.decode = decode
        self._prefill_cycle = itertools.cycle(prefill)
        self._decode_cycle = itertools.cycle(decode)
        self.metrics = JsonlMetricsWriter(metrics_file, metrics_queue_size)
        self.backend_token = backend_token
        self.first_token_timeout = first_token_timeout
        if self.first_token_timeout <= 0:
            raise ValueError("first_token_timeout must be positive")
        self.sessions: dict[str, TransferSession] = {}

        self.app = FastAPI(title="MTSC PD Proxy")
        self.app.post("/v1/completions")(self.completions)
        self.app.post("/v1/chat/completions")(self.chat_completions)
        self.app.get("/status")(self.status)
        self.app.router.add_event_handler("shutdown", self.close)

    async def status(self) -> dict[str, Any]:
        return {
            "prefill": [asdict(endpoint) for endpoint in self.prefill],
            "decode": [asdict(endpoint) for endpoint in self.decode],
        }

    def _new_session(self, raw_request: Request, route: str) -> TransferSession:
        client_request_id = raw_request.headers.get("x-request-id")
        request_id = uuid.uuid4().hex
        transfer_id = f"xfer-{request_id}"
        prefill = next(self._prefill_cycle)
        decode = next(self._decode_cycle)
        metrics = RequestMetrics(
            request_id=request_id,
            client_request_id=client_request_id,
            transfer_id=transfer_id,
            endpoint=route,
            prefill=prefill,
            decode=decode,
        )
        session = TransferSession(request_id, transfer_id, prefill, decode, metrics)
        self.sessions[request_id] = session
        return session

    def _headers(
        self, raw_request: Request, session: TransferSession
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": session.request_id,
        }
        authorization = (
            f"Bearer {self.backend_token}"
            if self.backend_token
            else raw_request.headers.get("authorization")
        )
        if authorization:
            headers["Authorization"] = authorization
        return headers

    @staticmethod
    def _prefill_body(body: dict[str, Any], session: TransferSession) -> dict[str, Any]:
        result = dict(body)
        result["stream"] = False
        result.pop("stream_options", None)
        result["max_tokens"] = 1
        if "max_completion_tokens" in result:
            result["max_completion_tokens"] = 1
        result["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "transfer_id": session.transfer_id,
        }
        return result

    @staticmethod
    def _decode_body(body: dict[str, Any], session: TransferSession) -> dict[str, Any]:
        result = dict(body)
        result["kv_transfer_params"] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_engine_id": session.prefill.engine_id,
            "remote_bootstrap_addr": session.prefill.bootstrap_addr,
            "remote_dp_rank": session.prefill.dp_rank or 0,
            "transfer_id": session.transfer_id,
        }
        return result

    async def _run_prefill(
        self,
        client: aiohttp.ClientSession,
        route: str,
        body: dict[str, Any],
        headers: dict[str, str],
        session: TransferSession,
    ) -> None:
        metrics = session.metrics
        metrics.prefill_start_at = time.time()
        metrics.prefill_start_mono = time.monotonic()
        p_headers = dict(headers)
        if session.prefill.dp_rank is not None:
            p_headers["X-data-parallel-rank"] = str(session.prefill.dp_rank)
        url = f"{session.prefill.api_url}{route}"
        try:
            async with client.post(url, json=body, headers=p_headers) as response:
                metrics.prefill_http_status = response.status
                payload = await response.read()
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"Prefill returned HTTP {response.status}: "
                        f"{payload[:512].decode(errors='replace')}"
                    )
        finally:
            metrics.prefill_end_at = time.time()
            metrics.prefill_end_mono = time.monotonic()

    async def _finish(
        self,
        metrics: RequestMetrics,
        outcome: str,
        error: BaseException | None = None,
        error_stage: str | None = None,
    ) -> None:
        if metrics.request_end_at is not None:
            return
        metrics.request_end_at = time.time()
        metrics.request_end_mono = time.monotonic()
        metrics.outcome = outcome
        if error is not None:
            metrics.error_stage = error_stage
            metrics.error_type = type(error).__name__
            metrics.error_message = str(error)
        self.sessions.pop(metrics.request_id, None)
        await self.metrics.write(metrics.to_record())

    async def close(self) -> None:
        tasks = [
            session.prefill_task
            for session in self.sessions.values()
            if session.prefill_task is not None and not session.prefill_task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.metrics.close()

    async def _stream_decode(
        self,
        client: aiohttp.ClientSession,
        response: aiohttp.ClientResponse,
        prefill_task: asyncio.Task[None],
        session: TransferSession,
    ):
        metrics = session.metrics
        iterator = response.content.iter_any().__aiter__()
        pending_prefill: asyncio.Task[None] | None = prefill_task
        failure_stage = "decode"
        first_chunk_deadline = (
            metrics.decode_start_mono + self.first_token_timeout
            if metrics.decode_start_mono is not None
            else time.monotonic() + self.first_token_timeout
        )
        try:
            while True:
                next_chunk = asyncio.create_task(iterator.__anext__())
                wait_for: set[asyncio.Task[Any]] = {next_chunk}
                if pending_prefill is not None:
                    wait_for.add(pending_prefill)
                done, _ = await asyncio.wait(
                    wait_for,
                    timeout=(
                        max(0.0, first_chunk_deadline - time.monotonic())
                        if metrics.decode_first_chunk_at is None
                        else None
                    ),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    next_chunk.cancel()
                    await asyncio.gather(next_chunk, return_exceptions=True)
                    raise TimeoutError(
                        f"Decode first chunk timed out after "
                        f"{self.first_token_timeout:.3f}s"
                    )
                if pending_prefill is not None and pending_prefill in done:
                    try:
                        failure_stage = "prefill"
                        await pending_prefill
                    except BaseException:
                        if not next_chunk.done():
                            next_chunk.cancel()
                            await asyncio.gather(next_chunk, return_exceptions=True)
                        raise
                    else:
                        failure_stage = "decode"
                    pending_prefill = None
                if next_chunk not in done:
                    await next_chunk
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break
                if metrics.decode_first_chunk_at is None:
                    metrics.decode_first_chunk_at = time.time()
                    metrics.decode_first_chunk_mono = time.monotonic()
                yield chunk

            if pending_prefill is not None:
                failure_stage = "prefill"
                await pending_prefill
            metrics.decode_end_at = time.time()
            metrics.decode_end_mono = time.monotonic()
            await self._finish(metrics, "success")
        except asyncio.CancelledError as exc:
            if not prefill_task.done():
                prefill_task.cancel()
            await asyncio.gather(prefill_task, return_exceptions=True)
            metrics.decode_end_at = time.time()
            metrics.decode_end_mono = time.monotonic()
            await self._finish(metrics, "client_cancelled", exc, error_stage="client")
            raise
        except BaseException as exc:
            if not prefill_task.done():
                prefill_task.cancel()
            await asyncio.gather(prefill_task, return_exceptions=True)
            logger.warning(
                "MTSC proxy request failed: request_id=%s error=%s",
                session.request_id,
                exc,
            )
            metrics.decode_end_at = time.time()
            metrics.decode_end_mono = time.monotonic()
            outcome = (
                "timeout" if isinstance(exc, TimeoutError) else f"{failure_stage}_error"
            )
            await self._finish(metrics, outcome, exc, error_stage=failure_stage)
            raise
        finally:
            if metrics.decode_end_at is None:
                metrics.decode_end_at = time.time()
                metrics.decode_end_mono = time.monotonic()
            response.release()
            await client.close()

    async def _handle(self, raw_request: Request, route: str) -> Response:
        session = self._new_session(raw_request, route)
        metrics = session.metrics
        content_type = raw_request.headers.get("content-type", "").lower()
        if not content_type.startswith("application/json"):
            error = HTTPException(status_code=415, detail="application/json required")
            await self._finish(metrics, "proxy_error", error, "request")
            raise error
        try:
            body = await raw_request.json()
        except Exception as exc:  # noqa: BLE001 - client payload boundary
            await self._finish(metrics, "proxy_error", exc, "request")
            return JSONResponse(status_code=400, content={"detail": "invalid JSON"})
        if not isinstance(body, dict):
            error = HTTPException(status_code=400, detail="JSON object required")
            await self._finish(metrics, "proxy_error", error, "request")
            raise error
        metrics.set_body(body)
        headers = self._headers(raw_request, session)
        p_body = self._prefill_body(body, session)
        d_body = self._decode_body(body, session)
        client = aiohttp.ClientSession(timeout=TIMEOUT)
        prefill_task = asyncio.create_task(
            self._run_prefill(client, route, p_body, headers, session),
            name=f"mtsc-prefill-{session.request_id}",
        )
        session.prefill_task = prefill_task

        metrics.decode_start_at = time.time()
        metrics.decode_start_mono = time.monotonic()
        decode_headers_task = asyncio.create_task(
            client.post(
                f"{session.decode.api_url}{route}", json=d_body, headers=headers
            ),
            name=f"mtsc-decode-headers-{session.request_id}",
        )
        try:
            done, _ = await asyncio.wait(
                {prefill_task, decode_headers_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if prefill_task in done:
                try:
                    await prefill_task
                except BaseException as exc:
                    decode_headers_task.cancel()
                    await asyncio.gather(decode_headers_task, return_exceptions=True)
                    await client.close()
                    metrics.decode_end_at = time.time()
                    metrics.decode_end_mono = time.monotonic()
                    await self._finish(metrics, "prefill_error", exc, "prefill")
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    return JSONResponse(status_code=502, content={"detail": str(exc)})
            response = await decode_headers_task
            metrics.decode_http_status = response.status
            metrics.decode_headers_at = time.time()
            metrics.decode_headers_mono = time.monotonic()
            if not 200 <= response.status < 300:
                payload = await response.read()
                error = RuntimeError(
                    f"Decode returned HTTP {response.status}: "
                    f"{payload[:512].decode(errors='replace')}"
                )
                if not prefill_task.done():
                    prefill_task.cancel()
                await asyncio.gather(prefill_task, return_exceptions=True)
                response.release()
                await client.close()
                metrics.decode_end_at = time.time()
                metrics.decode_end_mono = time.monotonic()
                await self._finish(metrics, "decode_error", error, "decode")
                return Response(
                    content=payload,
                    status_code=response.status,
                    media_type=response.headers.get(
                        "Content-Type", "application/json"
                    ).split(";", 1)[0],
                )
            if prefill_task.done():
                try:
                    await prefill_task
                except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                    response.release()
                    await client.close()
                    await self._finish(metrics, "prefill_error", exc, "prefill")
                    return JSONResponse(status_code=502, content={"detail": str(exc)})
        except BaseException as exc:
            if not decode_headers_task.done():
                decode_headers_task.cancel()
                await asyncio.gather(decode_headers_task, return_exceptions=True)
            if not prefill_task.done():
                prefill_task.cancel()
            await asyncio.gather(prefill_task, return_exceptions=True)
            await client.close()
            metrics.decode_end_at = time.time()
            metrics.decode_end_mono = time.monotonic()
            outcome = "timeout" if isinstance(exc, TimeoutError) else "decode_error"
            await self._finish(metrics, outcome, exc, "decode")
            if isinstance(exc, asyncio.CancelledError):
                raise
            return JSONResponse(status_code=502, content={"detail": str(exc)})

        media_type = response.headers.get("Content-Type", "application/json").split(
            ";", 1
        )[0]
        return StreamingResponse(
            self._stream_decode(client, response, prefill_task, session),
            status_code=response.status,
            media_type=media_type,
            headers={"X-Request-Id": session.request_id},
        )

    async def completions(self, request: Request) -> Response:
        return await self._handle(request, "/v1/completions")

    async def chat_completions(self, request: Request) -> Response:
        return await self._handle(request, "/v1/chat/completions")


def _url(value: str) -> str:
    value = value.rstrip("/")
    return value if "://" in value else f"http://{value}"


def _parse_prefill(value: str) -> PrefillEndpoint:
    # API_URL,ENGINE_ID,BOOTSTRAP_ADDR[,DP_RANK]
    fields = value.split(",")
    if len(fields) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "prefill must be API_URL,ENGINE_ID,BOOTSTRAP_ADDR[,DP_RANK]"
        )
    return PrefillEndpoint(
        api_url=_url(fields[0]),
        engine_id=fields[1],
        bootstrap_addr=_url(fields[2]),
        dp_rank=int(fields[3]) if len(fields) == 4 else None,
    )


def _parse_decode(value: str) -> DecodeEndpoint:
    return DecodeEndpoint(api_url=_url(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill", action="append", type=_parse_prefill, required=True
    )
    parser.add_argument("--decode", action="append", type=_parse_decode, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    metrics_default = os.getenv("MTSC_PROXY_METRICS_FILE")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path(metrics_default) if metrics_default else None,
    )
    parser.add_argument("--metrics-queue-size", type=int, default=4096)
    parser.add_argument("--first-token-timeout", type=float, default=180.0)
    parser.add_argument("--backend-token", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    proxy = PDProxy(
        args.prefill,
        args.decode,
        args.metrics_file,
        args.backend_token,
        metrics_queue_size=args.metrics_queue_size,
        first_token_timeout=args.first_token_timeout,
    )
    uvicorn.run(
        proxy.app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
