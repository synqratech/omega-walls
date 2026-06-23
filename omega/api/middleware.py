"""Fail-closed request size and deadline middleware."""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class StreamingBodyLimitMiddleware:
    def __init__(self, app: Callable[..., Any], max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max(1, int(max_body_bytes))

    async def __call__(self, scope: dict, receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {bytes(k).lower(): bytes(v) for k, v in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                await self._reject(send, status=400, detail=b'{"detail":"invalid_content_length"}')
                return
        seen = 0
        rejected = False

        async def limited_receive() -> dict:
            nonlocal seen, rejected
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_body_bytes:
                    rejected = True
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            if not rejected:
                raise
            await self._reject(send)

    @staticmethod
    async def _reject(send: Callable[..., Any], *, status: int = 413, detail: bytes = b'{"detail":"request_body_too_large"}') -> None:
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": detail, "more_body": False})


class _BodyTooLarge(Exception):
    pass


class RequestDeadlineMiddleware:
    def __init__(self, app: Callable[..., Any], timeout_sec: int) -> None:
        self.app = app
        self.timeout_sec = max(1, int(timeout_sec))

    async def __call__(self, scope: dict, receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracked_send(message: dict) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await asyncio.wait_for(self.app(scope, receive, tracked_send), timeout=float(self.timeout_sec))
        except asyncio.TimeoutError:
            if not response_started:
                await send({"type": "http.response.start", "status": 504, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"detail":"request_timeout"}', "more_body": False})
