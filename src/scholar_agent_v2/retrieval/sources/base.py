from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from ..models import (
    AdapterFetchResult,
    AdapterSearchResult,
    RouteSpec,
    SearchSource,
    SourceCapabilities,
)


class SourceError(RuntimeError):
    pass


class SourceConfigurationError(SourceError):
    pass


class SourceHTTPError(SourceError):
    def __init__(self, status: int, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class SourceRateLimitError(SourceHTTPError):
    pass


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


@dataclass
class HTTPRequestCounter:
    """单个异步执行上下文内实际发出的 HTTP 请求数。"""

    count: int = 0


_ACTIVE_HTTP_COUNTER: ContextVar[HTTPRequestCounter | None] = ContextVar(
    "retrieval_http_request_counter",
    default=None,
)


class AcademicSourceAdapter(ABC):
    source: SearchSource
    capabilities: SourceCapabilities

    def __init__(self, *, timeout_seconds: float = 15.0, user_agent: str = "ScholarAgent/1.0") -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    @abstractmethod
    async def search(self, route: RouteSpec) -> AdapterSearchResult:
        raise NotImplementedError

    @abstractmethod
    async def fetch_details(self, ids: list[str]) -> AdapterFetchResult:
        raise NotImplementedError

    @contextmanager
    def track_http_requests(self) -> Iterator[HTTPRequestCounter]:
        """隔离统计当前 route attempt 发出的 HTTP 请求，支持并发任务。"""
        counter = HTTPRequestCounter()
        token = _ACTIVE_HTTP_COUNTER.set(counter)
        try:
            yield counter
        finally:
            _ACTIVE_HTTP_COUNTER.reset(token)

    @staticmethod
    def _record_http_request() -> None:
        counter = _ACTIVE_HTTP_COUNTER.get()
        if counter is not None:
            counter.count += 1

    async def _get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        if params:
            encoded = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None},
                doseq=True,
            )
            url = f"{url}{'&' if '?' in url else '?'}{encoded}"
        self._record_http_request()
        return await asyncio.to_thread(self._request_sync, "GET", url, None, headers or {})

    async def _post_json(
        self,
        url: str,
        payload: Any,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        if params:
            encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, doseq=True)
            url = f"{url}{'&' if '?' in url else '?'}{encoded}"
        body = json.dumps(payload).encode("utf-8")
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        self._record_http_request()
        return await asyncio.to_thread(self._request_sync, "POST", url, body, hdrs)

    def _request_sync(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> HTTPResponse:
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json", **headers}
        req = urllib.request.Request(url, data=body, method=method, headers=request_headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                return HTTPResponse(
                    status=response.status,
                    body=response.read(),
                    headers={k.lower(): v for k, v in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                retry_after_value = float(retry_after) if retry_after else None
            except ValueError:
                retry_after_value = None
            message = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code == 429:
                raise SourceRateLimitError(exc.code, message or "rate limited", retry_after=retry_after_value) from exc
            raise SourceHTTPError(exc.code, message or str(exc), retry_after=retry_after_value) from exc
        except urllib.error.URLError as exc:
            raise SourceError(str(exc.reason)) from exc


__all__ = [
    "AcademicSourceAdapter",
    "SourceError",
    "SourceConfigurationError",
    "SourceHTTPError",
    "SourceRateLimitError",
    "HTTPRequestCounter",
]
