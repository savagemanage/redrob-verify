"""httpx.AsyncClient wrappers for the four module endpoints.

Latency is measured client-side. Server-provided latency_ms is ignored.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx


class ApiClient:
    def __init__(
        self,
        endpoints: dict[str, str],
        *,
        timeout_seconds: float = 30.0,
        retry_count: int = 3,
    ) -> None:
        self.endpoints = {k: v.rstrip("/") for k, v in endpoints.items()}
        self.timeout = timeout_seconds
        self.retry_count = retry_count
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ApiClient:
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ApiClient must be used as an async context manager")
        return self._client

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], float]:
        client = self._require_client()
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_count + 1):
            t0 = time.perf_counter()
            try:
                resp = await client.request(method, url, **kwargs)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise ValueError(f"expected JSON object from {url}, got {type(data)}")
                data = dict(data)
                data["_client_latency_ms"] = latency_ms
                return data, latency_ms
            except (httpx.HTTPError, ValueError) as e:
                last_exc = e
                if attempt >= self.retry_count:
                    break
                await asyncio.sleep(0.2 * attempt)
        assert last_exc is not None
        raise last_exc

    async def ocr_extract(
        self,
        image_path: Path | str,
        *,
        record_id: str | None = None,
        doc_type: str | None = None,
        lang: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        path = Path(image_path)
        url = f"{self.endpoints['ocr']}/v1/ocr/extract"
        content = path.read_bytes()
        files = {"file": (path.name, content, "application/octet-stream")}
        data: dict[str, str] = {"id": record_id or path.stem}
        if doc_type:
            data["doc_type"] = doc_type
        if lang:
            data["lang"] = lang
        return await self._request("POST", url, files=files, data=data)

    async def forgery_detect(
        self,
        image_path: Path | str,
        *,
        record_id: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        path = Path(image_path)
        url = f"{self.endpoints['forgery']}/v1/forgery/detect"
        content = path.read_bytes()
        files = {"file": (path.name, content, "application/octet-stream")}
        data = {"id": record_id or path.stem}
        return await self._request("POST", url, files=files, data=data)

    async def face_compare(
        self,
        img_a: Path | str,
        img_b: Path | str,
        *,
        record_id: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        a, b = Path(img_a), Path(img_b)
        url = f"{self.endpoints['face']}/v1/face/compare"
        files = {
            "img_a": (a.name, a.read_bytes(), "application/octet-stream"),
            "img_b": (b.name, b.read_bytes(), "application/octet-stream"),
        }
        data = {"id": record_id or a.stem}
        return await self._request("POST", url, files=files, data=data)

    async def identity_aggregate(
        self,
        resume_path: Path | str,
        profiles: dict[str, str] | None = None,
        *,
        record_id: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        path = Path(resume_path)
        url = f"{self.endpoints['identity']}/v1/identity/aggregate"
        files = {"file": (path.name, path.read_bytes(), "application/octet-stream")}
        form: dict[str, str] = {"id": record_id or path.stem}
        if profiles:
            for k, v in profiles.items():
                form[f"profile_{k}"] = v
        return await self._request("POST", url, files=files, data=form)
