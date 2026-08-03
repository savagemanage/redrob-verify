"""Identity aggregation orchestration with parallel, isolated source lookups."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from services.identity.cache import DiskCache
from services.identity.parser import extract_handles, extract_name, ocr_fallback_reason, parse_pdf
from services.identity.scoring import score_identity
from services.identity.sources.codeforces import CodeforcesAdapter
from services.identity.sources.github import GitHubAdapter
from services.identity.sources.leetcode import LeetCodeAdapter
from services.identity.sources.stackoverflow import StackOverflowAdapter

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class IdentityPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cache = DiskCache(
            Path(str(config.get("cache_dir", "results/.identity_cache"))),
            float(config.get("cache_ttl_seconds", 3600)),
        )
        self.timeout_seconds = float(config.get("source_timeout_seconds", 8))
        self.weights = dict(config.get("score_weights") or {})

    async def _fetch_one(self, adapter: Any, handle: str | None) -> dict[str, Any]:
        cached = self.cache.get(adapter.name, handle)
        if cached:
            LOGGER.info("identity source=%s cached handle=%s", adapter.name, handle)
            return cached
        LOGGER.info("identity source=%s START network handle=%s", adapter.name, handle)
        started = perf_counter()
        try:
            result = await asyncio.wait_for(adapter.fetch(handle), timeout=self.timeout_seconds)
            payload = result.as_dict()
        except TimeoutError:
            payload = {
                "source": adapter.name,
                "handle": handle,
                "status": "timeout",
                "data": {},
                "error": f"exceeded {self.timeout_seconds:.1f}s timeout",
                "cached": False,
            }
        except Exception as error:  # adapter isolation is part of the API contract
            payload = {
                "source": adapter.name,
                "handle": handle,
                "status": "error",
                "data": {},
                "error": str(error),
                "cached": False,
            }
        elapsed_ms = (perf_counter() - started) * 1000
        LOGGER.info(
            "identity source=%s END network status=%s elapsed_ms=%.1f",
            adapter.name,
            payload["status"],
            elapsed_ms,
        )
        self.cache.put(adapter.name, handle, payload)
        return payload

    async def aggregate(
        self, content: bytes | None, path: str | None, supplied_handles: dict[str, str]
    ) -> dict[str, Any]:
        started = perf_counter()
        parser: dict[str, Any] = {"status": "not_used", "urls": []}
        text = ""
        if content is not None:
            try:
                text, urls = parse_pdf(content)
                parser = {"status": "ok", "urls": urls, "name": extract_name(text)}
            except Exception as error:
                parser = ocr_fallback_reason(error)
        elif path:
            try:
                text, urls = parse_pdf(Path(path).read_bytes())
                parser = {"status": "ok", "urls": urls, "name": extract_name(text)}
            except Exception as error:
                parser = ocr_fallback_reason(error)

        resume_handles = extract_handles(text, parser.get("urls", []))
        handles = {**resume_handles, **supplied_handles}
        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            adapters = [
                GitHubAdapter(client),
                LeetCodeAdapter(client),
                CodeforcesAdapter(client),
                StackOverflowAdapter(client),
            ]
            gather_t0 = perf_counter()
            sources = await asyncio.gather(
                *(self._fetch_one(adapter, handles.get(adapter.name)) for adapter in adapters)
            )
            gather_ms = (perf_counter() - gather_t0) * 1000
            LOGGER.info(
                "identity gather_parallel_ms=%.1f sources=%d",
                gather_ms,
                len(sources),
            )
        sources = list(sources)
        sources.sort(key=lambda source: str(source["source"]))
        score, discrepancies = score_identity(
            resume_handles, supplied_handles, sources, self.weights
        )
        return {
            "consistency_score": score,
            "sources": sources,
            "discrepancies": discrepancies,
            "latency_ms": int(round((perf_counter() - started) * 1000)),
            "parsed": {
                "handles": resume_handles,
                "parser": parser,
                "path_provided": bool(path),
            },
        }
