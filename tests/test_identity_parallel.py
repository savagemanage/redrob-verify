"""Prove identity sources run in parallel (artificial per-source delay)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import httpx
import pytest

from services.identity.pipeline import IdentityPipeline
from services.identity.sources.base import SourceResult, maybe_inject_delay


class SlowAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    async def fetch(self, handle: str | None) -> SourceResult:
        await maybe_inject_delay()
        return SourceResult(self.name, handle, "ok", {"handle": handle})


def test_four_sources_parallel_with_one_second_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IDENTITY_SOURCE_DELAY_SEC", "1.0")
    adapters = [SlowAdapter(n) for n in ("github", "leetcode", "codeforces", "stackoverflow")]

    async def gather_like_pipeline() -> None:
        await asyncio.gather(*(a.fetch("x") for a in adapters))

    t0 = time.perf_counter()
    asyncio.run(gather_like_pipeline())
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"expected ~1s parallel wall time, got {elapsed:.2f}s"
    assert elapsed >= 0.9, f"delay not applied? elapsed={elapsed:.2f}s"


def test_pipeline_issues_network_start_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    names = iter(["github", "leetcode", "codeforces", "stackoverflow"])

    class NamedFake:
        def __init__(self, client: httpx.AsyncClient) -> None:
            self.client = client
            self.name = next(names)

        async def fetch(self, handle: str | None) -> SourceResult:
            await asyncio.sleep(0.05)
            return SourceResult(self.name, handle, "ok", {})

    import services.identity.pipeline as pipe

    monkeypatch.setattr(pipe, "GitHubAdapter", NamedFake)
    monkeypatch.setattr(pipe, "LeetCodeAdapter", NamedFake)
    monkeypatch.setattr(pipe, "CodeforcesAdapter", NamedFake)
    monkeypatch.setattr(pipe, "StackOverflowAdapter", NamedFake)

    pipeline = IdentityPipeline(
        {
            "cache_dir": str(tmp_path / "cache"),
            "cache_ttl_seconds": 0,
            "source_timeout_seconds": 5,
        }
    )

    async def run() -> dict:
        return await pipeline.aggregate(
            None,
            None,
            {
                "github": "octocat",
                "leetcode": "leetcode",
                "codeforces": "tourist",
                "stackoverflow": "1",
            },
        )

    result = asyncio.run(run())
    assert "START network" in caplog.text
    assert "gather_parallel_ms" in caplog.text
    assert result["latency_ms"] >= 40
    assert len(result["sources"]) == 4
