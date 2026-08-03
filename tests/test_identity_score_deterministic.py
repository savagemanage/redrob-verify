"""Regression test: deterministic identity scoring must never vary."""

from __future__ import annotations

import statistics

from services.identity.scoring import score_identity


def test_identity_score_has_zero_variance_across_five_runs() -> None:
    resume_handles = {"github": "octocat", "codeforces": "tourist"}
    supplied_handles = {"github": "octocat", "codeforces": "tourist"}
    sources = [
        {"source": "github", "status": "ok"},
        {"source": "codeforces", "status": "ok"},
    ]
    weights = {"github": 2.0, "codeforces": 1.0}

    scores = [
        score_identity(resume_handles, supplied_handles, sources, weights)[0]
        for _ in range(5)
    ]

    assert scores == [100] * 5
    assert statistics.pvariance(scores) == 0
