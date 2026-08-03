"""LeetCode's publicly reachable GraphQL adapter."""

from __future__ import annotations

import httpx

from services.identity.sources.base import SourceResult, maybe_inject_delay

_QUERY = """
query userPublicProfile($username: String!) {
  matchedUser(username: $username) {
    username
    profile { realName ranking countryName }
    submitStatsGlobal { acSubmissionNum { difficulty count } }
  }
}
"""


class LeetCodeAdapter:
    name = "leetcode"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, handle: str | None) -> SourceResult:
        await maybe_inject_delay()
        if not handle:
            return SourceResult(self.name, handle, "not_provided")
        try:
            response = await self.client.post(
                "https://leetcode.com/graphql",
                json={"query": _QUERY, "variables": {"username": handle}},
            )
            response.raise_for_status()
            user = response.json().get("data", {}).get("matchedUser")
            if not user:
                return SourceResult(self.name, handle, "not_found")
            profile = user.get("profile") or {}
            return SourceResult(
                self.name,
                handle,
                "ok",
                {
                    "username": user.get("username"),
                    "name": profile.get("realName"),
                    "ranking": profile.get("ranking"),
                    "country": profile.get("countryName"),
                    "submissions": user.get("submitStatsGlobal", {}).get(
                        "acSubmissionNum", []
                    ),
                },
            )
        except httpx.HTTPError as error:
            return SourceResult(self.name, handle, "error", error=str(error))
