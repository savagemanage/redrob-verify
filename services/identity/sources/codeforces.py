"""Codeforces public REST API adapter."""

from __future__ import annotations

import httpx

from services.identity.sources.base import SourceResult, maybe_inject_delay


class CodeforcesAdapter:
    name = "codeforces"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, handle: str | None) -> SourceResult:
        await maybe_inject_delay()
        if not handle:
            return SourceResult(self.name, handle, "not_provided")
        try:
            response = await self.client.get(
                "https://codeforces.com/api/user.info", params={"handles": handle}
            )
            response.raise_for_status()
            payload = response.json()
            users = payload.get("result") if payload.get("status") == "OK" else []
            if not users:
                return SourceResult(self.name, handle, "not_found")
            user = users[0]
            return SourceResult(
                self.name,
                handle,
                "ok",
                {
                    "handle": user.get("handle"),
                    "name": " ".join(
                        value for value in [user.get("firstName"), user.get("lastName")] if value
                    ),
                    "country": user.get("country"),
                    "rating": user.get("rating"),
                    "max_rating": user.get("maxRating"),
                },
            )
        except httpx.HTTPError as error:
            return SourceResult(self.name, handle, "error", error=str(error))
