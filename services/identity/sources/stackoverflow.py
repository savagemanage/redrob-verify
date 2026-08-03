"""Stack Exchange public API adapter for Stack Overflow profiles."""

from __future__ import annotations

import httpx

from services.identity.sources.base import SourceResult, maybe_inject_delay


class StackOverflowAdapter:
    name = "stackoverflow"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, handle: str | None) -> SourceResult:
        await maybe_inject_delay()
        if not handle:
            return SourceResult(self.name, handle, "not_provided")
        try:
            response = await self.client.get(
                "https://api.stackexchange.com/2.3/users",
                params={"site": "stackoverflow", "inname": handle, "pagesize": 10},
            )
            response.raise_for_status()
            users = response.json().get("items", [])
            normalized = handle.casefold()
            user = next(
                (
                    candidate
                    for candidate in users
                    if str(candidate.get("display_name", "")).casefold() == normalized
                ),
                None,
            )
            if not user:
                return SourceResult(self.name, handle, "not_found")
            return SourceResult(
                self.name,
                handle,
                "ok",
                {
                    "display_name": user.get("display_name"),
                    "reputation": user.get("reputation"),
                    "user_id": user.get("user_id"),
                    "profile_image": user.get("profile_image"),
                },
            )
        except httpx.HTTPError as error:
            return SourceResult(self.name, handle, "error", error=str(error))
