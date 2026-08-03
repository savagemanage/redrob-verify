"""GitHub public REST API adapter."""

from __future__ import annotations

import os

import httpx

from services.identity.sources.base import SourceResult, maybe_inject_delay


class GitHubAdapter:
    name = "github"

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, handle: str | None) -> SourceResult:
        await maybe_inject_delay()
        if not handle:
            return SourceResult(self.name, handle, "not_provided")
        headers = {"Accept": "application/vnd.github+json"}
        if token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        try:
            response = await self.client.get(
                f"https://api.github.com/users/{handle}", headers=headers
            )
            if response.status_code == 404:
                return SourceResult(self.name, handle, "not_found")
            response.raise_for_status()
            payload = response.json()
            return SourceResult(
                self.name,
                handle,
                "ok",
                {
                    "login": payload.get("login"),
                    "name": payload.get("name"),
                    "company": payload.get("company"),
                    "location": payload.get("location"),
                    "public_repos": payload.get("public_repos"),
                },
            )
        except httpx.HTTPError as error:
            return SourceResult(self.name, handle, "error", error=str(error))
