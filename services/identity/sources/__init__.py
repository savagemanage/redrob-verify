"""Re-export public identity source adapters (free APIs only)."""

from services.identity.sources.codeforces import CodeforcesAdapter
from services.identity.sources.github import GitHubAdapter
from services.identity.sources.leetcode import LeetCodeAdapter
from services.identity.sources.stackoverflow import StackOverflowAdapter

__all__ = [
    "GitHubAdapter",
    "LeetCodeAdapter",
    "CodeforcesAdapter",
    "StackOverflowAdapter",
]
