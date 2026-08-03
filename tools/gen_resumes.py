"""Build TC6 resumes anchored to real public GitHub (and optional other) handles.

Flow:
  1) Fetch ~N active GitHub users via public API
  2) Attach LeetCode/Codeforces/SO URLs when discoverable from profile/blog/bio
  3) Generate a resume body (LLM if OPENAI_API_KEY / RESUME_LLM_URL set, else template)
  4) Inject intentional discrepancies on half the set; record gt_discrepancies
  5) Render PDF with clickable profile hyperlinks for pdfplumber extraction
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from harness.config_util import REPO_ROOT

OUTPUT_ROOT = REPO_ROOT / "data" / "4_resume"
USER_AGENT = "redrob-verify-resume-gen/0.1 (research harness; contact: local)"


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf_with_links(path: Path, lines: list[str], links: list[tuple[str, str]]) -> None:
    """Minimal PDF-1.4 with URI link annotations (pdfplumber-readable)."""
    y_start = 740
    line_h = 18
    content_ops = ["BT", "/F1 12 Tf", f"50 {y_start} Td"]
    for i, line in enumerate(lines):
        if i:
            content_ops.append(f"0 -{line_h} Td")
        content_ops.append(f"({_pdf_escape(line)}) Tj")
    content_ops.append("ET")
    content = "\n".join(content_ops).encode("latin-1", errors="replace")

    annots: list[bytes] = []
    annot_refs: list[str] = []
    # Place link rects roughly beside matching URL lines
    for idx, (label, url) in enumerate(links):
        y = y_start - line_h * (3 + idx)  # after name/title/blank
        # PDF y grows up; MediaBox height 792
        y_ll = y - 4
        y_ur = y + 12
        annot_obj = (
            f"<< /Type /Annot /Subtype /Link /Rect [50 {y_ll} 560 {y_ur}] "
            f"/Border [0 0 0] /A << /S /URI /URI ({_pdf_escape(url)}) >> >>"
        ).encode("latin-1", errors="replace")
        annots.append(annot_obj)

    # Object layout: 1 catalog, 2 pages, 3 page, 4 font, 5 contents, 6.. annotations
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    ]
    annot_start = 6
    for i in range(len(annots)):
        annot_refs.append(f"{annot_start + i} 0 R")
    annots_arr = "[" + " ".join(annot_refs) + "]" if annot_refs else "[]"
    objects.append(
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> "
            b"/Contents 5 0 R /Annots "
            + annots_arr.encode()
            + b" >>"
        )
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
    objects.extend(annots)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    pdf.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(pdf)


def github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


# Well-known public GitHub logins used when API search is rate-limited / unauthenticated.
# Resumes still anchor to real profiles; live identity sources hit GitHub at eval time.
_SEED_LOGINS: tuple[str, ...] = (
    "torvalds", "gvanrossum", "fabpot", "antirez", "keon", "sindresorhus", "tj",
    "gaearon", "yyx990803", "a8m", "schacon", "mojombo", "defunkt", "pjhyett",
    "wycats", "dhh", "jashkenas", "mrdoob", "addyosmani", "paulirish", "getify",
    "substack", "isaacs", "creationix", "indexzero", "felixge", "visionmedia",
    "rauchg", "timuric", "shuding", "leeoniya", "Rich-Harris", "sveltejs",
    "evanw", "jakearchibald", "surma", "developit", "lukeed", "padolsey",
    "kennethreitz", "mitsuhiko", "pallets", "encode", "tiangolo", "samuelcolvin",
    "simonw", "willmcgugan", "Textualize", "fastapi", "Kludex",
    "psf", "pypa", "numpy", "scipy", "matplotlib",
    "pytorch", "huggingface", "explosion", "spaCy", "nltk", "Homebrew",
    "kubernetes", "golang", "rust-lang", "apple", "microsoft", "google",
    "facebook", "amzn", "netflix", "uber", "airbnb", "spotify",
    "hashicorp", "docker", "cncf", "prometheus", "grafana", "elastic",
    "redis", "ClickHouse", "nginx", "traefik", "caddyserver", "vercel", "netlify",
    "cloudflare", "digitalocean", "linode", "hetzneronline", "ovh",
    "rails", "django", "laravel", "spring-projects", "dotnet", "JetBrains",
    "jenkinsci", "gitlabhq", "bitnami", "bitwarden",
    "obsproject", "godotengine", "blender", "gimp", "inkscape", "kde",
    "gnome", "freedesktop", "wayland", "xorg", "swaywm",
    "neovim", "vim", "emacs-mirror", "helix-editor", "zed-industries",
    "astral-sh", "charliermarsh", "BurntSushi", "sharkdp", "junegunn",
)


def _seed_user(login: str) -> dict[str, Any]:
    return {
        "login": login,
        "name": login,
        "company": None,
        "bio": None,
        "blog": None,
        "location": None,
        "html_url": f"https://github.com/{login}",
        "public_repos": 1,
        "repos": [{"name": login, "language": None, "description": None, "stargazers_count": 0}],
    }


def fetch_active_github_users(
    client: httpx.Client, n: int, *, seed: int, allow_seeds: bool = True
) -> list[dict[str, Any]]:
    """Collect users that have public repos and recent activity signals."""
    rng = random.Random(seed)
    # Search queries biased toward accounts with repos
    queries = [
        "repos:>3 followers:>10",
        "repos:>5 language:Python",
        "repos:>5 language:JavaScript",
        "repos:>3 location:India",
        "repos:>3 location:Bangalore",
    ]
    found: dict[str, dict[str, Any]] = {}
    page = 1
    rate_backoff = 5.0
    search_failures = 0
    use_search = bool(os.getenv("GITHUB_TOKEN")) or not allow_seeds
    if not use_search:
        print("no GITHUB_TOKEN — using curated public seed logins", flush=True)
    while use_search and len(found) < n and page <= 20:
        q = rng.choice(queries)
        resp = client.get(
            "https://api.github.com/search/users",
            params={"q": q, "per_page": 30, "page": page},
            headers=github_headers(),
            timeout=30.0,
        )
        if resp.status_code in {403, 429}:
            search_failures += 1
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else rate_backoff
            print(f"github rate-limited status={resp.status_code} sleep={wait:.0f}s", flush=True)
            if allow_seeds and search_failures >= 2:
                print("falling back to curated public GitHub seed logins", flush=True)
                break
            time.sleep(min(wait, 30.0))
            rate_backoff = min(rate_backoff * 1.5, 120.0)
            continue
        resp.raise_for_status()
        rate_backoff = 5.0
        search_failures = 0
        for item in resp.json().get("items") or []:
            login = item.get("login")
            if not login or login in found:
                continue
            detail = client.get(
                f"https://api.github.com/users/{login}",
                headers=github_headers(),
                timeout=30.0,
            )
            if detail.status_code in {403, 429}:
                time.sleep(float(detail.headers.get("Retry-After") or 20))
                continue
            if detail.status_code != 200:
                continue
            user = detail.json()
            if int(user.get("public_repos") or 0) < 1:
                continue
            repos = client.get(
                f"https://api.github.com/users/{login}/repos",
                params={"sort": "updated", "per_page": 5},
                headers=github_headers(),
                timeout=30.0,
            )
            if repos.status_code in {403, 429}:
                time.sleep(float(repos.headers.get("Retry-After") or 20))
                continue
            repo_payload = repos.json() if repos.status_code == 200 else []
            if not isinstance(repo_payload, list) or not repo_payload:
                continue
            found[login] = {
                "login": login,
                "name": user.get("name") or login,
                "company": user.get("company"),
                "bio": user.get("bio"),
                "blog": user.get("blog"),
                "location": user.get("location"),
                "html_url": user.get("html_url"),
                "public_repos": user.get("public_repos"),
                "repos": [
                    {
                        "name": r.get("name"),
                        "language": r.get("language"),
                        "description": r.get("description"),
                        "stargazers_count": r.get("stargazers_count"),
                    }
                    for r in repo_payload[:5]
                    if isinstance(r, dict)
                ],
            }
            if len(found) >= n:
                break
            time.sleep(0.2)
        page += 1
        time.sleep(0.5)

    if allow_seeds and len(found) < n:
        seeds = list(dict.fromkeys(_SEED_LOGINS))  # preserve order, unique
        rng.shuffle(seeds)
        live = bool(os.getenv("GITHUB_TOKEN"))
        for login in seeds:
            if login in found:
                continue
            if not live:
                found[login] = _seed_user(login)
            else:
                detail = client.get(
                    f"https://api.github.com/users/{login}",
                    headers=github_headers(),
                    timeout=20.0,
                )
                if detail.status_code == 200:
                    user = detail.json()
                    repos = client.get(
                        f"https://api.github.com/users/{login}/repos",
                        params={"sort": "updated", "per_page": 5},
                        headers=github_headers(),
                        timeout=20.0,
                    )
                    repo_payload = repos.json() if repos.status_code == 200 else []
                    if not isinstance(repo_payload, list):
                        repo_payload = []
                    found[login] = {
                        "login": login,
                        "name": user.get("name") or login,
                        "company": user.get("company"),
                        "bio": user.get("bio"),
                        "blog": user.get("blog"),
                        "location": user.get("location"),
                        "html_url": user.get("html_url") or f"https://github.com/{login}",
                        "public_repos": user.get("public_repos"),
                        "repos": [
                            {
                                "name": r.get("name"),
                                "language": r.get("language"),
                                "description": r.get("description"),
                                "stargazers_count": r.get("stargazers_count"),
                            }
                            for r in repo_payload[:5]
                            if isinstance(r, dict)
                        ]
                        or _seed_user(login)["repos"],
                    }
                else:
                    found[login] = _seed_user(login)
            if len(found) % 20 == 0 or len(found) >= n:
                print(f"seed users {len(found)}/{n}", flush=True)
            if len(found) >= n:
                break
            if live:
                time.sleep(0.05)
    return list(found.values())[:n]


def discover_extra_profiles(user: dict[str, Any]) -> dict[str, str]:
    profiles = {"github": user["html_url"]}
    blob = " ".join(
        str(x)
        for x in (user.get("blog"), user.get("bio"), *(r.get("description") for r in user.get("repos") or []))
        if x
    )
    patterns = {
        "leetcode": r"(?:https?://)?(?:www\.)?leetcode\.com/(?:u/)?([A-Za-z0-9_-]+)",
        "codeforces": r"(?:https?://)?(?:www\.)?codeforces\.com/profile/([A-Za-z0-9_-]+)",
        "stackoverflow": r"(?:https?://)?(?:www\.)?stackoverflow\.com/users/\d+/([A-Za-z0-9_-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, blob, re.I)
        if match:
            handle = match.group(1)
            if key == "leetcode":
                profiles[key] = f"https://leetcode.com/u/{handle}"
            elif key == "codeforces":
                profiles[key] = f"https://codeforces.com/profile/{handle}"
            else:
                profiles[key] = f"https://stackoverflow.com/users/{quote(handle)}"
    return profiles


def llm_resume_text(user: dict[str, Any], profiles: dict[str, str]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("RESUME_LLM_URL", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("RESUME_LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    prompt = (
        "Write a concise one-page resume in plain text for an Indian software jobseeker. "
        "Use ONLY the provided GitHub profile facts. Include experience and skills consistent "
        "with their public repositories. Do not invent employers that contradict the profile.\n\n"
        f"PROFILE_JSON:\n{json.dumps(user, ensure_ascii=False)}\n"
        f"PROFILE_LINKS:\n{json.dumps(profiles, ensure_ascii=False)}\n"
    )
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                base,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You write factual resumes from public profile JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"]).strip()
    except Exception:
        return None


def template_resume_text(user: dict[str, Any], profiles: dict[str, str]) -> str:
    langs = sorted(
        {
            str(r.get("language"))
            for r in user.get("repos") or []
            if r.get("language")
        }
    )
    repo_lines = [
        f"- {r.get('name')}: {r.get('description') or 'public repository'} ({r.get('language') or 'n/a'})"
        for r in user.get("repos") or []
    ]
    return "\n".join(
        [
            str(user.get("name") or user["login"]),
            "Software Engineer",
            str(user.get("location") or "India"),
            str(user.get("bio") or "Engineer with public open-source work."),
            "Skills: " + (", ".join(langs) if langs else "Software development"),
            "Public projects:",
            *repo_lines,
            "Profiles:",
            *[f"{k}: {v}" for k, v in profiles.items()],
        ]
    )


def inject_discrepancies(
    text: str, profiles: dict[str, str], rng: random.Random
) -> tuple[str, list[dict[str, Any]]]:
    discrepancies: list[dict[str, Any]] = []
    lines = text.splitlines()
    # 1) fake tenure
    fake_tenure = f"Senior Engineer at Nonexistent Labs ({2015 + rng.randint(0, 3)}–{2019 + rng.randint(0, 4)})"
    lines.insert(min(6, len(lines)), fake_tenure)
    discrepancies.append(
        {
            "field": "employment_tenure",
            "action": "insert_fake_employer_period",
            "injected": fake_tenure,
            "reason": "intentional mismatch vs public GitHub timeline",
        }
    )
    # 2) fake skill
    fake_skill = rng.choice(["COBOL mainframe tuning", "SAP ABAP wizardry", "Quantum FPGA place-and-route"])
    lines.append(f"Additional skill: {fake_skill}")
    discrepancies.append(
        {
            "field": "skills",
            "action": "add_unattested_skill",
            "injected": fake_skill,
        }
    )
    # 3) education inflation
    edu = "PhD in Artificial General Intelligence, Invisible Institute"
    lines.append(edu)
    discrepancies.append(
        {
            "field": "education",
            "action": "inflate_degree",
            "injected": edu,
        }
    )
    # 4) nonexistent project
    proj = "Led Project Chimera: private 10M LOC rewrite (no public artifact)"
    lines.append(proj)
    discrepancies.append(
        {
            "field": "projects",
            "action": "add_unverifiable_project",
            "injected": proj,
        }
    )
    # keep profiles truthful so TC6 network still anchors
    _ = profiles
    return "\n".join(lines), discrepancies


def generate(*, n: int, seed: int) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    resume_dir = OUTPUT_ROOT / "resume"
    resume_dir.mkdir(exist_ok=True)
    rng = random.Random(seed)
    with httpx.Client(follow_redirects=True) as client:
        users = fetch_active_github_users(client, n, seed=seed)
    if len(users) < max(1, n // 5):
        raise RuntimeError(
            f"only collected {len(users)} GitHub users (need network + optional GITHUB_TOKEN)"
        )

    records: list[dict[str, Any]] = []
    for i, user in enumerate(users, start=1):
        profiles = discover_extra_profiles(user)
        body = llm_resume_text(user, profiles) or template_resume_text(user, profiles)
        discrepancies: list[dict[str, Any]] = []
        if i % 2 == 0:
            body, discrepancies = inject_discrepancies(body, profiles, rng)
        lines = body.splitlines()
        # Ensure profile URLs appear as lines for link placement
        link_pairs = list(profiles.items())
        for key, url in link_pairs:
            if url not in body:
                lines.append(f"{key}: {url}")
        rid = f"rs_real_{i:04d}"
        rel = f"resume/{rid}.pdf"
        write_pdf_with_links(OUTPUT_ROOT / rel, lines[:80], [(k, v) for k, v in link_pairs])
        records.append(
            {
                "id": rid,
                "path": rel,
                "profiles": profiles,
                "origin": "synthetic_generated",
                "gt_discrepancies": discrepancies,
                "anchors": {
                    "github_login": user["login"],
                    "public_repos": user.get("public_repos"),
                    "repo_names": [r.get("name") for r in user.get("repos") or []],
                    "llm_used": bool(os.getenv("OPENAI_API_KEY")),
                },
            }
        )

    # Write directly into data/4_resume/
    (OUTPUT_ROOT / "resume").mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # jmeter csv
    import csv

    with (OUTPUT_ROOT / "jmeter_resumes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "filepath"))
        writer.writeheader()
        for row in records:
            writer.writerow({"id": row["id"], "filepath": str((OUTPUT_ROOT / row["path"]).resolve())})
    return {
        "n": len(records),
        "with_discrepancies": sum(1 for r in records if r["gt_discrepancies"]),
        "output": str(OUTPUT_ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resumes anchored to real GitHub users")
    parser.add_argument("--n", type=int, default=20, help="Target count (use 100 when rate limits allow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(generate(n=args.n, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
