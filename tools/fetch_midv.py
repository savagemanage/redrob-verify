#!/usr/bin/env python3
"""Download MIDV-2020 stage-1 assets from the Smart Engines FTP (author source).

Source (required): ftp://smartengines.com/midv-2020
Do NOT use the L3i-Share mirror for this fetch (commercial-permission clause).

Stage 1 (default):
  - scan_upright.tar + scan_rotated.tar  (~2k scans)
  - photo.tar                            (~1k photos)
  - templates.tar
  - license.txt / readme.txt / md5.txt

Video clips are skipped (stage 2 later).

Resume-safe FTP RETR with REST; verifies MD5 from md5.txt after each file.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from ftplib import FTP, error_perm
from pathlib import Path

from harness.config_util import REPO_ROOT

FTP_HOST = "smartengines.com"
FTP_ROOT = "midv-2020"
OUT_ROOT = REPO_ROOT / "results" / "midv_archives"

STAGE1_ARCHIVES = (
    "scan_upright.tar",
    "scan_rotated.tar",
    "photo.tar",
    "templates.tar",
)
META_FILES = ("license.txt", "readme.txt", "md5.txt")


def _parse_md5(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 32:
            mapping[parts[1].lstrip("*")] = parts[0].lower()
    return mapping


def _md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _ftp_size(ftp: FTP, remote: str) -> int | None:
    try:
        return ftp.size(remote)
    except Exception:
        return None


def _connect() -> FTP:
    ftp = FTP(FTP_HOST, timeout=600)
    ftp.login()
    ftp.set_pasv(True)
    return ftp


def download_file(ftp: FTP, remote: str, dest: Path, *, retries: int = 8) -> FTP:
    """Download with resume. Returns a live FTP (may reconnect after timeouts)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _ftp_size(ftp, remote)
    offset = dest.stat().st_size if dest.is_file() else 0
    if remote_size is not None and offset == remote_size and remote_size > 0:
        print(f"skip (complete): {dest.name} ({offset} bytes)", flush=True)
        return ftp
    if offset and remote_size is not None and offset > remote_size:
        dest.unlink()
        offset = 0

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        offset = dest.stat().st_size if dest.is_file() else 0
        if remote_size is not None and offset == remote_size and remote_size > 0:
            return ftp
        mode = "ab" if offset else "wb"
        print(
            f"fetch {remote} → {dest} (resume={offset}, attempt={attempt}/{retries})",
            flush=True,
        )
        try:
            with dest.open(mode) as out:

                def _write(block: bytes) -> None:
                    out.write(block)

                if offset:
                    ftp.sendcmd(f"REST {offset}")
                ftp.retrbinary(f"RETR {remote}", _write, blocksize=1 << 20)
            if remote_size is not None:
                got = dest.stat().st_size
                if got != remote_size:
                    raise RuntimeError(
                        f"size mismatch {dest.name}: got {got}, expected {remote_size}"
                    )
            return ftp
        except error_perm as e:
            if offset:
                print(f"REST failed ({e}); restarting {dest.name}", flush=True)
                dest.unlink(missing_ok=True)
                last_exc = e
                continue
            raise
        except (TimeoutError, OSError, EOFError, RuntimeError) as e:
            last_exc = e
            print(
                f"transfer interrupted ({type(e).__name__}: {e}); reconnecting…",
                flush=True,
            )
            try:
                ftp.quit()
            except Exception:
                pass
            ftp = _connect()
            # refresh size after reconnect
            remote_size = _ftp_size(ftp, remote) or remote_size
    assert last_exc is not None
    raise last_exc


def fetch_stage1(*, out_root: Path = OUT_ROOT) -> dict[str, str]:
    out_root.mkdir(parents=True, exist_ok=True)
    ftp = _connect()

    # meta first
    for name in META_FILES:
        ftp = download_file(ftp, f"{FTP_ROOT}/{name}", out_root / name)

    md5_map = _parse_md5((out_root / "md5.txt").read_text(encoding="utf-8", errors="replace"))
    dataset_dir = out_root / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    results: dict[str, str] = {}
    for name in STAGE1_ARCHIVES:
        dest = dataset_dir / name
        ftp = download_file(ftp, f"{FTP_ROOT}/dataset/{name}", dest)
        expected = md5_map.get(name)
        if expected:
            digest = _md5_file(dest)
            if digest != expected:
                raise RuntimeError(f"MD5 fail {name}: got {digest}, expected {expected}")
            print(f"md5 ok: {name}", flush=True)
            results[name] = digest
        else:
            results[name] = _md5_file(dest)
            print(f"md5 (no catalog entry): {name} = {results[name]}", flush=True)

    try:
        ftp.quit()
    except Exception:
        pass
    summary = out_root / "fetch_stage1.json"
    import json

    summary.write_text(
        json.dumps(
            {
                "source": f"ftp://{FTP_HOST}/{FTP_ROOT}",
                "stage": 1,
                "archives": results,
                "skipped": ["clips.tar", "clips_video.tar", "*_tif.tar"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MIDV-2020 stage-1 from Smart Engines FTP")
    parser.add_argument("--out", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    try:
        fetch_stage1(out_root=args.out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
