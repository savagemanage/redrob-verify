"""Generate synthetic Indian identity documents (MIDV-2020-style, not real scans).

Templates (HTML source of truth) + raster capture simulation.
Aadhaar numbers intentionally fail Verhoeff so they cannot collide with real IDs.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harness.config_util import REPO_ROOT

OUTPUT_ROOT = REPO_ROOT / "results" / "indian_docs"
TEMPLATE_ROOT = REPO_ROOT / "tools" / "templates" / "indian_docs"
DOC_TYPES = ("aadhaar", "degree_certificate", "marksheet", "experience_certificate")

# Verhoeff multiplication table / permutation / inverse (public algorithm).
_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_checksum_digit(number_without_check: str) -> int:
    c = 0
    for i, ch in enumerate(reversed(number_without_check)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return _VERHOEFF_INV[c]


def verhoeff_ok(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def invalid_aadhaar(rng: random.Random) -> str:
    """12-digit Aadhaar-shaped number that deliberately fails Verhoeff."""
    # Real Aadhaar never starts with 0 or 1; keep that cosmetic rule.
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    correct = verhoeff_checksum_digit(body)
    wrong = (correct + rng.randint(1, 9)) % 10
    candidate = body + str(wrong)
    assert not verhoeff_ok(candidate)
    return candidate


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _fields_for(doc_type: str, index: int, rng: random.Random) -> dict[str, Any]:
    first = rng.choice(["Aarav", "Vihaan", "Aditi", "Ananya", "Kabir", "Ishaan", "Meera", "Diya"])
    last = rng.choice(["Sharma", "Patel", "Singh", "Reddy", "Iyer", "Nair", "Khan", "Das"])
    name = f"{first} {last}"
    dob = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1985, 2004)}"
    if doc_type == "aadhaar":
        return {
            "doc_type": doc_type,
            "name": name,
            "dob": dob,
            "gender": rng.choice(["M", "F"]),
            "aadhaar_number": invalid_aadhaar(rng),
            "address": f"{rng.randint(12, 99)} MG Road, Bengaluru, KA {rng.randint(560001, 560099)}",
            "vid": "".join(str(rng.randint(0, 9)) for _ in range(16)),
        }
    if doc_type == "degree_certificate":
        return {
            "doc_type": doc_type,
            "name": name,
            "degree": rng.choice(["B.Tech Computer Science", "B.E. Electronics", "B.Sc Mathematics"]),
            "university": rng.choice(
                ["Northstar Technical University", "Deccan Institute of Technology", "Sahyadri University"]
            ),
            "reg_no": f"NTU-{2018 + index % 6}-{1000 + index}",
            "year": str(2019 + index % 5),
            "grade": rng.choice(["First Class", "Distinction", "Second Class"]),
        }
    if doc_type == "marksheet":
        subjects = {
            "Mathematics": rng.randint(55, 98),
            "Physics": rng.randint(55, 98),
            "Chemistry": rng.randint(55, 98),
            "English": rng.randint(55, 98),
            "Computer Science": rng.randint(55, 98),
        }
        return {
            "doc_type": doc_type,
            "name": name,
            "board": "Central Board of Secondary Education",
            "roll_no": f"CBSE{2015 + index % 8}{100000 + index}",
            "subjects": subjects,
            "total": sum(subjects.values()),
            "result": "PASS",
        }
    # experience_certificate
    return {
        "doc_type": doc_type,
        "name": name,
        "employer": rng.choice(
            ["Orbital Systems Pvt Ltd", "Nimbus Analytics India", "Riverstone Softwares"]
        ),
        "designation": rng.choice(["Software Engineer", "Data Analyst", "QA Engineer"]),
        "from_date": f"0{rng.randint(1, 9)}/201{rng.randint(6, 9)}",
        "to_date": f"0{rng.randint(1, 9)}/202{rng.randint(1, 4)}",
        "employee_id": f"EMP-{8000 + index}",
    }


def fields_to_gt_text(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if key == "subjects" and isinstance(value, dict):
            parts.extend(f"{k}: {v}" for k, v in value.items())
        else:
            parts.append(str(value))
    return "\n".join(parts)


def write_html_template(doc_type: str, fields: dict[str, Any], path: Path) -> None:
    safe = {k: html.escape(str(v)) for k, v in fields.items() if k != "subjects"}
    body = "<br/>".join(f"<b>{html.escape(k)}</b>: {v}" for k, v in safe.items())
    if "subjects" in fields and isinstance(fields["subjects"], dict):
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in fields["subjects"].items()
        )
        body += f"<table border='1' cellpadding='4'>{rows}</table>"
    path.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'/><title>{doc_type}</title></head>"
        f"<body><h1>{html.escape(doc_type)}</h1>{body}</body></html>",
        encoding="utf-8",
    )


def render_document_image(doc_type: str, fields: dict[str, Any], seed: int) -> Image.Image:
    """Rasterize from the same field dict used in the HTML template (no real scans)."""
    rng = random.Random(seed)
    w, h = 900, 1200
    paper = Image.new("RGB", (w, h), (248, 244, 232))
    draw = ImageDraw.Draw(paper)
    draw.rectangle((30, 30, w - 30, h - 30), outline=(40, 40, 40), width=3)
    title = {
        "aadhaar": "UNIQUE IDENTIFICATION AUTHORITY OF INDIA — AADHAAR (SYNTHETIC)",
        "degree_certificate": "DEGREE CERTIFICATE (SYNTHETIC)",
        "marksheet": "MARKSHEET (SYNTHETIC)",
        "experience_certificate": "EXPERIENCE CERTIFICATE (SYNTHETIC)",
    }[doc_type]
    draw.text((50, 50), title, fill=(20, 20, 120), font=_font(18, bold=True))
    y = 120
    for key, value in fields.items():
        if key == "subjects" and isinstance(value, dict):
            draw.text((50, y), "Subjects:", fill=(0, 0, 0), font=_font(20, bold=True))
            y += 36
            for subj, mark in value.items():
                draw.text((70, y), f"{subj}: {mark}", fill=(0, 0, 0), font=_font(18))
                y += 30
            continue
        label = key.replace("_", " ").title()
        draw.text((50, y), f"{label}: {value}", fill=(0, 0, 0), font=_font(20))
        y += 40
    # Artificial face placeholder (geometric) — not a photo of a real person
    face = Image.new("RGB", (160, 200), (210, 190, 170))
    fd = ImageDraw.Draw(face)
    fd.ellipse((20, 30, 140, 170), fill=(180, 140, 110), outline=(80, 50, 40), width=2)
    paper.paste(face, (w - 220, 100))
    draw.text((50, h - 80), f"synthetic seed={seed} origin=synthetic_generated", fill=(90, 90, 90), font=_font(14))
    # mild paper grain
    arr = np.asarray(paper, dtype=np.float32)
    arr += rng.gauss(0, 2.0)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def simulate_capture(image: Image.Image, seed: int) -> Image.Image:
    """Phone-capture simulation: tilt, lighting, shadow, fold, clutter background, JPEG."""
    rng = np.random.default_rng(seed)
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # Background clutter canvas
    canvas = np.full((int(h * 1.25), int(w * 1.25), 3), 90, dtype=np.uint8)
    canvas[:] = (
        int(rng.integers(40, 120)),
        int(rng.integers(40, 120)),
        int(rng.integers(40, 120)),
    )
    noise = rng.normal(0, 18, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # Perspective tilt
    margin = 0.05
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [rng.uniform(0, w * margin), rng.uniform(0, h * margin)],
            [w - 1 - rng.uniform(0, w * margin), rng.uniform(0, h * margin)],
            [w - 1 - rng.uniform(0, w * margin), h - 1 - rng.uniform(0, h * margin)],
            [rng.uniform(0, w * margin), h - 1 - rng.uniform(0, h * margin)],
        ]
    )
    warped = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (w, h))
    # Lighting gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / w) * rng.uniform(-40, 40) + (yy / h) * rng.uniform(-30, 30)
    lit = np.clip(warped.astype(np.float32) + grad[..., None], 0, 255)
    # Soft shadow blob
    cy, cx = int(rng.uniform(0.2, 0.8) * h), int(rng.uniform(0.2, 0.8) * w)
    shadow = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.2 * min(h, w)) ** 2)))
    lit -= shadow[..., None] * rng.uniform(25, 55)
    # Fold crease
    crease_x = int(rng.uniform(0.3, 0.7) * w)
    lit[:, max(0, crease_x - 2) : crease_x + 3] *= rng.uniform(0.7, 0.85)
    lit = np.clip(lit, 0, 255).astype(np.uint8)
    # Paste onto clutter background
    y0, x0 = int(0.1 * canvas.shape[0]), int(0.1 * canvas.shape[1])
    canvas[y0 : y0 + h, x0 : x0 + w] = lit
    # JPEG recompression
    ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(35, 70))])
    if not ok:
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def generate(*, n_per_type: int, seed: int) -> dict[str, int]:
    if OUTPUT_ROOT.exists():
        import shutil

        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
    img_dir = OUTPUT_ROOT / "img"
    html_dir = OUTPUT_ROOT / "html"
    img_dir.mkdir(exist_ok=True)
    html_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    index = 0
    for doc_type in DOC_TYPES:
        for i in range(1, n_per_type + 1):
            index += 1
            rng = random.Random(seed + index * 997)
            fields = _fields_for(doc_type, index, rng)
            if doc_type == "aadhaar":
                assert not verhoeff_ok(str(fields["aadhaar_number"]))
            gt_text = fields_to_gt_text(fields)
            stem = f"{doc_type}_{index:04d}"
            html_path = html_dir / f"{stem}.html"
            write_html_template(doc_type, fields, html_path)
            # also keep a shared template skeleton once
            skeleton = TEMPLATE_ROOT / f"{doc_type}.html.j2"
            if not skeleton.is_file():
                skeleton.write_text(
                    "<!-- Field placeholders rendered by tools/gen_indian_docs.py -->\n"
                    f"<h1>{doc_type}</h1>\n{{{{ fields }}}}\n",
                    encoding="utf-8",
                )
            clean = render_document_image(doc_type, fields, seed + index)
            captured = simulate_capture(clean, seed + index * 13)
            rel = f"img/{stem}.jpg"
            captured.save(OUTPUT_ROOT / rel, quality=85)
            records.append(
                {
                    "id": f"ind_{index:04d}",
                    "path": rel,
                    "doc_type": doc_type,
                    "gt_text": gt_text,
                    "gt_fields": fields,
                    "origin": "synthetic_generated",
                    "html_template": f"html/{stem}.html",
                }
            )
    manifest = OUTPUT_ROOT / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Convenience copy layout for 1_ocr consumers
    ds1 = OUTPUT_ROOT / "1_ocr"
    ds1.mkdir(exist_ok=True)
    (ds1 / "img").mkdir(exist_ok=True)
    for row in records:
        src = OUTPUT_ROOT / row["path"]
        dst = ds1 / row["path"]
        if not dst.is_file():
            dst.write_bytes(src.read_bytes())
    with (ds1 / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"n": len(records), "per_type": n_per_type}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Indian document generator")
    parser.add_argument("--n-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = generate(n_per_type=args.n_per_type, seed=args.seed)
    print(json.dumps({"output": str(OUTPUT_ROOT), **summary}, indent=2))


if __name__ == "__main__":
    main()
