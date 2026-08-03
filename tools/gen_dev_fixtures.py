"""Generate deterministic plumbing fixtures for local harness smoke tests.

Output: ``results/dev_fixtures``. Every row ``origin=dev_fixture``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harness.config_util import REPO_ROOT, load_config

OUTPUT_ROOT = REPO_ROOT / "results" / "dev_fixtures"
LAYOUTS = ("degree_certificate", "experience_certificate", "id_card", "marksheet")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    fonts = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in fonts:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)


def _paper(size: tuple[int, int], seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 5, (size[1], size[0], 1))
    base = np.full((size[1], size[0], 3), (245, 239, 221), dtype=np.float32)
    base += grain
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(150):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        radius = int(rng.integers(1, 5))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(130, 110, 70, 10))
    return image


def procedural_face(seed: int, size: tuple[int, int] = (180, 220)) -> Image.Image:
    """Return a recognisable geometric face whose identity is controlled by seed."""
    rng = random.Random(seed)
    image = Image.new("RGB", size, (218, 228, 235))
    draw = ImageDraw.Draw(image)
    skin = (rng.randrange(155, 220), rng.randrange(105, 175), rng.randrange(75, 145))
    hair = (rng.randrange(20, 80), rng.randrange(15, 65), rng.randrange(10, 50))
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse((cx - 58, cy - 78, cx + 58, cy + 76), fill=skin, outline=(70, 45, 30), width=3)
    draw.pieslice((cx - 63, cy - 86, cx + 63, cy + 14), 180, 360, fill=hair)
    eye_y = cy - 20 + rng.randrange(-5, 6)
    eye_space = 30 + rng.randrange(-5, 6)
    for x in (cx - eye_space, cx + eye_space):
        draw.ellipse((x - 10, eye_y - 7, x + 10, eye_y + 7), fill=(250, 250, 245))
        draw.ellipse((x - 4, eye_y - 4, x + 4, eye_y + 4), fill=(25, 35, 45))
    nose = (cx + rng.randrange(-7, 8), cy + 8)
    draw.polygon([(nose[0], nose[1] - 13), (nose[0] - 7, nose[1] + 12), (nose[0] + 8, nose[1] + 12)], fill=(130, 80, 60))
    mouth_y = cy + 45 + rng.randrange(-4, 5)
    draw.arc((cx - 26, mouth_y - 8, cx + 26, mouth_y + 14), 5, 175, fill=(100, 35, 35), width=4)
    return image


def _document_text(doc_type: str, index: int) -> list[str]:
    name = f"Avery{index:03d} Sharma"
    if doc_type == "degree_certificate":
        return [
            "NORTHSTAR TECHNICAL UNIVERSITY",
            "DEGREE CERTIFICATE",
            f"This certifies that {name}",
            "has completed Bachelor of Engineering",
            f"Registration No: NTU-{2020 + index % 5}-{1000 + index}",
            "Issued by the Registrar",
        ]
    if doc_type == "experience_certificate":
        return [
            "ORBITAL SYSTEMS PRIVATE LIMITED",
            "EXPERIENCE CERTIFICATE",
            f"This is to certify that {name}",
            "worked as Software Engineer from 2021 to 2024",
            f"Employee ID: OS-{8000 + index}",
            "Authorized Human Resources Officer",
        ]
    if doc_type == "id_card":
        return [
            "CITY CITIZEN IDENTITY CARD",
            f"Name: {name}",
            f"Date of Birth: 199{index % 10}-0{index % 9 + 1}-15",
            f"Card Number: CID-{index:06d}",
            "Address: 42 Fixture Avenue, Test City",
            "Issuing Authority",
        ]
    return [
        "NORTHSTAR TECHNICAL UNIVERSITY",
        "SEMESTER MARKSHEET",
        f"Student Name: {name}",
        f"Roll Number: MS-{2020 + index % 5}-{1000 + index}",
        "Programme: Computer Engineering",
        "Result: PASS",
    ]


def render_document(doc_type: str, index: int, seed: int) -> tuple[Image.Image, str]:
    width, height = (1000, 660) if doc_type == "id_card" else (900, 1180)
    image = _paper((width, height), seed)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, body_font, small_font = _font(30, bold=True), _font(21), _font(17)
    lines = _document_text(doc_type, index)
    draw.rectangle((28, 28, width - 28, height - 28), outline=(82, 72, 52, 180), width=4)
    y = 75
    for line_number, line in enumerate(lines):
        font = title_font if line_number < 2 else body_font
        draw.text((70, y), line, font=font, fill=(30, 35, 38, 255))
        y += 50 if line_number < 2 else 38

    table_x, table_y = 70, y + 35
    table_w = width - 140 if doc_type != "id_card" else width - 350
    row_h = 37
    for row in range(4):
        draw.rectangle((table_x, table_y + row * row_h, table_x + table_w, table_y + (row + 1) * row_h), outline=(70, 75, 80, 180), width=2)
        draw.line((table_x + table_w * 0.62, table_y + row * row_h, table_x + table_w * 0.62, table_y + (row + 1) * row_h), fill=(70, 75, 80, 180), width=2)
        left = ("Subject", "Document reference", "Verification status", "Date")[row]
        right = ("Engineering", f"REF-{index:05d}", "VALID", f"202{index % 5}-06-15")[row]
        draw.text((table_x + 12, table_y + row * row_h + 8), left, font=small_font, fill=(20, 25, 28, 255))
        draw.text((table_x + int(table_w * 0.65), table_y + row * row_h + 8), right, font=small_font, fill=(20, 25, 28, 255))

    photo = procedural_face(seed + 50_000, (130, 160))
    photo_x, photo_y = width - 205, 100
    image.paste(photo, (photo_x, photo_y))
    draw.rectangle((photo_x, photo_y, photo_x + 130, photo_y + 160), outline=(30, 40, 50, 255), width=3)
    draw.text((photo_x, photo_y + 165), "PHOTO", font=small_font, fill=(30, 35, 38, 255))
    stamp_x, stamp_y = width - 190, height - 180
    draw.ellipse((stamp_x, stamp_y, stamp_x + 125, stamp_y + 125), outline=(150, 35, 35, 180), width=5)
    draw.text((stamp_x + 18, stamp_y + 52), "STAMP", font=small_font, fill=(150, 35, 35, 180))
    draw.line((75, height - 110, 330, height - 110), fill=(50, 50, 50, 160), width=2)
    draw.text((75, height - 98), "Authorised signature", font=small_font, fill=(30, 35, 38, 255))
    return image, "\n".join(lines)


def _write_manifest(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _escape_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_resume_pdf(path: Path, name: str, profiles: dict[str, str]) -> None:
    lines = [name, "Software Engineer", "Synthetic development resume", *profiles.values()]
    stream = ["BT", "/F1 18 Tf", "72 740 Td"]
    for line in lines:
        stream.extend((f"({_escape_pdf(line)}) Tj", "0 -28 Td"))
    stream.append("ET")
    content = "\n".join(stream).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    pdf.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(pdf)


def generate(seed: int, *, n_ocr: int, n_face: int, n_forgery: int, n_resume: int) -> dict[str, int]:
    """Regenerate every development fixture, without writing outside OUTPUT_ROOT."""
    if OUTPUT_ROOT.resolve() != (REPO_ROOT / "results" / "dev_fixtures").resolve():
        raise RuntimeError("development fixture output path must remain results/dev_fixtures")
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    ocr_root = OUTPUT_ROOT / "1_ocr"
    (ocr_root / "img").mkdir(parents=True)
    ocr_records = []
    for index in range(1, n_ocr + 1):
        doc_type = LAYOUTS[(index - 1) % len(LAYOUTS)]
        image, gt_text = render_document(doc_type, index, seed + index)
        filename = f"ocr_{index:04d}.png"
        image.save(ocr_root / "img" / filename)
        ocr_records.append({"id": f"ocr_{index:04d}", "path": f"img/{filename}", "doc_type": doc_type, "gt_text": gt_text, "origin": "dev_fixture"})
    _write_manifest(ocr_root / "manifest.jsonl", ocr_records)

    forgery_root = OUTPUT_ROOT / "2_forgery"
    (forgery_root / "authentic").mkdir(parents=True)
    forgery_records = []
    for index in range(1, n_forgery + 1):
        doc_type = LAYOUTS[(index - 1) % len(LAYOUTS)]
        image, _ = render_document(doc_type, index, seed + 10_000 + index)
        filename = f"authentic_{index:04d}.png"
        image.save(forgery_root / "authentic" / filename)
        forgery_records.append(
            {
                "id": f"auth_{index:04d}",
                "path": f"authentic/{filename}",
                "label": 0,
                "origin": "dev_fixture",
                "fabrication": "authentic",
            }
        )
    _write_manifest(forgery_root / "manifest.jsonl", forgery_records)

    # DS-3: never build same-identity pairs by augmenting one image (measures
    # augment robustness, not verification). Different-identity pairs only for
    # plumbing smoke tests. Same-identity calibration → MIDV photo/scan/video
    # via tools/build_midv_face_pairs.py (user chooses public/synthetic source).
    face_root = OUTPUT_ROOT / "3_face"
    (face_root / "doc").mkdir(parents=True)
    (face_root / "selfie").mkdir(parents=True)
    face_records = []
    for offset in range(1, n_face + 1):
        a_name, b_name = f"fc_{offset:04d}_a.png", f"fc_{offset:04d}_b.png"
        procedural_face(seed + 30_000 + offset).save(face_root / "doc" / a_name)
        procedural_face(seed + 40_000 + offset).save(face_root / "selfie" / b_name)
        face_records.append(
            {
                "id": f"fc_{offset:04d}",
                "img_a": f"doc/{a_name}",
                "img_b": f"selfie/{b_name}",
                "same": False,
                "origin": "dev_fixture",
            }
        )
    _write_manifest(face_root / "manifest.jsonl", face_records)

    resume_root = OUTPUT_ROOT / "4_resume"
    (resume_root / "resume").mkdir(parents=True)
    resume_records = []
    with (resume_root / "jmeter_resumes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "path"))
        writer.writeheader()
        for index in range(1, n_resume + 1):
            # Well-known public profiles so TC6 issues real network lookups (not stubs).
            profiles = {
                "github": "https://github.com/octocat",
                "leetcode": "https://leetcode.com/u/leetcode",
                "codeforces": "https://codeforces.com/profile/tourist",
                "stackoverflow": "https://stackoverflow.com/users/1/jeff-atwood",
            }
            filename = f"rs_{index:04d}.pdf"
            _write_resume_pdf(resume_root / "resume" / filename, f"Fixture Candidate {index}", profiles)
            resume_records.append({"id": f"rs_{index:04d}", "path": f"resume/{filename}", "profiles": profiles, "origin": "dev_fixture"})
            writer.writerow({"id": f"rs_{index:04d}", "path": str((resume_root / "resume" / filename).resolve())})
    _write_manifest(resume_root / "manifest.jsonl", resume_records)
    return {"1_ocr": n_ocr, "2_forgery": n_forgery, "3_face": len(face_records), "4_resume": n_resume}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic development-only fixtures")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-ocr", type=int, default=40)
    parser.add_argument(
        "--n-face",
        type=int,
        default=50,
        help="different-identity smoke pairs only (no single-image augment same pairs)",
    )
    parser.add_argument("--n-forgery", type=int, default=50)
    parser.add_argument("--n-resume", type=int, default=20)
    args = parser.parse_args()
    if min(args.n_ocr, args.n_face, args.n_forgery, args.n_resume) < 1:
        parser.error("all fixture counts must be positive")
    cfg = load_config(args.config)
    counts = generate(int(cfg.get("seed", 42)), n_ocr=args.n_ocr, n_face=args.n_face, n_forgery=args.n_forgery, n_resume=args.n_resume)
    print(f"Wrote development fixtures under {OUTPUT_ROOT}")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
