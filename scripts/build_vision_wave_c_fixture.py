#!/usr/bin/env python3
"""Build the fixed Wave C document/embedded-image benchmark corpus."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/data/vision_wave_b_frozen/images"
OUT = ROOT / "tests/data/vision_wave_c_frozen"
FILES = OUT / "files"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf(image_names: list[str], output: Path) -> None:
    import fitz

    doc = fitz.open()
    for name in image_names:
        raw = (SOURCE / name).read_bytes()
        page = doc.new_page(width=720, height=360)
        page.insert_image(page.rect, stream=raw, keep_proportion=False)
    doc.set_metadata(
        {
            "title": "Omega Walls Vision Wave C fixture",
            "author": "Omega Walls",
            "creator": "fixture-builder",
            "producer": "fixture-builder",
            "creationDate": "D:20260101000000Z",
            "modDate": "D:20260101000000Z",
        }
    )
    output.write_bytes(doc.tobytes(garbage=4, deflate=True, clean=True))
    doc.close()


def _docx(image_names: list[str], output: Path) -> None:
    from docx import Document

    document = Document()
    for name in image_names:
        document.add_picture(BytesIO((SOURCE / name).read_bytes()))
    tmp = output.with_suffix(".tmp.docx")
    document.save(tmp)
    # Normalize ZIP metadata so rebuilding produces stable bytes.
    with (
        zipfile.ZipFile(tmp, "r") as src,
        zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as dst,
    ):
        for name in sorted(src.namelist()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            dst.writestr(info, src.read(name))
    tmp.unlink(missing_ok=True)


def _html(image_name: str, output: Path, *, include_remote: bool = True) -> None:
    encoded = base64.b64encode((SOURCE / image_name).read_bytes()).decode("ascii")
    remote = (
        '<img src="https://169.254.169.254/latest/meta-data/iam/">'
        if include_remote
        else ""
    )
    output.write_text(
        f'<!doctype html><html><body><img src="data:image/png;base64,{encoded}">{remote}</body></html>',
        encoding="utf-8",
    )


def main() -> int:
    FILES.mkdir(parents=True, exist_ok=True)
    specs = [
        ("VWC-A-001", "malicious", "secret_exfiltration", "pdf", ["VWB-A-002.png"]),
        (
            "VWC-A-002",
            "malicious",
            "override_instructions",
            "pdf",
            ["VWB-B-006.png", "VWB-A-007.png"],
        ),
        ("VWC-A-003", "malicious", "tool_or_action_abuse", "docx", ["VWB-A-009.png"]),
        (
            "VWC-A-004",
            "malicious",
            "policy_evasion",
            "docx",
            ["VWB-B-003.png", "VWB-A-008.png"],
        ),
        ("VWC-A-005", "malicious", "secret_exfiltration", "html", ["VWB-A-006.png"]),
        ("VWC-B-001", "benign", None, "pdf", ["VWB-B-001.png"]),
        ("VWC-B-002", "benign", None, "docx", ["VWB-B-008.png"]),
        ("VWC-B-003", "benign", None, "html", ["VWB-B-006.png"]),
    ]
    mime_by_kind = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "html": "text/html",
    }
    rows = []
    for sample_id, label, target_wall, kind, images in specs:
        path = FILES / f"{sample_id}.{kind}"
        if kind == "pdf":
            _pdf(images, path)
            expected_kinds = ["pdf_page"] * len(images)
        elif kind == "docx":
            _docx(images, path)
            expected_kinds = ["docx_embedded"] * len(images)
        else:
            _html(images[0], path)
            expected_kinds = ["html_data_uri"]
        rows.append(
            {
                "id": sample_id,
                "label": label,
                "target_wall": target_wall,
                "file": f"files/{path.name}",
                "filename": path.name,
                "mime": mime_by_kind[kind],
                "format": kind,
                "sha256": _sha(path),
                "expected_asset_count": len(expected_kinds),
                "expected_source_kinds": expected_kinds,
                "multi_image": len(expected_kinds) > 1,
                "contains_remote_image_reference": kind == "html",
            }
        )
    manifest = OUT / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
