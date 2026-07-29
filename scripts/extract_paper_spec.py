"""Extract reproducibility metadata and page text from the source paper.

This utility is intentionally limited to Stage 0 evidence collection. It does
not interpret equations or generate solver code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "references" / "AnEfficientGPU-basedHalpernAccelerating.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(pdf_path: Path) -> tuple[dict[str, Any], str]:
    reader = PdfReader(pdf_path)
    page_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text.append(f"\n\n===== PAGE {page_number} =====\n\n{page.extract_text() or ''}")

    metadata = {
        "file_name": pdf_path.name,
        "byte_size": pdf_path.stat().st_size,
        "sha256": sha256(pdf_path),
        "page_count": len(reader.pages),
        "pdf_metadata": {str(key): str(value) for key, value in (reader.metadata or {}).items()},
    }
    return metadata, "".join(page_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--text-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Source paper not found: {pdf_path}")

    metadata, text = extract(pdf_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))

    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
