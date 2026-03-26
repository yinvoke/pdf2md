#!/usr/bin/env python3
"""Benchmark: Docling PDF→MD conversion on test PDFs.

Runs from project root venv (uses app.converter).
Outputs JSON results to benchmark/results/docling.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.converter import convert_pdf_to_markdown

EXAMPLE_DIR = PROJECT_ROOT / "example"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

TEST_PDFS = [
    ("1225027862.PDF", "小米2025年报(MSungHK)"),
    ("1224680620.PDF", "小米2025半年报(HYQiHei)"),
    ("新和成_2024 年年度报告.pdf", "新和成2024年报(A股)"),
]


def garbled_ratio(text: str, sample_size: int = 2000) -> float:
    """Calculate ratio of suspicious Unicode code points."""
    stripped = re.sub(r"[\s#|>\-*_`\[\]()!]", "", text)
    if len(stripped) < 30:
        return 0.0
    sample = stripped[:sample_size]
    suspicious = 0
    for ch in sample:
        cp = ord(ch)
        if (
            0x3400 <= cp <= 0x4DBF
            or 0x2C80 <= cp <= 0x2DFF
            or 0x1D00 <= cp <= 0x1DFF
            or 0x2300 <= cp <= 0x23FF
            or 0x2700 <= cp <= 0x27BF
            or 0x27C0 <= cp <= 0x27EF
            or 0x2980 <= cp <= 0x29FF
            or 0x2400 <= cp <= 0x243F
            or 0x1F00 <= cp <= 0x1FFF
        ):
            suspicious += 1
    return suspicious / len(sample)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for filename, label in TEST_PDFS:
        pdf_path = EXAMPLE_DIR / filename
        if not pdf_path.exists():
            print(f"SKIP: {filename} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Docling: {label} ({filename})")
        print(f"{'=' * 60}")

        t0 = time.time()
        markdown = convert_pdf_to_markdown(str(pdf_path))
        duration = time.time() - t0

        ratio = garbled_ratio(markdown)
        chars = len(markdown)
        preview = markdown[:500].replace("\n", "\\n")

        out_name = Path(filename).stem + "_docling.md"
        (OUTPUT_DIR / out_name).write_text(markdown, encoding="utf-8")

        result = {
            "tool": "docling",
            "mode": "auto",
            "file": filename,
            "label": label,
            "chars": chars,
            "duration_s": round(duration, 2),
            "garbled_ratio": round(ratio, 4),
            "preview": preview,
        }
        results.append(result)
        print(f"  chars={chars}  time={duration:.1f}s  garbled={ratio:.4f}")

    out_path = RESULTS_DIR / "docling.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
