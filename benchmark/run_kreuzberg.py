#!/usr/bin/env python3
"""Benchmark: kreuzberg PDF→MD conversion on test PDFs.

Run inside kreuzberg venv: benchmark/.venv-kreuzberg/
Outputs JSON results to benchmark/results/kreuzberg.json

Modes tested:
  - normal: default pdfium text extraction
  - force_ocr: PaddleOCR forced OCR
  - layout_fast: layout detection (YOLO DocLayNet) + table extraction
  - layout_accurate: layout detection (RT-DETR v2) + SLANeXT table model
  - layout_accurate_ocr: accurate layout + force OCR + table extraction
"""

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXAMPLE_DIR = PROJECT_ROOT / "example"
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR = SCRIPT_DIR / "output"

TEST_PDFS = [
    ("1225027862.PDF", "小米2025年报(MSungHK)"),
    ("1224680620.PDF", "小米2025半年报(HYQiHei)"),
    ("新和成_2024 年年度报告.pdf", "新和成2024年报(A股)"),
]


def garbled_ratio(text: str, sample_size: int = 2000) -> float:
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


def build_modes():
    from kreuzberg import (
        ExtractionConfig,
        HierarchyConfig,
        LayoutDetectionConfig,
        OcrConfig,
        PdfConfig,
    )

    ocr_paddle = OcrConfig(backend="paddle-ocr", language="chi_sim")

    modes = {
        "normal": ExtractionConfig(
            output_format="markdown",
            ocr=ocr_paddle,
        ),
        "force_ocr": ExtractionConfig(
            output_format="markdown",
            force_ocr=True,
            ocr=ocr_paddle,
        ),
        "layout_fast": ExtractionConfig(
            output_format="markdown",
            include_document_structure=True,
            layout=LayoutDetectionConfig(preset="fast"),
            pdf_options=PdfConfig(hierarchy=HierarchyConfig(enabled=True)),
            ocr=ocr_paddle,
        ),
        "layout_accurate": ExtractionConfig(
            output_format="markdown",
            include_document_structure=True,
            layout=LayoutDetectionConfig(preset="accurate", table_model="slanet_plus"),
            pdf_options=PdfConfig(hierarchy=HierarchyConfig(enabled=True)),
            ocr=ocr_paddle,
        ),
        "layout_accurate_ocr": ExtractionConfig(
            output_format="markdown",
            include_document_structure=True,
            force_ocr=True,
            layout=LayoutDetectionConfig(preset="accurate", table_model="slanet_plus"),
            pdf_options=PdfConfig(hierarchy=HierarchyConfig(enabled=True)),
            ocr=ocr_paddle,
        ),
    }
    return modes


def merge_tables_into_content(content: str, tables) -> str:
    """Append table markdown at the end — kreuzberg does not inline tables into content."""
    if not tables:
        return content

    parts = [content.rstrip()]
    parts.append("\n\n---\n\n## Extracted Tables\n")
    for i, t in enumerate(tables):
        md = getattr(t, "markdown", None)
        page = getattr(t, "page_number", "?")
        if md:
            parts.append(f"\n### Table {i + 1} (page {page})\n\n{md}\n")
    return "\n".join(parts)


def main():
    from kreuzberg import extract_file_sync

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = build_modes()
    results = []

    for mode_name, config in modes.items():
        print(f"\n{'#' * 60}")
        print(f"kreuzberg mode: {mode_name}")
        print(f"{'#' * 60}")

        for filename, label in TEST_PDFS:
            pdf_path = EXAMPLE_DIR / filename
            if not pdf_path.exists():
                print(f"SKIP: {filename} not found")
                continue

            print(f"\n{'=' * 60}")
            print(f"kreuzberg[{mode_name}]: {label} ({filename})")
            print(f"{'=' * 60}")

            t0 = time.time()
            result_obj = extract_file_sync(str(pdf_path), config=config)
            duration = time.time() - t0

            markdown = result_obj.content
            tables = result_obj.tables or []
            table_count = len(tables)

            content_garbled = garbled_ratio(markdown)

            table_text = " ".join(
                t.markdown for t in tables if getattr(t, "markdown", None)
            )
            table_garbled = garbled_ratio(table_text) if table_text else 0.0

            merged = merge_tables_into_content(markdown, tables)
            chars = len(merged)

            out_name = f"{Path(filename).stem}_kreuzberg_{mode_name}.md"
            (OUTPUT_DIR / out_name).write_text(merged, encoding="utf-8")

            result = {
                "tool": "kreuzberg",
                "mode": mode_name,
                "file": filename,
                "label": label,
                "chars": chars,
                "content_chars": len(markdown),
                "table_count": table_count,
                "duration_s": round(duration, 2),
                "content_garbled": round(content_garbled, 4),
                "table_garbled": round(table_garbled, 4),
                "quality_score": result_obj.quality_score,
                "preview": markdown[:500].replace("\n", "\\n"),
            }
            results.append(result)

            print(f"  content_chars={len(markdown)}  tables={table_count}")
            print(
                f"  time={duration:.1f}s  content_garbled={content_garbled:.4f}  table_garbled={table_garbled:.4f}"
            )

    out_path = RESULTS_DIR / "kreuzberg.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
