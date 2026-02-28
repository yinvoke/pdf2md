"""
Test script: convert all PDFs in example/ with convert_report.

Run from project root:
  uv run python test/run_report_example.py
  uv run python test/run_report_example.py -v
"""
import argparse
import os
import signal
import sys
from pathlib import Path

# Ensure project root is on path when running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.converter import convert_report, get_accelerator_device
except ImportError:
    convert_report = None
    get_accelerator_device = None

EXAMPLE_DIR = PROJECT_ROOT / "example"
OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all PDFs in example/ to Markdown with convert_report."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed conversion summary for each PDF.",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only convert files whose names contain this text.",
    )
    return parser.parse_args()


def _install_fast_interrupt() -> None:
    """
    Install a fast Ctrl+C handler for local dev tests.
    Docling pipeline shutdown may block on worker thread cleanup; force-exit
    avoids waiting tens of seconds after KeyboardInterrupt.
    """

    def _on_sigint(_signum, _frame) -> None:
        print("\nReceived Ctrl+C, force exiting now.")
        os._exit(130)

    signal.signal(signal.SIGINT, _on_sigint)


def main() -> None:
    args = parse_args()
    if convert_report is None:
        raise RuntimeError("convert_report is not available in app.converter. Please implement it first.")

    _install_fast_interrupt()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = list(EXAMPLE_DIR.glob("*.pdf"))
    if args.name_contains:
        pdfs = [p for p in pdfs if args.name_contains in p.name]
    if not pdfs:
        print(f"No PDF files found in {EXAMPLE_DIR}")
        return

    device = get_accelerator_device().upper() if get_accelerator_device else "?"
    print(f"Found {len(pdfs)} PDF(s) in {EXAMPLE_DIR}, mode=report, device={device}")
    for pdf_path in sorted(pdfs):
        out_name = pdf_path.stem + ".report.md"
        out_path = OUTPUT_DIR / out_name
        print(f"Converting: {pdf_path.name} -> {out_path.name}")
        try:
            markdown, summary = convert_report(
                pdf_path,
                return_summary=True,
                verbose=args.verbose,
            )
            out_path.write_text(markdown, encoding="utf-8")
            print(f"  Written {len(markdown)} chars, {summary.pages} pages, {summary.duration_sec:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    print(f"Done. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
