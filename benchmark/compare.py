#!/usr/bin/env python3
"""Compare benchmark results from all tools.

Reads JSON files from benchmark/results/ and prints comparison tables.
No external dependencies required.
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

W_TOOL = 22
W_FILE = 28
W_CHARS = 10
W_TIME = 10
W_GARBLED = 10


def load_results():
    """Load all result JSON files."""
    all_results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            all_results.extend(data)
        except Exception as e:
            print(f"Warning: failed to load {f.name}: {e}", file=sys.stderr)
    return all_results


def print_separator(char="-"):
    total = W_TOOL + W_FILE + W_CHARS + W_TIME + W_GARBLED + 4 * 3
    print(char * total)


def print_row(tool, file, chars, time, garbled):
    print(
        f"{tool:<{W_TOOL}} | {file:<{W_FILE}} | {chars:>{W_CHARS}} | "
        f"{time:>{W_TIME}} | {garbled:>{W_GARBLED}}"
    )


def main():
    results = load_results()
    if not results:
        print("No results found in benchmark/results/")
        print("Run benchmark scripts first:")
        print("  python benchmark/run_docling.py")
        print("  benchmark/.venv-marker/bin/python benchmark/run_marker.py")
        print("  benchmark/.venv-kreuzberg/bin/python benchmark/run_kreuzberg.py")
        return

    files = {}
    for r in results:
        key = r["file"]
        if key not in files:
            files[key] = {"label": r["label"], "results": []}
        files[key]["results"].append(r)

    print("\n" + "=" * 100)
    print("PDF→Markdown Conversion Benchmark")
    print("=" * 100)

    print(f"\n{'SUMMARY':^100}")
    print_separator("=")
    print_row("Tool [mode]", "File", "Chars", "Time(s)", "Garbled%")
    print_separator("-")

    for file_key, file_data in files.items():
        for r in sorted(file_data["results"], key=lambda x: (x["tool"], x["mode"])):
            tool_mode = f"{r['tool']}[{r['mode']}]"
            garbled_pct = f"{r['garbled_ratio'] * 100:.1f}%"
            print_row(
                tool_mode,
                r["label"],
                str(r["chars"]),
                f"{r['duration_s']:.1f}",
                garbled_pct,
            )
        print_separator("·")

    print()

    for file_key, file_data in files.items():
        print(f"\n{'─' * 80}")
        print(f"📄 {file_data['label']} ({file_key})")
        print(f"{'─' * 80}")

        file_results = sorted(
            file_data["results"], key=lambda x: (x["garbled_ratio"], x["duration_s"])
        )

        best_quality = file_results[0]
        fastest = min(file_results, key=lambda x: x["duration_s"])

        print(
            f"  🏆 Best quality: {best_quality['tool']}[{best_quality['mode']}] "
            f"(garbled={best_quality['garbled_ratio'] * 100:.1f}%)"
        )
        print(
            f"  ⚡ Fastest:      {fastest['tool']}[{fastest['mode']}] "
            f"({fastest['duration_s']:.1f}s)"
        )

        print(f"\n  {'Tool [mode]':<22} {'Chars':>8} {'Time':>8} {'Garbled':>8}")
        print(f"  {'─' * 50}")
        for r in file_results:
            marker = ""
            if r["garbled_ratio"] > 0.08:
                marker = " ❌"
            elif r["garbled_ratio"] > 0.01:
                marker = " ⚠️"
            else:
                marker = " ✅"
            print(
                f"  {r['tool']}[{r['mode']}]"
                f"{'':>{22 - len(r['tool']) - len(r['mode']) - 2}}"
                f" {r['chars']:>8} {r['duration_s']:>7.1f}s "
                f"{r['garbled_ratio'] * 100:>6.1f}%{marker}"
            )

    print(f"\n\n{'=' * 100}")
    print("TEXT PREVIEWS (first 200 chars)")
    print("=" * 100)

    for r in results:
        print(f"\n--- {r['tool']}[{r['mode']}] → {r['label']} ---")
        preview = r.get("preview", "")[:200]
        print(preview.replace("\\n", "\n"))

    print()


if __name__ == "__main__":
    main()
