"""
Post-processing for Docling-generated Markdown.

Fixes known Docling conversion artifacts:
- Broken headings: a single heading split across consecutive ``## `` lines
- Image placeholders: ``<!-- image -->`` comments left by Docling
"""

from __future__ import annotations

import re
import logging

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fix_md: merge broken headings
# ---------------------------------------------------------------------------
#
# Docling sometimes splits one heading into two consecutive ``## `` lines
# separated by a single blank line.  Three merge strategies cover the
# observed patterns:
#
# Strategy A – **forward merge** (h2 continues h1):
#   ## 1              ## （ 1
#   (blank)    →      (blank)
#   ## 、企业文化       ## ）租赁的分类
#   Result:  ## 1、企业文化   /  ## （1）租赁的分类
#
# Strategy B – **reverse merge** (h1 continues h2):
#   ## 、持续经营       ## （ ）权益法
#   (blank)    →      (blank)
#   ## 2              ## 2
#   Result:  ## 2、持续经营   /  ## （2）权益法
#
# Strategy C – **guard** (must NOT merge):
#   ## 2 ）            ← closing fragment of previous heading
#   (blank)
#   ## 1 ）一般处理方法  ← independent next item
# ---------------------------------------------------------------------------


def fix_md(content: str) -> tuple[str, int]:
    """
    Fix broken ``## `` headings produced by Docling.

    Returns ``(fixed_content, fix_count)``.
    """
    lines = content.split("\n")
    result: list[str] = []
    fix_count = 0
    i = 0

    while i < len(lines):
        # Detect the three-line pattern: ## … / blank / ## …
        if (
            i + 2 < len(lines)
            and lines[i].startswith("## ")
            and lines[i + 1].strip() == ""
            and lines[i + 2].startswith("## ")
        ):
            h1_text = lines[i][3:].strip()
            h2_text = lines[i + 2][3:].strip()

            merged = _try_merge(h1_text, h2_text)
            if merged is not None:
                result.append(f"## {merged}")
                fix_count += 1
                i += 3  # skip h1 + blank + h2
                continue

        result.append(lines[i])
        i += 1

    if fix_count:
        _log.info("fix_md: merged %d broken heading(s)", fix_count)
    return "\n".join(result), fix_count


# ---- merge helpers --------------------------------------------------------

# Pre-compiled patterns used by the merge logic.
_RE_PUNCT_START = re.compile(r"^[、，。．\.\）\)]")
_RE_PURE_NUM = re.compile(r"^\d{1,3}$")
_RE_NUM_DOT = re.compile(r"^\d{1,3}[\.．]$")
_RE_NUM_CLOSE = re.compile(r"^\d{1,3}\s*[\)）]$")
_RE_OPEN_PAREN_NUM = re.compile(r"^[\(（]\s*\d{0,2}$")
_RE_CN_NUM_PAREN = re.compile(r"^[\(（]\s*[一二三四五六七八九十]+\s*[\)）]$")
# Reverse-merge: h1 starts with connective punctuation, h2 is pure number.
_RE_REVERSE_PUNCT = re.compile(r"^[、，．\.]")
# Reverse-merge: h1 is ``（ ）content`` with a space placeholder for the number.
_RE_EMPTY_PAREN = re.compile(r"^[\(（]\s*[\)）](.+)")
# Guard: h2 also looks like ``N ）content`` – parallel items, not a split.
_RE_NUM_CLOSE_CONTENT = re.compile(r"^\d{1,3}\s*[\)）].")
# Guard: h1 is a section-level heading (CJK number + 、) — never merge with
# a subsection fragment that happens to start with 、.
_CN_NUM = r"[一二三四五六七八九十百零]+"
_RE_SECTION_HEADING = re.compile(rf"^(第{_CN_NUM}[节章]|{_CN_NUM}\s*、)")


def _try_merge(h1: str, h2: str) -> str | None:
    """Return the merged heading text, or *None* if the pair should not merge."""
    h1_clean = h1.replace(" ", "")
    h2_clean = h2.replace(" ", "")

    # ------------------------------------------------------------------
    # Guard: reject merges that would be wrong
    # ------------------------------------------------------------------
    # h1 is a closing fragment like "2 ）" and h2 starts with "N ）…"
    # → they are parallel numbered items, not a single broken heading.
    if _RE_NUM_CLOSE.match(h1_clean) and _RE_NUM_CLOSE_CONTENT.match(h2_clean):
        return None

    # Both are lone open-parentheses → OCR noise, not a heading split.
    if h1_clean in ("（", "(") and h2_clean in ("（", "("):
        return None

    # ------------------------------------------------------------------
    # Strategy A: forward merge  (h1 is fragment, h2 is continuation)
    # ------------------------------------------------------------------
    if _RE_PUNCT_START.match(h2_clean):
        if _RE_SECTION_HEADING.match(h1_clean):
            return None
        return f"{h1}{h2}"

    # h1 is a pure number  ("1", "14")
    if _RE_PURE_NUM.match(h1_clean):
        return f"{h1}{h2}"

    # h1 is "N." / "N．"
    if _RE_NUM_DOT.match(h1_clean):
        return f"{h1}{h2}"

    # h1 is "N）" / "N)"  (already guarded above against parallel items)
    if _RE_NUM_CLOSE.match(h1_clean):
        return f"{h1}{h2}"

    # h1 is a lone open-parenthesis  "（" / "("
    if h1_clean in ("（", "("):
        return f"{h1}{h2}"

    # h1 is open-paren + optional digits  "（1" / "（"
    if _RE_OPEN_PAREN_NUM.match(h1_clean):
        return f"{h1}{h2}"

    # h1 is CJK-number in parentheses  "（一）" / "(二)"
    if _RE_CN_NUM_PAREN.match(h1_clean):
        return f"{h1}{h2}"

    # ------------------------------------------------------------------
    # Strategy B: reverse merge  (h1 has content, h2 has the number)
    # ------------------------------------------------------------------
    # h1 starts with connective punctuation, h2 is a pure number
    # e.g. h1="、持续经营"  h2="2"  →  "2、持续经营"
    if _RE_REVERSE_PUNCT.match(h1_clean) and _RE_PURE_NUM.match(h2_clean):
        return f"{h2}{h1}"

    # h1 is "（ ）content" with empty placeholder, h2 is the missing number
    # e.g. h1="（ ）权益法"  h2="2"  →  "（2）权益法"
    m = _RE_EMPTY_PAREN.match(h1_clean)
    if m and _RE_PURE_NUM.match(h2_clean):
        rest = m.group(1)
        # Determine which paren style was used in the original text.
        paren_open = "（" if h1.lstrip().startswith("（") else "("
        paren_close = "）" if "）" in h1 else ")"
        return f"{paren_open}{h2}{paren_close}{rest}"

    return None


# ---------------------------------------------------------------------------
# remove_image_placeholders
# ---------------------------------------------------------------------------

_RE_IMAGE_COMMENT = re.compile(r"^\s*<!--\s*image\s*-->\s*$", re.MULTILINE)


def remove_image_placeholders(content: str) -> str:
    """Strip ``<!-- image -->`` placeholder lines inserted by Docling."""
    cleaned = _RE_IMAGE_COMMENT.sub("", content)
    # Collapse runs of 3+ blank lines left behind into 2.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def postprocess_markdown(content: str) -> str:
    """
    Apply all Markdown post-processing steps.

    Currently:
    1. ``fix_md``  – merge broken headings
    2. ``remove_image_placeholders`` – strip ``<!-- image -->``
    """
    content, _ = fix_md(content)
    content = remove_image_placeholders(content)
    return content
