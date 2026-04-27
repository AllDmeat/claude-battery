#!/usr/bin/env python3
"""
mine_framing.py — quick correlation between positive vs negative framing in
prompts (and CLAUDE.md) and "satisfactory end state" signals across Claude
Code session transcripts.

What it does
------------
Walks ~/.claude/projects/**/*.jsonl, filters to a model substring
(default: "opus-4-7"), and for each session computes:

  framing inputs
    - user_framing_score:  (pos - neg) / (pos + neg) over real user prompts
    - claude_md_framing_score: same lexicon over CLAUDE.md at the session cwd
      (only if --include-claude-md is set; uses the current on-disk version)

  outcome signals
    - one_shot:           1 user prompt total, no follow-up correction tokens
    - n_corrections:      "no/wait/actually/wrong/revert/..." in user msgs 2..N
    - n_commendations:    "thanks/perfect/lgtm/works/..." in user msgs 2..N
    - compound_memory_writes:
        Write/Edit/MultiEdit/Bash tool uses that touch ".context/"
        (the gitignored dir compound-engineering writes session memories to)
    - tool_error_rate:    fraction of tool_results flagged is_error
    - n_user_prompts, n_assistant_turns, n_tool_uses, duration_min

Outputs
-------
  - CSV: one row per session
  - stdout: means of outcome signals grouped by user_framing_score tertile

Caveat
------
Heuristic and correlational. Confounds: task difficulty, repo, prompt clarity.
Use this as a hypothesis sniff test, not as proof. Run a real A/B (paired
tasks, same model pin, framing-only flip) before drawing conclusions.

Usage
-----
    python3 mine_framing.py
    python3 mine_framing.py --projects ~/.claude/projects \
        --model-substring opus-4-7 --out framing.csv --include-claude-md

    # dump the per-session rows pretty:
    python3 mine_framing.py --pretty
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------- lexicon ----------------------------------------------------------

# Word-boundary patterns. Keep small and obvious; tune later if needed.
NEGATIVE_PATTERNS = [
    r"\bnever\b",
    r"\bdon't\b",
    r"\bdo not\b",
    r"\bdoesn't\b",
    r"\bdoes not\b",
    r"\bavoid\b",
    r"\bmust not\b",
    r"\bshouldn't\b",
    r"\bshould not\b",
    r"\bwon't\b",
    r"\bwill not\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\brefrain\b",
    r"\bskip\b",
    r"\bwithout\b",
    r"\bno (longer|more|need|reason|excuse)\b",
    r"\bdo not\b",
    r"\binstead of\b",
    r"\brather than\b",
]

POSITIVE_PATTERNS = [
    r"\balways\b",
    r"\bprefer\b",
    r"\bensure\b",
    r"\bmake sure\b",
    r"\bdefault to\b",
    r"\bfavor\b",
    r"\bfavour\b",
    r"\bdo (this|the following)\b",
    r"\bplease\b",
    r"\buse\b",
    r"\bwrite\b",
    r"\bkeep\b",
    r"\bfollow\b",
    r"\bmatch\b",
    r"\benable\b",
    r"\bif (you|the|this)\b",
    r"\bwhen (you|the|this)\b",
]

CORRECTION_PATTERNS = [
    r"^\s*no[.,!\s]",
    r"^\s*nope\b",
    r"\bwait\b",
    r"\bactually\b",
    r"\bthat'?s wrong\b",
    r"\bnot what i (wanted|asked)\b",
    r"\bundo\b",
    r"\brevert\b",
    r"\bbroke(n)?\b",
    r"\bfix that\b",
    r"\btry again\b",
    r"\bhold on\b",
    r"\bstop\b",
    r"\bdon'?t do\b",
    r"\bregress(ed|ion)?\b",
]

COMMENDATION_PATTERNS = [
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bperfect\b",
    r"\blgtm\b",
    r"\bship it\b",
    r"\bworks\b",
    r"\bnice\b",
    r"\bgreat\b",
    r"\bawesome\b",
    r"\blove it\b",
    r"\bexactly\b",
    r"\bbeautiful\b",
    r"\bthat'?s it\b",
]

NEG_RE = re.compile("|".join(NEGATIVE_PATTERNS), re.IGNORECASE)
POS_RE = re.compile("|".join(POSITIVE_PATTERNS), re.IGNORECASE)
CORR_RE = re.compile("|".join(CORRECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)
COMM_RE = re.compile("|".join(COMMENDATION_PATTERNS), re.IGNORECASE)


def score_framing(text: str) -> tuple[int, int, float]:
    """Return (pos, neg, score). score = (pos-neg)/(pos+neg), 0.0 if both 0."""
    if not text:
        return 0, 0, 0.0
    pos = len(POS_RE.findall(text))
    neg = len(NEG_RE.findall(text))
    total = pos + neg
    score = (pos - neg) / total if total else 0.0
    return pos, neg, score


# ---------- transcript parsing ----------------------------------------------


def extract_user_text(msg) -> str:
    """Return the user-authored text portion of a message, ignoring tool results."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            # tool_result blocks are not user prose
            if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                chunks.append(blk["text"])
        return "\n".join(chunks)
    return ""


def is_real_user_prompt(event: dict) -> bool:
    """True only if this is human-authored prose (not a tool_result wrapper)."""
    if event.get("type") != "user":
        return False
    msg = event.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip() != ""
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text" and (b.get("text") or "").strip()
            for b in content
        )
    return False


def parse_ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def iter_session_files(root: Path):
    for p in sorted(root.glob("*/*.jsonl")):
        yield p


def load_events(path: Path):
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            # skip queue-operation bookkeeping rows
            if ev.get("type") == "queue-operation":
                continue
            out.append(ev)
    return out


def first_model_id(events) -> str | None:
    for ev in events:
        if ev.get("type") == "assistant":
            m = (ev.get("message") or {}).get("model")
            if m:
                return m
    return None


def session_meta(events):
    cwd = git_branch = session_id = None
    start = end = None
    for ev in events:
        cwd = ev.get("cwd") or cwd
        git_branch = ev.get("gitBranch") or git_branch
        session_id = ev.get("sessionId") or session_id
        ts = parse_ts(ev.get("timestamp"))
        if ts:
            start = ts if start is None or ts < start else start
            end = ts if end is None or ts > end else end
    return cwd, git_branch, session_id, start, end


def count_compound_memory_writes(events) -> int:
    n = 0
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for blk in (ev.get("message") or {}).get("content", []) or []:
            if not isinstance(blk, dict) or blk.get("type") != "tool_use":
                continue
            name = blk.get("name") or ""
            inp = blk.get("input") or {}
            path = inp.get("file_path") or inp.get("path") or ""
            cmd = inp.get("command") or ""
            blob = f"{path}\n{cmd}"
            if "/.context/" in blob or blob.startswith(".context/") or "/.context " in blob:
                n += 1
            elif name == "Bash" and re.search(r"(^|[\s/'\"])\.context/", cmd):
                n += 1
    return n


def count_tool_uses_and_errors(events):
    tool_uses = 0
    tool_errors = 0
    for ev in events:
        if ev.get("type") == "assistant":
            for blk in (ev.get("message") or {}).get("content", []) or []:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    tool_uses += 1
        elif ev.get("type") == "user":
            content = (ev.get("message") or {}).get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        if blk.get("is_error"):
                            tool_errors += 1
    return tool_uses, tool_errors


def session_user_prompts(events):
    return [
        extract_user_text(ev.get("message") or {})
        for ev in events
        if is_real_user_prompt(ev)
    ]


def read_claude_md(cwd: str | None) -> str | None:
    if not cwd:
        return None
    p = Path(cwd) / "CLAUDE.md"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


# ---------- per-session aggregation -----------------------------------------


def analyze_session(path: Path, include_md: bool):
    events = load_events(path)
    if not events:
        return None
    model = first_model_id(events)
    cwd, branch, sid, start, end = session_meta(events)
    prompts = session_user_prompts(events)
    if not prompts:
        return None  # nothing to score

    pos_total = neg_total = 0
    per_prompt_scores = []
    for p in prompts:
        pos, neg, sc = score_framing(p)
        pos_total += pos
        neg_total += neg
        if pos + neg > 0:
            per_prompt_scores.append(sc)
    user_score = (
        (pos_total - neg_total) / (pos_total + neg_total)
        if (pos_total + neg_total)
        else 0.0
    )

    # outcome signals look at user msgs 2..N (the corrections/commendations
    # that arrive AFTER the assistant has acted).
    follow_ups = "\n\n".join(prompts[1:]) if len(prompts) > 1 else ""
    n_corrections = len(CORR_RE.findall(follow_ups))
    n_commendations = len(COMM_RE.findall(follow_ups))

    one_shot = len(prompts) == 1 and n_corrections == 0

    tool_uses, tool_errors = count_tool_uses_and_errors(events)
    compound_writes = count_compound_memory_writes(events)

    duration_min = (
        (end - start).total_seconds() / 60.0 if (start and end) else 0.0
    )

    md_text = read_claude_md(cwd) if include_md else None
    md_pos, md_neg, md_score = (0, 0, 0.0)
    if md_text:
        md_pos, md_neg, md_score = score_framing(md_text)

    return {
        "session_file": str(path),
        "session_id": sid or "",
        "model": model or "",
        "cwd": cwd or "",
        "git_branch": branch or "",
        "start": start.isoformat() if start else "",
        "end": end.isoformat() if end else "",
        "duration_min": round(duration_min, 2),
        "n_user_prompts": len(prompts),
        "n_tool_uses": tool_uses,
        "n_tool_errors": tool_errors,
        "tool_error_rate": round(tool_errors / tool_uses, 4) if tool_uses else 0.0,
        "user_pos": pos_total,
        "user_neg": neg_total,
        "user_framing_score": round(user_score, 4),
        "claude_md_pos": md_pos,
        "claude_md_neg": md_neg,
        "claude_md_framing_score": round(md_score, 4) if md_text else "",
        "one_shot": int(one_shot),
        "n_corrections": n_corrections,
        "n_commendations": n_commendations,
        "compound_memory_writes": compound_writes,
    }


# ---------- aggregation ------------------------------------------------------


def tertile_buckets(rows, key):
    """Split rows into low/mid/high tertiles by row[key]; ties broken by index."""
    if not rows:
        return {"low": [], "mid": [], "high": []}
    sorted_rows = sorted(rows, key=lambda r: (r[key], r["session_id"]))
    n = len(sorted_rows)
    a = n // 3
    b = 2 * n // 3
    return {
        "low": sorted_rows[:a],
        "mid": sorted_rows[a:b],
        "high": sorted_rows[b:],
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else 0.0


def summarize(rows, score_key, label):
    print()
    print(f"=== framing tertile by {label} ({score_key}) ===")
    if not rows:
        print("  (no sessions)")
        return
    buckets = tertile_buckets(rows, score_key)
    headers = [
        "bucket",
        "n",
        "mean_score",
        "one_shot_rate",
        "mean_corrections",
        "mean_commendations",
        "mean_compound_writes",
        "mean_tool_err",
        "mean_prompts",
    ]
    print("  " + "  ".join(f"{h:>20}" for h in headers))
    for name in ("low", "mid", "high"):
        b = buckets[name]
        if not b:
            print(f"  {name:>20}  (empty)")
            continue
        row = [
            name,
            len(b),
            f"{mean([r[score_key] for r in b]):+.3f}",
            f"{mean([r['one_shot'] for r in b]):.2f}",
            f"{mean([r['n_corrections'] for r in b]):.2f}",
            f"{mean([r['n_commendations'] for r in b]):.2f}",
            f"{mean([r['compound_memory_writes'] for r in b]):.2f}",
            f"{mean([r['tool_error_rate'] for r in b]):.3f}",
            f"{mean([r['n_user_prompts'] for r in b]):.2f}",
        ]
        print("  " + "  ".join(f"{str(c):>20}" for c in row))


# ---------- main -------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--model-substring", default="opus-4-7",
                    help="case-insensitive substring match on assistant model id; "
                         "use '' to include all models")
    ap.add_argument("--out", default="framing.csv", help="output CSV path")
    ap.add_argument("--include-claude-md", action="store_true",
                    help="also score CLAUDE.md at each session's cwd (current on-disk version)")
    ap.add_argument("--min-prompts", type=int, default=1,
                    help="skip sessions with fewer than N real user prompts")
    ap.add_argument("--pretty", action="store_true",
                    help="print per-session rows to stdout in addition to summary")
    args = ap.parse_args()

    root = Path(args.projects).expanduser()
    if not root.exists():
        print(f"projects dir not found: {root}", file=sys.stderr)
        sys.exit(2)

    rows = []
    skipped_model = 0
    skipped_empty = 0
    for f in iter_session_files(root):
        analysis = analyze_session(f, include_md=args.include_claude_md)
        if analysis is None:
            skipped_empty += 1
            continue
        if args.model_substring and args.model_substring.lower() not in (analysis["model"] or "").lower():
            skipped_model += 1
            continue
        if analysis["n_user_prompts"] < args.min_prompts:
            skipped_empty += 1
            continue
        rows.append(analysis)

    if not rows:
        print(
            f"No sessions matched. Scanned {root}, skipped {skipped_model} for model "
            f"filter (--model-substring={args.model_substring!r}), {skipped_empty} empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = Path(args.out)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"scanned: {root}")
    print(f"sessions matched: {len(rows)}  (skipped {skipped_model} on model, {skipped_empty} empty)")
    print(f"csv written: {out_path.resolve()}")

    summarize(rows, "user_framing_score", "user prompts")
    if args.include_claude_md:
        rows_with_md = [r for r in rows if r["claude_md_framing_score"] != ""]
        summarize(rows_with_md, "claude_md_framing_score", "CLAUDE.md")

    if args.pretty:
        print("\n=== per-session rows ===")
        for r in rows:
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
