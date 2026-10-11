#!/usr/bin/env python3
"""
compare_designite.py - compare two Designite output folders (before vs. after).

For every smell CSV found in both folders (ArchitectureSmells.csv, DesignSmells.csv,
ImplementationSmells.csv, TestabilitySmells.csv, TestSmells.csv) it reports:
  - total smell count before / after
  - count per smell type before / after (and the change)
  - which individual smell instances DISAPPEARED and which are NEW

Usage:
    python compare_designite.py BEFORE_DIR AFTER_DIR
    python compare_designite.py BEFORE_DIR AFTER_DIR --focus org.apache.tika.parser.csv
    python compare_designite.py BEFORE_DIR AFTER_DIR --out report.md

--focus  keeps only rows whose Package/Class text contains the given string
         (use it to look at the package you refactored).
--out    also writes the report to a Markdown file.

Notes:
  * A smell instance is identified by (Package, Class, Method, Smell). If you split
    or rename a class, its old smells show as "disappeared" and the new classes'
    smells show as "new". That is expected - read those two lists together.
  * Designite refuses to write into a non-empty folder, so use a fresh output
    folder for the "after" run and keep the "before" folder untouched.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SMELL_FILES = [
    "ArchitectureSmells.csv",
    "DesignSmells.csv",
    "ImplementationSmells.csv",
    "TestabilitySmells.csv",
    "TestSmells.csv",
]
KEY_COLUMNS = ("Package", "Class", "Method", "Smell")

# Designite versions name the columns differently; map them all to one set.
ALIASES = {
    "Package": ("package", "package name"),
    "Class": ("class", "type name", "type"),
    "Method": ("method", "method name"),
    "Smell": ("smell", "code smell", "design smell"),
    "Description": ("description", "cause of the smell", "cause"),
}


def normalise(row):
    """Return a copy of the row keyed by the standard names."""
    lower = {(k or "").strip().lower(): v for k, v in row.items()}
    out = dict(row)
    for std, names in ALIASES.items():
        for n in names:
            if n in lower and lower[n] is not None:
                out[std] = lower[n]
                break
    return out


def read_rows(path):
    """Return (header, rows) for a CSV, tolerating a BOM. Missing file -> (None, [])."""
    if not path.exists():
        return None, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_header = reader.fieldnames or []
        rows = [normalise(r) for r in reader]
        header = list(raw_header)
        for std in ALIASES:
            if rows and std in rows[0] and std not in header:
                header.append(std)
            elif not rows and any(n in [h.strip().lower() for h in raw_header]
                                  for n in ALIASES[std]) and std not in header:
                header.append(std)
        return header, rows


def make_key(row, cols):
    return tuple((row.get(c) or "").strip() for c in cols)


def fmt_key(key, cols):
    return " | ".join(f"{c}={v}" for c, v in zip(cols, key) if v)


def compare_file(name, before_dir, after_dir, focus, lines):
    b_header, b_rows = read_rows(before_dir / name)
    a_header, a_rows = read_rows(after_dir / name)
    if b_header is None and a_header is None:
        return None
    lines.append(f"\n## {name}")
    if b_header is None or a_header is None:
        side = "BEFORE" if b_header is None else "AFTER"
        lines.append(f"- File missing in the {side} folder - skipped.")
        return None

    if focus:
        def keep(r):
            return focus in (r.get("Package", "") + "." + r.get("Class", ""))
        b_rows = [r for r in b_rows if keep(r)]
        a_rows = [r for r in a_rows if keep(r)]

    cols = [c for c in KEY_COLUMNS if c in b_header and c in a_header]
    if "Smell" not in cols:
        lines.append("- No 'Smell' column - skipped.")
        return None

    if not b_rows:
        lines.append("- WARNING: the BEFORE file has no rows"
                     + (" for this focus filter." if focus else
                        " (header only). The before-measurement may be unusable."))
    if not a_rows and b_rows:
        lines.append("- WARNING: the AFTER file has no rows. Check that the "
                     "after-run really produced data before trusting this.")

    b_count = Counter(make_key(r, cols) for r in b_rows)
    a_count = Counter(make_key(r, cols) for r in a_rows)

    lines.append(f"- Total smells: **{len(b_rows)} -> {len(a_rows)}** "
                 f"({len(a_rows) - len(b_rows):+d})")

    # per smell type
    b_types = Counter((r.get("Smell") or "").strip() for r in b_rows)
    a_types = Counter((r.get("Smell") or "").strip() for r in a_rows)
    all_types = sorted(set(b_types) | set(a_types),
                       key=lambda t: -(abs(a_types[t] - b_types[t])))
    lines.append("\n| Smell type | Before | After | Change |")
    lines.append("|---|---:|---:|---:|")
    for t in all_types:
        lines.append(f"| {t} | {b_types[t]} | {a_types[t]} | {a_types[t] - b_types[t]:+d} |")

    # instance-level differences (multiset difference)
    gone = b_count - a_count
    new = a_count - b_count
    still = sum((b_count & a_count).values())
    lines.append(f"\n- Unchanged instances: {still}, disappeared: {sum(gone.values())}, "
                 f"new: {sum(new.values())}")

    def dump(title, counter, limit=40):
        if not counter:
            return
        lines.append(f"\n**{title}** (showing up to {limit})")
        for key, n in sorted(counter.items())[:limit]:
            suffix = f" x{n}" if n > 1 else ""
            lines.append(f"- {fmt_key(key, cols)}{suffix}")
        extra = len(counter) - limit
        if extra > 0:
            lines.append(f"- ... and {extra} more")

    dump("Disappeared", gone)
    dump("New", new)

    # architecture smells keep their numbers in the description (e.g. LCC)
    if name == "ArchitectureSmells.csv" and "Description" in b_header:
        b_desc = {make_key(r, cols): (r.get("Description") or "") for r in b_rows}
        a_desc = {make_key(r, cols): (r.get("Description") or "") for r in a_rows}
        changed = [k for k in b_desc if k in a_desc and b_desc[k] != a_desc[k]]
        if changed:
            lines.append("\n**Same smell, but the description changed** "
                         "(e.g. a different LCC value - the smell is still there):")
            for k in changed[:20]:
                lines.append(f"- {fmt_key(k, cols)}")
    return len(b_rows), len(a_rows)


def main():
    p = argparse.ArgumentParser(description="Compare two Designite output folders.")
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("--focus", default="", help="only rows whose package/class contains this text")
    p.add_argument("--out", type=Path, help="also write the report to this Markdown file")
    args = p.parse_args()

    for d in (args.before, args.after):
        if not d.is_dir():
            sys.exit(f"Folder not found: {d}")

    lines = ["# Designite before/after comparison",
             f"- Before: `{args.before}`", f"- After: `{args.after}`"]
    if args.focus:
        lines.append(f"- Focus filter: `{args.focus}`")

    grand_b = grand_a = 0
    seen = False
    for name in SMELL_FILES:
        result = compare_file(name, args.before, args.after, args.focus, lines)
        if result:
            seen = True
            grand_b += result[0]
            grand_a += result[1]
    if not seen:
        sys.exit("No comparable smell CSVs found in both folders.")

    lines.append(f"\n## Overall\n- All smell files together: **{grand_b} -> {grand_a}** "
                 f"({grand_a - grand_b:+d})")
    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\n[report written to {args.out}]")


if __name__ == "__main__":
    main()
