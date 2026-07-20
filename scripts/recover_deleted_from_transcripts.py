#!/usr/bin/env python3
"""Recover files that were created via the Write tool in agent transcripts.

I mistakenly deleted several experiment scripts that subagents had written.  git
never tracked them, so they can only be recovered from the JSONL agent
transcripts, where each Write tool call carries the file's full content.

This scans the given transcript .output files for Write tool-uses whose
``file_path`` matches a wanted name and restores the LAST version of each to
disk (last write wins).  Idempotent; refuses to overwrite an existing file
unless --force.

Usage:
    python3 scripts/recover_deleted_from_transcripts.py \\
        --dir /path/to/tasks --name z3_backtrack_realize.py --name ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def iter_write_calls(path: str):
    """Yield (file_path, content) for every Write tool-use in a JSONL file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or '"Write"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # walk the record for any tool_use with name == "Write"
            for tu in _find_tool_uses(rec, "Write"):
                inp = tu.get("input", {})
                fp = inp.get("file_path")
                content = inp.get("content")
                if fp and content is not None:
                    yield fp, content


def _find_tool_uses(obj, name):
    found = []
    if isinstance(obj, dict):
        if obj.get("type") == "tool_use" and obj.get("name") == name:
            found.append(obj)
        for v in obj.values():
            found.extend(_find_tool_uses(v, name))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_tool_uses(v, name))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of *.output transcripts")
    ap.add_argument("--name", action="append", default=[],
                    help="basename of a file to recover (repeatable)")
    ap.add_argument("--out", default=".", help="repo root to restore into")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    wanted = set(args.name)
    latest: dict[str, str] = {}  # basename -> content (last write wins)
    for tpath in sorted(glob.glob(os.path.join(args.dir, "*.output"))):
        for fp, content in iter_write_calls(tpath):
            base = os.path.basename(fp)
            if base in wanted:
                latest[base] = content

    if not latest:
        print("no matching Write calls found for:", ", ".join(sorted(wanted)))
        return 1

    for base, content in latest.items():
        # restore under scripts/ (or repo root for scratch_*)
        dest = os.path.join(args.out, "scripts", base)
        if base.startswith("scratch"):
            dest = os.path.join(args.out, base)
        if os.path.exists(dest) and not args.force:
            print(f"skip (exists): {dest}")
            continue
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"restored: {dest}  ({len(content)} bytes)")

    missing = wanted - set(latest)
    if missing:
        print("NOT recovered:", ", ".join(sorted(missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
