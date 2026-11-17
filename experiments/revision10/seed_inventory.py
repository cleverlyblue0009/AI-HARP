"""Revision 10, step 0: inventory of every seed number used anywhere in the repository, and the choice of the
replication seeds R-A / R-B (results/revision10/READING_NOTES.md; seeds.json).

    python -m experiments.revision10.seed_inventory            -> results/revision10/seed_inventory.json, seeds.json

Sources, all read-only:
  1. data: every tracked results file -- CSV columns whose name contains "seed", and JSON / JSONL values under keys whose
     name contains "seed" (ints, lists of ints, "a-b" ranges); grouped by results/<revision>/ directory;
  2. paths: Veins bundle / shard names "<cell>_d<density>_s<seed>" in the tracked tree AND on branches r8-ci and r9-ci;
  3. code: every line of tracked .py / .yaml / .yml / .sh / .ps1 / ci/*.json files that mentions "seed", with the
     integers on it (range(a, b) and "a-b" expanded); kept as file:line evidence. Code integers are NOISY (counts,
     offsets, salts), so they are listed, not merged into the used set, except that any integer >= 230 on such a line
     is a candidate conflict and is reported for manual review.
The replication seeds are the first 20 integers >= 230 that occur in neither the data set nor the path set nor the
code-candidate set (any integer >= 230 on a seed line of code). Skipped integers are recorded with the reason.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import defaultdict

from common.config import PROJECT_ROOT
from experiments.revision1.provenance import stamp, write_json

OUT = PROJECT_ROOT / "results" / "revision10"
START, N = 230, 20
CODE_EXT = (".py", ".yaml", ".yml", ".sh", ".ps1")
RANGE = re.compile(r"range\(\s*(\d+)\s*,\s*(\d+)\s*\)")
DASH = re.compile(r"(?<![\d.])(\d{1,4})\s*-\s*(\d{1,4})(?![\d.])")
INT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
PATHSEED = re.compile(r"_d\d+(?:\.\d+)?_s(\d+)(?=[_/.]|$)")
csv.field_size_limit(1 << 30)


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout


def ints_of(v) -> set[int]:
    out = set()
    if isinstance(v, bool):
        return out
    if isinstance(v, int):
        out.add(v)
    elif isinstance(v, float) and v.is_integer():
        out.add(int(v))
    elif isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", v)
        if m and int(m[1]) <= int(m[2]) <= int(m[1]) + 5000:
            out |= set(range(int(m[1]), int(m[2]) + 1))
        elif v.strip().isdigit():
            out.add(int(v))
    elif isinstance(v, (list, tuple)):
        for x in v:
            out |= ints_of(x)
    return out


def walk(o, acc: set[int]) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            if "seed" in str(k).lower() and "salt" not in str(k).lower():
                acc |= ints_of(v)
            walk(v, acc)
    elif isinstance(o, list):
        for x in o:
            walk(x, acc)


def group(path: str) -> str:
    p = path.split("/")
    return "/".join(p[:2]) if len(p) > 2 else path


def data_seeds(files: list[str]) -> dict[str, set[int]]:
    by = defaultdict(set)
    for f in files:
        if not f.startswith("results/"):
            continue
        p = PROJECT_ROOT / f
        try:
            if f.endswith(".csv"):
                with open(p, newline="", encoding="utf-8") as fh:
                    r = csv.DictReader(fh)
                    cols = [c for c in (r.fieldnames or []) if "seed" in c.lower()]
                    if cols:
                        for row in r:
                            for c in cols:
                                by[group(f)] |= ints_of(row[c])
            elif f.endswith(".jsonl"):
                for line in open(p, encoding="utf-8"):
                    if "seed" in line:
                        walk(json.loads(line), by[group(f)])
            elif f.endswith(".json"):
                txt = p.read_text(encoding="utf-8")
                if "seed" in txt:
                    walk(json.loads(txt), by[group(f)])
        except (UnicodeDecodeError, json.JSONDecodeError, OSError):
            continue
    return by


def path_seeds(listing: list[str]) -> set[int]:
    out = set()
    for f in listing:
        out |= {int(m) for m in PATHSEED.findall(f)}
    return out


def code_lines(files: list[str]) -> list[dict]:
    out = []
    for f in files:
        if not (f.endswith(CODE_EXT) or (f.startswith("ci/") and f.endswith(".json"))):
            continue
        try:
            text = (PROJECT_ROOT / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "seed" not in line.lower():
                continue
            nums = set()
            for a, b in RANGE.findall(line):
                if int(b) - int(a) <= 5000:
                    nums |= set(range(int(a), int(b)))
            for a, b in DASH.findall(line):
                if int(a) <= int(b) <= int(a) + 5000:
                    nums |= set(range(int(a), int(b) + 1))
            rest = RANGE.sub(" ", line)                  # range(a, b): b is exclusive, already expanded above
            nums |= {int(x) for x in INT.findall(rest)}
            if nums:
                out.append({"file": f, "line": i, "text": line.strip()[:200], "ints": sorted(nums)})
    return out


def compact(s) -> str:
    s = sorted(s)
    if not s:
        return ""
    parts, a, b = [], s[0], s[0]
    for x in s[1:]:
        if x == b + 1:
            b = x
        else:
            parts.append(f"{a}" if a == b else f"{a}-{b}")
            a = b = x
    parts.append(f"{a}" if a == b else f"{a}-{b}")
    return ", ".join(parts)


def main(argv=None) -> int:
    files = git("ls-files").splitlines()
    data = data_seeds(files)
    used_data = set().union(*data.values()) if data else set()
    paths = {"tracked": path_seeds(files)}
    for br in ("r8-ci", "r9-ci"):
        git("fetch", "-q", "origin", br)
        paths[br] = path_seeds(git("ls-tree", "-r", "--name-only", f"origin/{br}").splitlines())
    used_paths = set().union(*paths.values())
    code = code_lines(files)
    code_candidates = defaultdict(list)
    for c in code:
        for x in c["ints"]:
            if x >= START:
                code_candidates[x].append(f"{c['file']}:{c['line']}")
    chosen, skipped, x = [], {}, START
    while len(chosen) < N:
        why = []
        if x in used_data:
            why.append("data: " + ", ".join(g for g, s in sorted(data.items()) if x in s))
        if x in used_paths:
            why.append("paths: " + ", ".join(k for k, s in paths.items() if x in s))
        if x in code_candidates:
            why.append("code: " + ", ".join(code_candidates[x][:8]))
        if why:
            skipped[x] = why
        else:
            chosen.append(x)
        x += 1
    inv = {"provenance": stamp("revision10/seed_inventory", {"start": START, "n": N}),
           "data_seeds_by_results_dir": {g: compact(s) for g, s in sorted(data.items())},
           "data_seeds_all": compact(used_data),
           "path_seeds": {k: compact(s) for k, s in paths.items()},
           "code_seed_lines": code,
           "code_integers_ge_230": {str(k): v for k, v in sorted(code_candidates.items())},
           "max_data_seed": max(used_data) if used_data else None}
    write_json(OUT / "seed_inventory.json", inv)
    seeds = {"provenance": stamp("revision10/seeds", {"start": START, "n": N}),
             "rule": "first 20 integers >= 230 that appear nowhere in results/revision10/seed_inventory.json "
                     "(data seeds, Veins path seeds on revision-10/r8-ci/r9-ci, integers >= 230 on seed lines of code)",
             "R_A": chosen[:10], "R_B": chosen[10:], "skipped": {str(k): v for k, v in skipped.items()}}
    write_json(OUT / "seeds.json", seeds)
    print("data seeds:", compact(used_data))
    print("path seeds:", {k: compact(s) for k, s in paths.items()})
    print("code ints >= 230:", {k: v[:3] for k, v in sorted(code_candidates.items())})
    print("R-A", chosen[:10], "R-B", chosen[10:], "skipped", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
