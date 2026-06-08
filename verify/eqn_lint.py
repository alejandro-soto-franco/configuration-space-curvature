#!/usr/bin/env python3
"""
Equation-coverage linter (Layer A).

Extracts every displayed identity (\\label{eq: ...}) from the paper, cross-references
it against tools/equation_registry.toml, and gates on coverage:

  * every displayed equation must be registered (no unclassified identity);
  * no registry entry may be stale (a label that no longer exists in the .tex);
  * every `algebraic` equation/claim must bind a symbolic check that PASSES,
    where a pass requires the true identity to hold AND a counter-test (perturbed
    identity) to be rejected.

This closes the coverage drift that let the 2026-06-08 sign error through: a new
algebraic identity cannot enter the paper unchecked, and a vacuously-passing check
is caught by its counter-test.

Run:  uv run --with numpy --with sympy python tools/eqn_lint.py [paper_dir]
Exit code 0 iff fully covered and all checks pass.
"""

import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGISTRY = os.path.join(HERE, "equation_registry.toml")
DEFAULT_PAPER = os.path.join(ROOT, "paper")
SYMPY_DIR = HERE
SCRIPTS = {"cf2": "cf2_curvature_pressure_channel_verify.py",
           "cf3": "cf3_imbalance_production_verify.py"}

LABEL_RE = re.compile(r"\\label\{(eq: [^}]+)\}")
GREEN, RED, YEL, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def displayed_labels(paper_dir):
    labels = {}
    for base, _, files in os.walk(paper_dir):
        for fn in files:
            if not fn.endswith(".tex"):
                continue
            path = os.path.join(base, fn)
            with open(path, encoding="utf-8") as fh:
                for m in LABEL_RE.finditer(fh.read()):
                    labels.setdefault(m.group(1), os.path.relpath(path, paper_dir))
    return labels


_script_cache = {}


def run_script(key):
    if key in _script_cache:
        return _script_cache[key]
    proc = subprocess.run([sys.executable, os.path.join(SYMPY_DIR, SCRIPTS[key])],
                          capture_output=True, text=True)
    ok = proc.returncode == 0
    _script_cache[key] = (ok, "exit 0" if ok else f"exit {proc.returncode}")
    return _script_cache[key]


def run_check(spec):
    """spec is 'native:<name>' or 'cf2'/'cf3'. Returns (ok, detail)."""
    if spec in SCRIPTS:
        return run_script(spec)
    if spec.startswith("native:"):
        sys.path.insert(0, SYMPY_DIR)
        import equation_checks
        r = equation_checks.run(spec.split(":", 1)[1])
        return r["ok"], f"pos {r['pos']:.1e} / counter {r['counter']:.1e}"
    return False, f"unknown check backend '{spec}'"


def main():
    paper_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAPER
    with open(REGISTRY, "rb") as fh:
        reg = tomllib.load(fh)
    equations = reg.get("equations", {})
    claims = reg.get("claims", {})

    found = displayed_labels(paper_dir)
    failures, table = [], []

    # 1. coverage: every displayed label must be registered
    for label in sorted(found):
        if label not in equations:
            failures.append(f"unregistered displayed equation: {label}  ({found[label]})")

    # 2. stale registry entries
    for label in sorted(equations):
        if label not in found:
            failures.append(f"stale registry entry (no such label in tex): {label}")

    # 3. run checks for algebraic equations and claims
    checked = {}  # spec -> (ok, detail) memo so each check runs once
    for label, meta in sorted(equations.items()):
        if label not in found:
            continue
        kind = meta.get("kind", "?")
        if kind == "algebraic":
            spec = meta.get("check")
            if not spec:
                failures.append(f"algebraic equation without a check: {label}")
                table.append((label, kind, "MISSING", False))
                continue
            if spec not in checked:
                checked[spec] = run_check(spec)
            ok, detail = checked[spec]
            table.append((label, kind, f"{spec}  [{detail}]", ok))
            if not ok:
                failures.append(f"check failed for {label}: {spec} ({detail})")
        else:
            table.append((label, kind, "", True))

    for name, meta in sorted(claims.items()):
        spec = meta.get("check", "")
        if spec not in checked:
            checked[spec] = run_check(spec)
        ok, detail = checked[spec]
        table.append((f"[claim] {name}", meta.get("kind", "?"), f"{spec}  [{detail}]", ok))
        if not ok:
            failures.append(f"check failed for claim {name}: {spec} ({detail})")

    # report
    print(f"equation coverage: {len(found)} displayed identities in "
          f"{os.path.relpath(paper_dir, ROOT)}\n")
    print(f"  {'label':<30}{'kind':<12}{'check'}")
    for label, kind, info, ok in table:
        mark = f"{GREEN}ok{OFF}" if ok else f"{RED}FAIL{OFF}"
        info_s = f"{info}" if info else f"{DIM}(no check needed){OFF}"
        flag = "" if ok else f"  {mark}"
        print(f"  {label:<30}{kind:<12}{info_s}{flag}")

    print()
    if failures:
        print(f"{RED}FAIL{OFF}  {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    n_alg = sum(1 for _, k, _, _ in table if k == "algebraic")
    print(f"{GREEN}PASS{OFF}  all {len(found)} equations classified; "
          f"{n_alg} algebraic identities verified (with counter-tests).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
