#!/usr/bin/env python3
"""
Aggregate per-scenario STEM records into report tables.

Reads the record format written by `evaluate_directory.py` and
`evaluate_predictions.py`. Point it at one system's results, or at
several concatenated together to compare systems -- records are grouped
by their `method` field.

    python scripts/aggregate_results.py --input outputs/my_model/results.jsonl
    cat outputs/*/results.jsonl > all.jsonl
    python scripts/aggregate_results.py --input all.jsonl --latex

Reported: overall PTM/EPA, bootstrap 95% CIs, per-pattern PTM, failure
type distribution, and -- when more than one system is present --
Kruskal-Wallis plus pairwise Mann-Whitney U with Cohen's d. Detector
TPR/TNR against S* is always reported, since it bounds how much of any
PTM gap can be attributed to generation rather than detection.

Uses the same estimators and seed as the paper's own table generation.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.failure_taxonomy import classify_failures, TYPE_NAMES  # noqa: E402
from stem.metrics import cohens_d  # noqa: E402
from span.pattern_library import PATTERN_NAMES  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B_BOOTSTRAP = 10000
SEED = 42


def load_records(path):
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        payload = json.load(f)
    return payload["records"] if isinstance(payload, dict) else payload


def group_by_method(records):
    """Group records by their `method` label, preserving first-seen order."""
    groups = OrderedDict()
    for r in records:
        groups.setdefault(r.get("method", "system"), []).append(r)
    return groups


def label(method):
    return method


def get_ptm(r):
    return r["ptm"] if "ptm" in r else r["match_prescribed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="per-scenario records (.jsonl or .json)")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table bodies")
    args = ap.parse_args()

    records = load_records(args.input)
    by_method = group_by_method(records)
    methods = list(by_method)

    ptm = {m: np.array([get_ptm(r) for r in by_method[m]]) for m in methods}
    epa = {m: np.array([r["epa"] for r in by_method[m]]) for m in methods}
    frob = {m: np.array([r["frob_dist"] for r in by_method[m]]) for m in methods}

    print(f"Input: {args.input}")
    print(f"Records: {len(records)}  Systems: {', '.join(label(m) for m in methods)}")

    # --- Table 1: overall -------------------------------------------
    print("\n" + "=" * 66)
    print("TABLE 1  Overall PTM / EPA")
    print("=" * 66)
    print(f"{'System':<14}{'N':>6}{'PTM':>10}{'EPA':>10}{'||E||_F':>10}")
    for m in methods:
        print(f"{label(m):<14}{len(ptm[m]):>6}{ptm[m].mean():>10.4f}"
              f"{epa[m].mean():>10.4f}{frob[m].mean():>10.4f}")

    ranking = sorted(methods, key=lambda m: -ptm[m].mean())
    print(f"\nRanking by PTM: {' > '.join(label(m) for m in ranking)}")

    # --- Bootstrap CIs ----------------------------------------------
    print("\n" + "=" * 66)
    print(f"TABLE 2  Bootstrap 95% CIs for PTM (B={B_BOOTSTRAP}, seed={SEED})")
    print("=" * 66)
    rng = np.random.default_rng(SEED)
    print(f"{'System':<14}{'PTM':>10}{'95% CI':>22}")
    for m in methods:
        vals = ptm[m]
        boot = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                         for _ in range(B_BOOTSTRAP)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{label(m):<14}{vals.mean():>10.4f}{f'[{lo:.4f}, {hi:.4f}]':>22}")

    # --- Per-pattern PTM --------------------------------------------
    print("\n" + "=" * 66)
    print("TABLE 3  Per-pattern PTM")
    print("=" * 66)
    patterns = [p for p in PATTERN_NAMES
                if any(r["pattern"] == p for r in records)]
    print(f"{'Pattern':<18}" + "".join(f"{label(m):>12}" for m in methods))
    for pat in patterns:
        cells = []
        for m in methods:
            arr = np.array([get_ptm(r) for r in by_method[m] if r["pattern"] == pat])
            cells.append(f"{arr.mean():>12.4f}" if len(arr) else f"{'--':>12}")
        print(f"{pat.replace('_', ' '):<18}" + "".join(cells))

    # --- Failure distribution ---------------------------------------
    print("\n" + "=" * 66)
    print("TABLE 4  Failure distribution (% of videos with >=1 of each type)")
    print("=" * 66)
    print(f"{'Failure type':<28}" + "".join(f"{label(m):>12}" for m in methods))
    for ft in ("I", "II", "III", "IV", "None"):
        cells = []
        for m in methods:
            entries = by_method[m]
            hit = sum(1 for r in entries
                      if ft in classify_failures(r["delta_S_ideal"], r["E_error"]))
            cells.append(f"{100.0 * hit / len(entries):>11.1f}%")
        print(f"{f'Type {ft} ({TYPE_NAMES[ft]})'[:27]:<28}" + "".join(cells))

    # --- Significance ------------------------------------------------
    if len(methods) > 1:
        print("\n" + "=" * 66)
        print("TABLE 5  Significance")
        print("=" * 66)
        h, p = stats.kruskal(*[ptm[m] for m in methods])
        print(f"Kruskal-Wallis across {len(methods)} systems: H={h:.2f}, p={p:.3e}")
        print(f"\n{'Pair':<22}{'U':>12}{'p':>14}{'Cohen d':>10}")
        for m1, m2 in combinations(methods, 2):
            u, pv = stats.mannwhitneyu(ptm[m1], ptm[m2], alternative="two-sided")
            d = cohens_d(ptm[m1], ptm[m2])
            sig = "*" if pv < 0.05 else " "
            print(f"{label(m1) + '--' + label(m2):<22}{u:>12.0f}{pv:>14.3e}{d:>10.4f} {sig}")

    # --- Detector quality --------------------------------------------
    print("\n" + "=" * 66)
    print("TABLE 7  Detector quality (S_obs vs S*)")
    print("=" * 66)
    print(f"{'System':<14}{'TP':>8}{'FN':>8}{'TN':>8}{'FP':>8}{'TPR':>9}{'TNR':>9}")
    tot = dict(tp=0, fn=0, tn=0, fp=0)
    for m in methods:
        c = dict(tp=0, fn=0, tn=0, fp=0)
        for r in by_method[m]:
            so, si = np.array(r["S_obs"]), np.array(r["S_ideal"])
            c["tp"] += int(np.sum((si == 1) & (so == 1)))
            c["fn"] += int(np.sum((si == 1) & (so == 0)))
            c["tn"] += int(np.sum((si == 0) & (so == 0)))
            c["fp"] += int(np.sum((si == 0) & (so == 1)))
        for k in tot:
            tot[k] += c[k]
        tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
        tnr = c["tn"] / max(c["tn"] + c["fp"], 1)
        print(f"{label(m):<14}{c['tp']:>8}{c['fn']:>8}{c['tn']:>8}{c['fp']:>8}"
              f"{tpr:>9.4f}{tnr:>9.4f}")
    tpr = tot["tp"] / max(tot["tp"] + tot["fn"], 1)
    tnr = tot["tn"] / max(tot["tn"] + tot["fp"], 1)
    n_cells = sum(tot.values())
    print(f"{'All':<14}{tot['tp']:>8}{tot['fn']:>8}{tot['tn']:>8}{tot['fp']:>8}"
          f"{tpr:>9.4f}{tnr:>9.4f}")
    print(f"\n{n_cells} presence cells evaluated.")
    print("TPR/TNR are the detector's own error rates. Read PTM and EPA as")
    print("comparative diagnostics across systems scored by the SAME detector,")
    print("not as absolute quality scores.")

    if args.latex:
        print("\n" + "=" * 66)
        print("LaTeX table bodies")
        print("=" * 66)
        print(r"% Table 1")
        for m in methods:
            print(f"{label(m)} & {len(ptm[m])} & {ptm[m].mean():.4f} & {epa[m].mean():.4f} \\\\")
        print(r"% Table 3 (per-pattern PTM)")
        for pat in patterns:
            cells = []
            for m in methods:
                arr = np.array([get_ptm(r) for r in by_method[m] if r["pattern"] == pat])
                cells.append(f"{arr.mean():.4f}" if len(arr) else "--")
            print(f"{pat.replace('_', ' ')} & {' & '.join(cells)} \\\\")


if __name__ == "__main__":
    main()
