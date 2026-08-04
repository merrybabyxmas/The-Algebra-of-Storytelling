#!/usr/bin/env python3
"""
Validate a MEMS-1K benchmark file.

Checks structure, pattern counts, and -- most importantly -- the internal
consistency that the metrics rely on:

    * every scenario's S* and ΔS* equal the canonical pattern matrices
    * ΔS* == diff(S*) exactly
    * entity count == matrix column count, ids sorted A, B, C
    * shot_structure round-trips through the presence matrix
    * scenario ids unique, prompt strings present and non-empty
    * gen_prompt == eval_prompt + positive_suffix

With --results, additionally cross-checks the benchmark against a set of
evaluation records (entity order, S_ideal, ΔS_ideal) -- use it to confirm
that results you produced were scored against the benchmark you think.

Usage:
    python scripts/validate_mems1k.py
    python scripts/validate_mems1k.py --data mems1k/mems1k.json \
                                      --results outputs/my_model/results.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from span.pattern_library import PATTERNS  # noqa: E402
from stem.matrices import shot_structure  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Report:
    def __init__(self):
        self.errors = []
        self.checks = 0

    def check(self, ok, message):
        self.checks += 1
        if not ok:
            self.errors.append(message)
        return ok

    def line(self, label, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"{label:<44} {status}{('  ' + detail) if detail else ''}")


def load_records(path):
    """Read .jsonl or .json list of evaluation records."""
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        payload = json.load(f)
    return payload["records"] if isinstance(payload, dict) else payload


def validate(data_path, results_path=None):
    rep = Report()
    with open(data_path) as f:
        payload = json.load(f)
    scenarios = payload["scenarios"]

    print(f"MEMS-1K validation: {data_path}")
    print("=" * 66)

    # --- counts -------------------------------------------------------
    print(f"Total scenarios: {len(scenarios)}")
    counts = Counter(s["pattern"] for s in scenarios)
    expected = {name: p["scenarios"] for name, p in PATTERNS.items()}
    for name in PATTERNS:
        got = counts.get(name, 0)
        rep.check(got == expected[name],
                  f"{name}: {got} scenarios, expected {expected[name]}")
        print(f"  {name:<16} {got:>4}")
    print()

    total_ok = sum(counts.values()) == payload["num_scenarios"] == len(scenarios)
    rep.check(total_ok, "declared num_scenarios disagrees with the array length")
    rep.line("Scenario count", total_ok)

    # --- unique ids ---------------------------------------------------
    ids = [s["scenario_id"] for s in scenarios]
    dup = [i for i, n in Counter(ids).items() if n > 1]
    rep.check(not dup, f"duplicate scenario ids: {dup[:5]}")
    rep.line("Duplicate scenario IDs", not dup)

    # --- matrices -----------------------------------------------------
    matrix_bad, align_bad, struct_bad, prompt_bad = [], [], [], []
    styled = n_prompts = 0
    for s in scenarios:
        sid = s["scenario_id"]
        spec = PATTERNS.get(s["pattern"])
        if spec is None:
            matrix_bad.append(f"{sid}: unknown pattern {s['pattern']}")
            continue
        S = np.array(s["presence_matrix"])
        dS = np.array(s["transition_matrix"])

        if not np.array_equal(S, spec["S_star"]):
            matrix_bad.append(f"{sid}: S* != canonical {s['pattern']}")
        if not np.array_equal(dS, spec["delta_S_star"]):
            matrix_bad.append(f"{sid}: ΔS* != canonical {s['pattern']}")
        if not np.array_equal(np.diff(S, axis=0), dS):
            matrix_bad.append(f"{sid}: ΔS* != diff(S*)")
        if not np.isin(S, [0, 1]).all():
            matrix_bad.append(f"{sid}: S* not binary")
        if int(np.count_nonzero(dS)) != s["metadata"]["num_active_transitions"]:
            matrix_bad.append(f"{sid}: num_active_transitions wrong")

        eids = [e["entity_id"] for e in s["entities"]]
        if eids != sorted(eids):
            align_bad.append(f"{sid}: entity ids not in column order: {eids}")
        if len(eids) != S.shape[1]:
            align_bad.append(f"{sid}: {len(eids)} entities vs {S.shape[1]} columns")
        if len(eids) != s["num_entities"]:
            align_bad.append(f"{sid}: num_entities disagrees with entities[]")
        if S.shape[0] != s["num_shots"]:
            align_bad.append(f"{sid}: num_shots disagrees with S* rows")

        if shot_structure(S, eids) != s["shot_structure"]:
            struct_bad.append(f"{sid}: shot_structure != render(S*)")

        suffix = (s.get("generation_settings") or {}).get("positive_suffix") or ""
        for k in (f"shot_{i+1}" for i in range(s["num_shots"])):
            ev, gp = s["eval_prompt"].get(k), s["gen_prompt"].get(k)
            if not ev or not gp:
                prompt_bad.append(f"{sid}/{k}: empty prompt")
            elif gp == ev + suffix:
                styled += 1
            elif gp != ev:
                # eval_prompt must be gen_prompt with the style suffix
                # stripped; anything else means the two have drifted apart.
                prompt_bad.append(f"{sid}/{k}: eval_prompt is not gen_prompt minus suffix")
            n_prompts += 1

    for label, bucket in [
        ("Presence/transition consistency", matrix_bad),
        ("Entity-column alignment", align_bad),
        ("shot_structure round-trip", struct_bad),
        ("Prompt integrity", prompt_bad),
    ]:
        rep.check(not bucket, f"{label}: {len(bucket)} problems, e.g. {bucket[:3]}")
        rep.line(label, not bucket, "" if not bucket else f"({len(bucket)} problems)")

    print(f"  (style suffix present on {styled}/{n_prompts} generation prompts; "
          f"the rest are the short 3-entity continuation shots)")

    # --- optional cross-check against stored results ------------------
    if results_path:
        records = load_records(results_path)
        index = {s["scenario_id"]: s for s in scenarios}
        res_bad, matched = [], 0
        for r in records:
            sid = r.get("scenario_id") or r.get("vid")
            s = index.get(sid)
            if s is None:
                res_bad.append(f"{sid}: not in benchmark")
                continue
            matched += 1
            if r["pattern"] != s["pattern"]:
                res_bad.append(f"{sid}: pattern mismatch")
            if r["entities"] != [e["eval_name"] for e in s["entities"]]:
                res_bad.append(f"{sid}: entity order mismatch")
            if r["S_ideal"] != s["presence_matrix"]:
                res_bad.append(f"{sid}: S_ideal mismatch")
            if r["delta_S_ideal"] != s["transition_matrix"]:
                res_bad.append(f"{sid}: ΔS_ideal mismatch")
        rep.check(not res_bad, f"result cross-check: {len(res_bad)} problems, e.g. {res_bad[:3]}")
        rep.line("Cross-check vs stored results", not res_bad,
                 f"({matched} records)" if not res_bad else f"({len(res_bad)} problems)")

    print("=" * 66)
    if rep.errors:
        print(f"FAILED  {len(rep.errors)} of {rep.checks} checks")
        for e in rep.errors:
            print(f"  - {e}")
        return 1
    print(f"OK  all {rep.checks} checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO, "mems1k", "mems1k.json"))
    ap.add_argument("--results", default=None,
                    help="optional evaluation records (.json/.jsonl) to cross-check")
    args = ap.parse_args()
    sys.exit(validate(args.data, args.results))


if __name__ == "__main__":
    main()
