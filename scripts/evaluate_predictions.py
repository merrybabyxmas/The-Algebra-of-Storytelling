#!/usr/bin/env python3
"""
Score presence predictions with STEM -- no VLM, no GPU, no videos.

Use this when you already have per-shot entity presence from your own
detector, a different VLM, or human annotation. This is the recommended
entry point for anyone who wants the metric but not our detector.

Input is a JSON file mapping scenario_id -> observed presence matrix:

    {
      "DM1K_777_Relay_001": [[1, 0], [1, 1], [1, 1]],
      "DM1K_508_Relay_002": {"S_obs": [[1, 0], [1, 1], [0, 1]]}
    }

Rows are shots in order; columns follow `entities` in the scenario
(entity_id order A, B, C). Both the bare-list and the {"S_obs": ...}
form are accepted.

Usage:
    # your own MEMS-1K predictions (default benchmark)
    python scripts/evaluate_predictions.py --observed my_preds.json \
        --output outputs/my_model.jsonl --method my_model

    # the bundled ground-truth example, which uses the demo case set
    python scripts/evaluate_predictions.py \
        --observed examples/predicted_presence.json \
        --benchmark demo/assets/gt_cases.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stem.evaluator import load_benchmark, evaluate_presence, DEFAULT_BENCHMARK  # noqa: E402
from stem.failure_taxonomy import TYPE_NAMES  # noqa: E402


def signed(v):
    """Render a transition value: '+1', '-1', '0' (never '+0')."""
    return f"{v:+d}" if v else "0"


def read_predictions(path):
    with open(path) as f:
        payload = json.load(f)
    preds = payload.get("predictions", payload) if isinstance(payload, dict) else payload
    if isinstance(preds, list):  # [{"scenario_id": ..., "S_obs": ...}, ...]
        preds = {p["scenario_id"]: p["S_obs"] for p in preds}
    return {
        sid: (v["S_obs"] if isinstance(v, dict) else v)
        for sid, v in preds.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observed", required=True, help="predicted presence matrices (JSON)")
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--scenario", default=None, help="score a single scenario_id")
    ap.add_argument("--method", default=None, help="label for the system under evaluation")
    ap.add_argument("--output", default=None, help="write records as .jsonl")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    args = ap.parse_args()

    bench = load_benchmark(args.benchmark)
    preds = read_predictions(args.observed)
    if args.scenario:
        if args.scenario not in preds:
            raise SystemExit(f"{args.scenario} not present in {args.observed}")
        preds = {args.scenario: preds[args.scenario]}

    unknown = [sid for sid in preds if sid not in bench]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} scenario id(s) are not in {args.benchmark}, "
            f"e.g. {unknown[:3]}\n"
            f"If these are the ground-truth demo cases, point --benchmark at "
            f"demo/assets/gt_cases.json."
        )

    records = []
    for sid, S_obs in preds.items():
        rec = evaluate_presence(bench[sid], S_obs, method=args.method)
        records.append(rec)

        if not args.quiet:
            sc = bench[sid]
            print(f"\n{sid}  ({sc['pattern']}, {' -> '.join(sc['shot_structure'])})")
            print(f"  entities : {[e['eval_name'] for e in sc['entities']]}")
            print(f"  S*       : {rec['S_ideal']}")
            print(f"  S_obs    : {rec['S_obs']}")
            print(f"  ΔS*      : {rec['delta_S_ideal']}")
            print(f"  ΔS_obs   : {rec['delta_S_obs']}")
            print(f"  E        : {rec['E_error']}")
            print(f"  PTM = {rec['ptm']:.4f}   EPA = {rec['epa']:.4f}   "
                  f"||E||_F = {rec['frob_dist']:.4f}")
            if rec["failure_cells"]:
                print("  Detected failures:")
                for c in rec["failure_cells"]:
                    print(f"    - Type {c['type']}: {c['type_name']}")
                    print(f"        transition={c['transition_index']}, "
                          f"entity={c['entity_id']} "
                          f"(prescribed {signed(c['prescribed'])}, "
                          f"observed {signed(c['observed'])})")
            else:
                print("  Detected failures: none")

    print("\n" + "=" * 58)
    print(f"Scored {len(records)} scenario(s)"
          + (f" for '{args.method}'" if args.method else ""))
    print(f"  mean PTM = {np.mean([r['ptm'] for r in records]):.4f}")
    print(f"  mean EPA = {np.mean([r['epa'] for r in records]):.4f}")
    counts = {t: 0 for t in ("I", "II", "III", "IV")}
    for r in records:
        for t in r["failure_types"]:
            if t in counts:
                counts[t] += 1
    for t, n in counts.items():
        print(f"  videos with Type {t:<3} ({TYPE_NAMES[t]:<19}): "
              f"{n:>5}  ({100.0 * n / len(records):.1f}%)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
