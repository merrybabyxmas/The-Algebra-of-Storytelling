#!/usr/bin/env python3
"""
Build `mems1k/mems1k.json` (+ per-pattern splits) from the raw prompt file
used to generate and evaluate the paper's videos.

This script exists for provenance: the released benchmark is a *mechanical*
restructuring of `prompts/diagonal_diversified_prompts.json` from the
experiment repository, not a re-authoring. It only

  * renames fields into the documented public schema,
  * attaches S* / ΔS* from the canonical pattern library, and
  * derives `shot_structure` and transition counts from those matrices.

Entity column order is `sorted(core_entities)` (A, B, C), exactly as in
eval_vlm_fullscale.py line 117 -- this is what makes the released
matrices line up with the published results.

One deviation from a pure copy, see mems1k/README.md "Known issues":
in the source file the `eval_prompt` field is a stale Relay-template
string for the 800 non-Relay scenarios (it was never re-permuted when
the derived patterns were added). `gen_prompt` -- the field every
generator actually read, and the only one that produced the released
videos -- is correct throughout. The published `eval_prompt` is
therefore derived as `gen_prompt` minus the shared style suffix, which
is exactly what the field name promises. No generation prompt is edited.

Usage:
    python scripts/build_mems1k.py --source /path/to/diagonal_diversified_prompts.json
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from span.pattern_library import PATTERNS, get_pattern  # noqa: E402
from stem.matrices import shot_structure  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_VERSION = "1.0.0"
N_SHOTS = 3


def build_scenario(scenario_id, raw):
    """Convert one raw prompt record into a public MEMS-1K scenario."""
    meta = raw["metadata"]
    pattern = meta["pattern_name"]
    spec = get_pattern(pattern)

    S_star = spec["S_star"]
    delta_S_star = spec["delta_S_star"]

    # Column order: sorted entity keys (A, B, C) -- eval_vlm_fullscale.py:117
    core = meta["core_entities"]
    types = meta.get("entity_types", {})
    ids = sorted(core.keys())
    entities = [
        {
            "entity_id": eid,
            # In MEMS-1K the generation noun phrase and the detector query
            # string coincide; both are kept so alternative detectors can
            # diverge from the generator wording without a schema change.
            "name": core[eid],
            "eval_name": core[eid],
            "entity_type": types.get(eid),
        }
        for eid in ids
    ]

    if len(entities) != S_star.shape[1]:
        raise ValueError(
            f"{scenario_id}: {len(entities)} entities but pattern {pattern} "
            f"prescribes {S_star.shape[1]} columns"
        )

    shots = raw["shots"]
    shot_keys = [f"shot_{i+1}" for i in range(N_SHOTS)]
    missing = [k for k in shot_keys if k not in shots]
    if missing:
        raise ValueError(f"{scenario_id}: missing shots {missing}")

    settings = raw.get("system_settings", {})
    suffix = settings.get("positive_suffix") or ""

    def bare(key):
        """gen_prompt with the shared style suffix removed."""
        text = shots[key]["gen_prompt"]
        return text[: -len(suffix)] if suffix and text.endswith(suffix) else text

    return {
        "scenario_id": scenario_id,
        "pattern": pattern,
        "pattern_class": spec["type"],              # Basis | Derived
        "notation": meta.get("pattern"),            # e.g. "A→AB→B"
        "theme": meta.get("theme"),
        "num_shots": N_SHOTS,
        "num_entities": len(entities),
        "entities": entities,
        "shot_structure": shot_structure(S_star, ids),
        "presence_matrix": S_star.tolist(),         # S*
        "transition_matrix": delta_S_star.tolist(),  # ΔS*
        "gen_prompt": {k: shots[k]["gen_prompt"] for k in shot_keys},
        "eval_prompt": {k: bare(k) for k in shot_keys},
        "diag_prompt": {k: shots[k].get("diag_prompt") for k in shot_keys},
        "generation_settings": {
            "camera": settings.get("camera"),
            "positive_suffix": settings.get("positive_suffix"),
            "negative_prompt": settings.get("negative_prompt"),
        },
        "metadata": {
            "basis_type": spec["type"].lower(),
            "num_active_transitions": int(np.count_nonzero(delta_S_star)),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", required=True,
        help="path to diagonal_diversified_prompts.json from the experiment repo",
    )
    ap.add_argument("--out-dir", default=os.path.join(REPO, "mems1k"))
    args = ap.parse_args()

    with open(args.source) as f:
        raw_prompts = json.load(f)

    scenarios = [build_scenario(sid, raw) for sid, raw in sorted(raw_prompts.items())]

    counts = Counter(s["pattern"] for s in scenarios)
    expected = {name: p["scenarios"] for name, p in PATTERNS.items()}
    if counts != Counter(expected):
        raise SystemExit(f"pattern counts {dict(counts)} != expected {expected}")

    payload = {
        "name": "MEMS-1K",
        "version": BENCHMARK_VERSION,
        "description": (
            "Multi-Entity Multi-Shot 1K: 1,000 three-shot narrative scenarios "
            "over 7 patterns, each with a prescribed presence matrix S* and "
            "transition matrix ΔS*."
        ),
        "num_scenarios": len(scenarios),
        "num_shots": N_SHOTS,
        "patterns": {
            name: {
                "class": p["type"],
                "notation": p["notation"],
                "num_entities": p["entities"],
                "num_scenarios": p["scenarios"],
                "presence_matrix": p["S_star"].tolist(),
                "transition_matrix": p["delta_S_star"].tolist(),
            }
            for name, p in PATTERNS.items()
        },
        "scenarios": scenarios,
    }

    os.makedirs(os.path.join(args.out_dir, "splits"), exist_ok=True)
    out_path = os.path.join(args.out_dir, "mems1k.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path}  ({len(scenarios)} scenarios)")

    for name in PATTERNS:
        subset = [s for s in scenarios if s["pattern"] == name]
        split_path = os.path.join(args.out_dir, "splits", f"{name.lower()}.json")
        with open(split_path, "w") as f:
            json.dump(
                {**{k: v for k, v in payload.items() if k not in ("scenarios", "patterns")},
                 "pattern": name,
                 "num_scenarios": len(subset),
                 "scenarios": subset},
                f, ensure_ascii=False, indent=1,
            )
        print(f"  splits/{name.lower()}.json  ({len(subset)})")


if __name__ == "__main__":
    main()
