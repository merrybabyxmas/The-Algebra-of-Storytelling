#!/usr/bin/env python3
"""
Evaluate a single multi-shot video with the full STEM pipeline.

    video -> middle frame per shot -> Qwen2-VL presence queries
          -> S_obs -> ΔS_obs -> E -> PTM / EPA / failure types

Requires a GPU and the detector extras (see requirements.txt). If you
already have presence predictions, use `evaluate_predictions.py` instead;
it needs neither.

The video must contain the scenario's shots concatenated in order, all of
equal length -- the same convention used for every video in the paper.

Usage:
    python scripts/evaluate_video.py \
        --video samples/DM1K_777_Relay_001.mp4 \
        --scenario-id DM1K_777_Relay_001 \
        --output outputs/DM1K_777_Relay_001.json
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import apply_config  # noqa: E402
from stem.evaluator import (  # noqa: E402
    STEMEvaluator, DEFAULT_BENCHMARK, scenario_id_from_path,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--scenario-id", default=None,
                    help="defaults to the video filename stem")
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--method", default=None, help="label for the system under evaluation")
    ap.add_argument("--output", default=None, help="write the record as JSON")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-id", default=None, help="override the detector checkpoint")
    ap.add_argument("--frames-per-shot", type=int, default=1,
                    help="1 = paper setting (middle frame); >1 = majority vote")
    ap.add_argument("--fracs", default="0.25,0.5,0.75",
                    help="within-shot sample positions when frames-per-shot > 1")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale frames whose longest side exceeds this "
                         "(default 720; 0 disables). No-op for MEMS-1K.")
    ap.add_argument("--config", default=None,
                    help="YAML config supplying defaults, e.g. configs/default_eval.yaml")
    args = ap.parse_args()
    args = apply_config(args, ap, args.config, REPO_ROOT)

    if not os.path.exists(args.video):
        raise SystemExit(f"video not found: {args.video}")
    scenario_id = args.scenario_id or scenario_id_from_path(args.video)

    evaluator = STEMEvaluator(
        benchmark=args.benchmark, model_id=args.model_id, gpu=args.gpu,
        device=args.device, frames_per_shot=args.frames_per_shot,
        max_side=args.max_side,
        fracs=tuple(float(x) for x in args.fracs.split(",")),
    )
    record = evaluator.evaluate_video(args.video, scenario_id, method=args.method)

    scenario = evaluator.benchmark[scenario_id]
    print(f"{scenario_id}  ({scenario['pattern']}: "
          f"{' -> '.join(scenario['shot_structure'])})")
    print(f"  entities : {record['entities']}")
    print(f"  S*       : {record['S_ideal']}")
    print(f"  S_obs    : {record['S_obs']}")
    print(f"  E        : {record['E_error']}")
    print(f"\n  PTM = {record['ptm']:.4f}   EPA = {record['epa']:.4f}   "
          f"||E||_F = {record['frob_dist']:.4f}")
    if record["failure_cells"]:
        print("\n  Detected failures:")
        for c in record["failure_cells"]:
            print(f"    - Type {c['type']}: {c['type_name']}")
            print(f"        transition={c['transition_index']}, entity={c['entity_id']}")
    else:
        print("\n  Detected failures: none")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
