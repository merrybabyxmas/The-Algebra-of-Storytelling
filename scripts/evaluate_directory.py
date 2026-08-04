#!/usr/bin/env python3
"""
Evaluate a directory of generated videos against MEMS-1K.

Videos are matched to scenarios by filename stem: `DM1K_777_Relay_001.mp4`
scores against scenario `DM1K_777_Relay_001`. Subdirectories are searched
recursively, so the layout `{video_dir}/{Pattern}/{scenario_id}.mp4` works
as-is. Files whose stem is not a scenario id are skipped and reported.

Resumable: an existing output file is read back and already-scored
scenarios are skipped. Shard across GPUs with --num-shards/--shard-id.

Usage:
    python scripts/evaluate_directory.py \
        --video-dir generated_videos/my_model \
        --output-dir outputs/my_model --method my_model --gpu 0

    # 4 GPUs, one process each
    for i in 0 1 2 3; do
      python scripts/evaluate_directory.py --video-dir videos/my_model \
        --output-dir outputs/my_model --gpu $i --num-shards 4 --shard-id $i &
    done; wait
    python scripts/aggregate_results.py --input outputs/my_model/results.jsonl
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import apply_config  # noqa: E402
from stem.evaluator import (  # noqa: E402
    STEMEvaluator, DEFAULT_BENCHMARK, scenario_id_from_path,
)

SAVE_EVERY = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    ap.add_argument("--method", default=None, help="label for the system under evaluation")
    ap.add_argument("--pattern", default=None, help="restrict to one narrative pattern")
    ap.add_argument("--limit", type=int, default=0, help="cap videos (0 = all)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--frames-per-shot", type=int, default=1)
    ap.add_argument("--fracs", default="0.25,0.5,0.75")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--no-answers", action="store_true",
                    help="do not store raw VLM answer strings")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale frames whose longest side exceeds this "
                         "(default 720; 0 disables). No-op for MEMS-1K.")
    ap.add_argument("--config", default=None,
                    help="YAML config supplying defaults, e.g. configs/default_eval.yaml")
    args = ap.parse_args()
    args = apply_config(args, ap, args.config, REPO_ROOT)

    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard-id must be in [0, --num-shards)")

    evaluator = STEMEvaluator(
        benchmark=args.benchmark, model_id=args.model_id, gpu=args.gpu,
        device=args.device, frames_per_shot=args.frames_per_shot,
        max_side=args.max_side,
        fracs=tuple(float(x) for x in args.fracs.split(",")),
    )
    bench = evaluator.benchmark

    videos = sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
    matched, unmatched = [], []
    for vp in videos:
        sid = scenario_id_from_path(vp)
        if sid not in bench:
            unmatched.append(os.path.basename(vp))
        elif args.pattern and bench[sid]["pattern"] != args.pattern:
            continue
        else:
            matched.append((vp, sid))

    if args.num_shards > 1:
        matched = [v for i, v in enumerate(matched) if i % args.num_shards == args.shard_id]
    if args.limit:
        matched = matched[:args.limit]

    print(f"{len(videos)} mp4 found, {len(matched)} to score"
          + (f" (shard {args.shard_id}/{args.num_shards})" if args.num_shards > 1 else ""))
    if unmatched:
        print(f"  skipped {len(unmatched)} file(s) with no matching scenario id, "
              f"e.g. {unmatched[:3]}")
    if not matched:
        raise SystemExit("nothing to evaluate")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
    jsonl_path = os.path.join(args.output_dir, f"results{suffix}.jsonl")

    records, done = [], set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        done = {r["scenario_id"] for r in records}
        print(f"  resuming: {len(done)} already scored")

    def flush():
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    t0 = time.time()
    todo = [(vp, sid) for vp, sid in matched if sid not in done]
    for i, (vp, sid) in enumerate(todo, 1):
        records.append(evaluator.evaluate_video(
            vp, sid, method=args.method, keep_answers=not args.no_answers))
        if i % SAVE_EVERY == 0:
            flush()
            rate = i / (time.time() - t0)
            eta = (len(todo) - i) / rate / 60 if rate else 0
            print(f"  [{i}/{len(todo)}] {rate*60:.1f} videos/min, ETA {eta:.0f} min")
    flush()

    csv_path = os.path.join(args.output_dir, f"results{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scenario_id", "pattern", "ptm", "epa",
                    "frob_dist", "failure_types"])
        for r in records:
            w.writerow([r.get("method", ""), r["scenario_id"], r["pattern"],
                        f"{r['ptm']:.6f}", f"{r['epa']:.6f}",
                        f"{r['frob_dist']:.6f}", "|".join(r["failure_types"])])

    ptm = np.array([r["ptm"] for r in records])
    epa = np.array([r["epa"] for r in records])
    print(f"\nScored {len(records)} videos in {(time.time()-t0)/60:.1f} min")
    print(f"  PTM = {ptm.mean():.4f}   EPA = {epa.mean():.4f}")
    print(f"  wrote {jsonl_path}")
    print(f"  wrote {csv_path}")
    if args.num_shards > 1:
        print("\nMerge the shards, then aggregate:")
        print(f"  cat {args.output_dir}/results.shard*.jsonl > {args.output_dir}/results.jsonl")
    print(f"  python scripts/aggregate_results.py --input {args.output_dir}/results.jsonl")


if __name__ == "__main__":
    main()
