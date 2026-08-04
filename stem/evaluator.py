#!/usr/bin/env python3
"""
STEM Pipeline: orchestration.

Two entry points, deliberately separated:

    evaluate_presence(scenario, S_obs)          -- no GPU, no VLM.
        Score a presence matrix you obtained any way you like.

    STEMEvaluator(...).evaluate_video(...)      -- GPU.
        Run the paper's Qwen2-VL detector on a video, then score it.

Both return the same record shape, which is what the evaluation CLIs
write out and what `scripts/aggregate_results.py` reads back.
"""
import json
import os

import numpy as np

from .matrices import transition_matrix, validate_presence_matrix
from .metrics import compute_metrics
from .failure_taxonomy import classify_failures, failure_labels, classify_cells

try:
    from span.pattern_library import get_pattern
except ImportError:  # running from inside the package directory
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from span.pattern_library import get_pattern

DEFAULT_BENCHMARK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mems1k", "mems1k.json"
)


# ============================================================
# Benchmark access
# ============================================================

def load_benchmark(path=DEFAULT_BENCHMARK):
    """Load MEMS-1K as {scenario_id: scenario}."""
    with open(path) as f:
        payload = json.load(f)
    scenarios = payload["scenarios"] if isinstance(payload, dict) else payload
    return {s["scenario_id"]: s for s in scenarios}


def entity_names(scenario):
    """Detector query strings, in the column order of S* (entity A, B, C...)."""
    return [e["eval_name"] for e in scenario["entities"]]


def entity_ids(scenario):
    return [e["entity_id"] for e in scenario["entities"]]


def prescribed_matrices(scenario):
    """
    (S*, ΔS*) for a scenario.

    Taken from the scenario record, which `scripts/build_mems1k.py` derived
    from the canonical pattern library; the pattern library is consulted as
    a cross-check so a hand-edited benchmark file cannot silently drift.
    """
    S_star = np.asarray(scenario["presence_matrix"], dtype=int)
    delta_S_star = np.asarray(scenario["transition_matrix"], dtype=int)

    canonical = get_pattern(scenario["pattern"])
    if not np.array_equal(S_star, canonical["S_star"]):
        raise ValueError(
            f"{scenario['scenario_id']}: presence_matrix disagrees with the "
            f"canonical {scenario['pattern']} pattern"
        )
    if not np.array_equal(delta_S_star, canonical["delta_S_star"]):
        raise ValueError(
            f"{scenario['scenario_id']}: transition_matrix disagrees with the "
            f"canonical {scenario['pattern']} pattern"
        )
    return S_star, delta_S_star


# ============================================================
# Scoring (GPU-free)
# ============================================================

def evaluate_presence(scenario, S_obs, method=None, extra=None):
    """
    Score an observed presence matrix against a MEMS-1K scenario.

    Args:
        scenario: a MEMS-1K scenario record
        S_obs:    T x N binary matrix (list or ndarray); column order must
                  match `scenario["entities"]`
        method:   optional label for the system under evaluation
        extra:    optional dict merged into the record (e.g. raw VLM answers)
    """
    S_star, delta_S_star = prescribed_matrices(scenario)
    S_obs = validate_presence_matrix(
        S_obs, n_entities=S_star.shape[1], n_shots=S_star.shape[0], name="S_obs"
    )

    m = compute_metrics(S_obs, S_star, delta_S_star)
    delta_S_obs = transition_matrix(S_obs)
    ids = entity_ids(scenario)

    record = {
        "scenario_id": scenario["scenario_id"],
        "pattern": scenario["pattern"],
        "entities": entity_names(scenario),
        "S_obs": S_obs.tolist(),
        "S_ideal": S_star.tolist(),
        "delta_S_obs": m["delta_S_obs"],
        "delta_S_ideal": delta_S_star.tolist(),
        "E_error": m["E_error"],
        "ptm": m["match_prescribed"],
        "epa": m["epa"],
        "frob_dist": m["frob_dist"],
        "matrix_distance": m["matrix_distance"],
        "failures": failure_labels(delta_S_star, delta_S_obs, m["E_error"]),
        "failure_types": sorted(classify_failures(delta_S_star, m["E_error"])),
        "failure_cells": classify_cells(delta_S_star, delta_S_obs, m["E_error"], ids),
    }
    # `match_prescribed` is the field name used in the paper's records;
    # keep it as an alias so old analysis scripts keep working.
    record["match_prescribed"] = record["ptm"]
    if method is not None:
        record["method"] = method
    if extra:
        record.update(extra)
    return record


# ============================================================
# Full pipeline (GPU)
# ============================================================

class STEMEvaluator:
    """
    Video -> S_obs -> PTM/EPA/failures, using the paper's Qwen2-VL detector.

    The model is loaded lazily on first use, so constructing an evaluator
    is free and `evaluate_presence` stays importable on machines with no GPU.
    """

    def __init__(self, benchmark=DEFAULT_BENCHMARK, model_id=None, gpu=0,
                 device="cuda:0", frames_per_shot=1, fracs=(0.25, 0.5, 0.75),
                 max_side=None):
        self.benchmark = load_benchmark(benchmark) if isinstance(benchmark, str) else benchmark
        self.gpu = gpu
        self.device = device
        self.frames_per_shot = frames_per_shot
        self.fracs = fracs
        if max_side is None:
            from .presence_detection import DEFAULT_MAX_SIDE
            max_side = DEFAULT_MAX_SIDE
        self.max_side = max_side
        self._model = None
        self._processor = None
        if model_id is None:
            from .presence_detection import DEFAULT_MODEL_ID
            model_id = DEFAULT_MODEL_ID
        self.model_id = model_id

    def _ensure_model(self):
        if self._model is None:
            from .presence_detection import load_vlm
            self._model, self._processor = load_vlm(self.model_id, gpu=self.gpu)
        return self._model, self._processor

    def detect(self, video_path, scenario):
        """Video -> (S_obs, raw_answers)."""
        from .presence_detection import build_S_obs, build_S_obs_multiframe
        model, processor = self._ensure_model()
        names = entity_names(scenario)
        n_shots = scenario["num_shots"]
        if self.frames_per_shot > 1:
            return build_S_obs_multiframe(
                model, processor, video_path, names,
                fracs=self.fracs, n_shots=n_shots, device=self.device,
                max_side=self.max_side
            )
        return build_S_obs(
            model, processor, video_path, names,
            n_shots=n_shots, device=self.device, max_side=self.max_side
        )

    def evaluate_video(self, video_path, scenario_id, method=None, keep_answers=True):
        """Detect presence in a video and score it against its scenario."""
        scenario = self.benchmark.get(scenario_id)
        if scenario is None:
            raise KeyError(f"scenario_id {scenario_id!r} is not in the benchmark")
        S_obs, answers = self.detect(video_path, scenario)
        extra = {"video": os.path.basename(video_path)}
        if keep_answers:
            extra["answers"] = answers
        return evaluate_presence(scenario, S_obs, method=method, extra=extra)


def scenario_id_from_path(video_path):
    """MEMS-1K videos are named `<scenario_id>.mp4`."""
    return os.path.splitext(os.path.basename(video_path))[0]


if __name__ == "__main__":
    bench = load_benchmark()
    sid = next(iter(bench))
    sc = bench[sid]
    print(f"Loaded {len(bench)} scenarios; demo on {sid} ({sc['pattern']})")
    print(f"  entities: {entity_names(sc)}")
    print(f"  S*      : {sc['presence_matrix']}")
    perfect = evaluate_presence(sc, sc["presence_matrix"], method="oracle")
    print(f"  oracle  : PTM={perfect['ptm']:.3f} EPA={perfect['epa']:.3f} "
          f"failures={perfect['failure_types']}")
