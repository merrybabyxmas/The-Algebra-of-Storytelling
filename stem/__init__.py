"""
STEM: State Transition Error Matching.

The evaluation pipeline. Two layers, importable independently:

  detection (GPU)   frame_extraction, presence_detection   video -> S_obs
  algebra   (CPU)   matrices, metrics, failure_taxonomy     S_obs -> PTM/EPA/failures

Only the detection layer needs torch/transformers, so the metric layer
imports cleanly on a CPU-only machine.
"""
from .matrices import (
    transition_matrix,
    error_matrix,
    prescribed_mask,
    shot_structure,
    presence_from_structure,
    validate_presence_matrix,
)
from .metrics import compute_metrics, ptm, epa, cohens_d
from .failure_taxonomy import (
    classify_failures,
    failure_labels,
    classify_cells,
    TYPE_NAMES,
)
from .evaluator import (
    load_benchmark,
    evaluate_presence,
    entity_names,
    entity_ids,
    prescribed_matrices,
    scenario_id_from_path,
    STEMEvaluator,
)

__version__ = "1.0.0"

__all__ = [
    "transition_matrix", "error_matrix", "prescribed_mask", "shot_structure",
    "presence_from_structure", "validate_presence_matrix",
    "compute_metrics", "ptm", "epa", "cohens_d",
    "classify_failures", "failure_labels", "classify_cells", "TYPE_NAMES",
    "load_benchmark", "evaluate_presence", "entity_names", "entity_ids",
    "prescribed_matrices", "scenario_id_from_path", "STEMEvaluator",
    "__version__",
]
