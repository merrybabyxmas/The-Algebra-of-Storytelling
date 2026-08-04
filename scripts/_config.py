"""
Shared config loading for the evaluation CLIs.

A config file supplies defaults; any flag the user passes explicitly wins.
Configs are optional -- every CLI runs with the paper settings out of the box.
"""
import os


def load_config(path):
    """Load a YAML config, resolving `detector_config` if it points elsewhere."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "--config needs PyYAML: pip install pyyaml "
            "(or drop --config and pass the flags directly)"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    nested = cfg.get("detector_config")
    if isinstance(nested, str):
        nested_path = nested if os.path.isabs(nested) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(path))), nested
        )
        if os.path.exists(nested_path):
            with open(nested_path) as f:
                cfg["detector"] = (yaml.safe_load(f) or {}).get("detector", {})
    return cfg


def apply_config(args, parser, config_path, repo_root):
    """
    Overlay config values onto `args` for any option the user left at its
    default. Returns `args` unchanged when no config is given.
    """
    if not config_path:
        return args
    cfg = load_config(config_path)
    detector = cfg.get("detector", {}) or {}
    sampling = cfg.get("sampling", {}) or {}
    output = cfg.get("output", {}) or {}

    mapping = {
        "model_id": detector.get("model_id"),
        "device": detector.get("device"),
        "frames_per_shot": sampling.get("frames_per_shot"),
        "max_side": detector.get("max_side"),
        "benchmark": cfg.get("benchmark"),
    }
    if sampling.get("fracs"):
        mapping["fracs"] = ",".join(str(x) for x in sampling["fracs"])
    if output.get("keep_answers") is False:
        mapping["no_answers"] = True

    for dest, value in mapping.items():
        if value is None or not hasattr(args, dest):
            continue
        if getattr(args, dest) == parser.get_default(dest):
            if dest == "benchmark" and not os.path.isabs(value):
                value = os.path.join(repo_root, value)
            setattr(args, dest, value)
    return args
