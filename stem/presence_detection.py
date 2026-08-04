#!/usr/bin/env python3
"""
STEM Pipeline: VLM entity-presence detection (video -> S_obs).

This is the only GPU-dependent part of STEM. Everything downstream
(matrices, metrics, taxonomy) consumes S_obs and never imports this
module, so a different detector -- or human annotation -- can be
substituted without touching the metric code.

Model, prompt and decoding settings are those used for every number in
the paper (eval_vlm_fullscale.py lines 62-72 and 130-153):
    Qwen2-VL-7B-Instruct, fp16, greedy (do_sample=False), max_new_tokens=5.

Requirements (GPU only):
    pip install transformers torch qwen-vl-utils opencv-python-headless pillow
"""
import numpy as np

from .frame_extraction import extract_middle_frame, extract_frames_at

# ============================================================
# VLM Query (from eval_vlm_fullscale.py lines 130-153)
# ============================================================

#: Exact prompt used for every detection in the paper. Changing this
#: changes the benchmark; report it if you do.
ENTITY_QUERY = "Is there a {entity} visible in this image? Answer only Yes or No."

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

#: Frames longer than this on their longest side are downscaled before the
#: query. Qwen2-VL turns pixels into visual tokens, so full-resolution
#: footage (e.g. 1080x1920 phone video) will exhaust a 24 GB GPU.
#:
#: This is a no-op for MEMS-1K, whose videos are 512x320 -- the benchmark
#: numbers are unaffected by it. 720 is the bound the paper used when it
#: ran this detector on real high-resolution footage. Pass max_side=0 to
#: disable resizing entirely.
DEFAULT_MAX_SIDE = 720


def _fit(image, max_side=DEFAULT_MAX_SIDE):
    """Downscale so the longest side is at most `max_side`. Never upscales."""
    if not max_side:
        return image
    from PIL import Image
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def load_vlm(model_id=DEFAULT_MODEL_ID, gpu=0):
    """Load Qwen2-VL-7B-Instruct (from eval_vlm_fullscale.py lines 62-72)."""
    import os, torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def query_entity(model, processor, frame, entity, device="cuda:0",
                 max_side=DEFAULT_MAX_SIDE):
    """
    Query the VLM for binary entity presence. Greedy decoding, no threshold.
    Directly extracted from eval_vlm_fullscale.py lines 134-153.

    Returns (0|1, raw_answer_string).
    """
    import os, tempfile
    import torch
    from qwen_vl_utils import process_vision_info

    frame = _fit(frame, max_side)

    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="stem_query_")
    os.close(fd)
    try:
        frame.save(tmp_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{tmp_path}"},
                {"type": "text", "text": ENTITY_QUERY.format(entity=entity)}
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            answer = processor.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return (1 if answer.lower().startswith("yes") else 0), answer


# ============================================================
# S_obs Construction (from eval_vlm_fullscale.py lines 122-153)
# ============================================================

def build_S_obs(model, processor, video_path, entity_names,
                n_shots=3, device="cuda:0", max_side=DEFAULT_MAX_SIDE):
    """
    Build the observed presence matrix S_obs (n_shots x n_entities) from
    the middle frame of each shot. This is the paper configuration.

    Returns (S_obs, raw_answers), where raw_answers maps 's{shot}_e{entity}'
    to the VLM's raw string -- the same field stored in the paper's records.
    """
    n_ent = len(entity_names)
    S_obs = np.zeros((n_shots, n_ent), dtype=int)
    answers = {}

    for shot_idx in range(n_shots):
        frame = extract_middle_frame(video_path, shot_idx, n_shots=n_shots)
        if frame is None:
            continue
        for ent_idx, ent_name in enumerate(entity_names):
            val, raw = query_entity(model, processor, frame, ent_name, device, max_side)
            S_obs[shot_idx, ent_idx] = val
            answers[f"s{shot_idx}_e{ent_idx}"] = raw

    return S_obs, answers


def build_S_obs_multiframe(model, processor, video_path, entity_names,
                           fracs=(0.25, 0.5, 0.75), n_shots=3, device="cuda:0",
                           max_side=DEFAULT_MAX_SIDE):
    """
    Majority-vote variant: query K frames per shot and take the per-cell
    majority. Used for the temporal sampling ablation (Appendix Y).

    With the default 3 fractions this costs ~3x the queries of `build_S_obs`.
    """
    n_ent = len(entity_names)
    S_obs = np.zeros((n_shots, n_ent), dtype=int)
    answers = {}

    for shot_idx in range(n_shots):
        frames = extract_frames_at(video_path, shot_idx, fracs, n_shots=n_shots)
        if not frames:
            continue
        for ent_idx, ent_name in enumerate(entity_names):
            votes = []
            for f_i, frame in enumerate(frames):
                val, raw = query_entity(model, processor, frame, ent_name, device, max_side)
                votes.append(val)
                answers[f"s{shot_idx}_e{ent_idx}_f{f_i}"] = raw
            # strict majority; ties (only possible with even K) resolve to 0
            S_obs[shot_idx, ent_idx] = int(sum(votes) * 2 > len(votes))

    return S_obs, answers


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from span.pattern_library import PATTERN_NAMES

    print("=== STEM VLM Detector ===")
    print(f"Model:  {DEFAULT_MODEL_ID}")
    print(f"Query:  '{ENTITY_QUERY}'")
    print(f"Patterns: {PATTERN_NAMES}")
    print("\nUsage (requires GPU):")
    print("  model, processor = load_vlm(gpu=0)")
    print("  S_obs, answers = build_S_obs(model, processor, 'video.mp4', ['dog', 'cat'])")
