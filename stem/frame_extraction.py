#!/usr/bin/env python3
"""
STEM Pipeline: Shot frame extraction.

A MEMS-1K video is a single mp4 containing T equal-length shots
concatenated in order. STEM samples one frame per shot (the middle
frame) and queries the detector on it.

`extract_middle_frame` is the function used to produce every number in
the paper (eval_vlm_fullscale.py lines 34-49). `extract_frames_at` is
the K-frame generalisation used for the temporal sampling ablation
(Appendix Y / rebuttal exp3), kept here so alternative sampling
strategies do not need a second copy of the shot arithmetic.

Requires: opencv-python-headless, pillow
"""

DEFAULT_N_SHOTS = 3
MIN_FRAMES = 9  # below this a 3-shot video cannot be split meaningfully


def extract_middle_frame(video_path, shot_idx, n_shots=DEFAULT_N_SHOTS):
    """
    Extract the middle frame of shot `shot_idx` (0-indexed).

    Returns a PIL.Image, or None if the video is unreadable/too short.
    Verbatim from eval_vlm_fullscale.py lines 34-49 (generalised to n_shots).
    """
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < MIN_FRAMES:
        cap.release()
        return None
    shot_len = total // n_shots
    mid = shot_idx * shot_len + shot_len // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None


def extract_frames_at(video_path, shot_idx, fracs=(0.25, 0.5, 0.75),
                      n_shots=DEFAULT_N_SHOTS):
    """
    Extract frames at the given within-shot fractions for shot `shot_idx`.

    Returns a list of PIL.Image (possibly shorter than `fracs` if reads fail).
    Verbatim from rebuttal_experiments/exp3_multiframe_vote.py.
    """
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < MIN_FRAMES:
        cap.release()
        return []
    shot_len = total // n_shots
    start = shot_idx * shot_len
    frames = []
    for fr in fracs:
        pos = start + min(int(fr * shot_len), shot_len - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def probe(video_path, n_shots=DEFAULT_N_SHOTS):
    """Return {frames, fps, shot_len, usable} for a video without decoding it."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {
        "frames": total,
        "fps": fps,
        "shot_len": total // n_shots if total else 0,
        "usable": total >= MIN_FRAMES,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m stem.frame_extraction VIDEO.mp4")
        sys.exit(1)
    path = sys.argv[1]
    print(f"probe({path}) = {probe(path)}")
    for s in range(DEFAULT_N_SHOTS):
        f = extract_middle_frame(path, s)
        print(f"  shot {s+1}: {'None' if f is None else f.size}")
