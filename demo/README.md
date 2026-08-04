# Demo

```
stem_overview_demo.ipynb   the whole pipeline on one real video, no GPU needed
assets/gt_cases.json       the five ground-truth cases, in benchmark schema
assets/gt_videos/*.mp4     the five videos themselves
assets/gt_detected.json    what Qwen2-VL-7B detected in each, cached
```

## The ground-truth cases

Three clips of two cats — the black and white one alone, both together, the
orange one alone — recut into five three-shot videos. All five are supposed to
tell the same Relay story, `A → AB → B`: one cat alone, the other joins, the
first leaves.

One video does that. The other four each break it in exactly one way, so the
right answer is known before any model runs:

| case | shot 1 | shot 2 | shot 3 | what breaks |
|---|---|---|---|---|
| `GT_Perfect` | black | both | orange | nothing |
| `Type_I_ContextBleeding` | black | both | **both** | A never leaves |
| `Type_II_Amnesia` | black | black | black | B never arrives, A never leaves |
| `Type_III_Spurious` | **orange** | both | orange | A appears unprompted; B was early |
| `Type_IV_Reversal` | **both** | black | orange | B's entrance runs backwards |

This is the check that STEM measures narrative failure rather than detector
error: the footage is real, and the edit is the ground truth.

## Run it

```bash
jupyter notebook demo/stem_overview_demo.ipynb
```

The notebook uses the cached detections by default. Set `USE_GPU = True` in
the detection cell to re-run Qwen2-VL-7B on the videos yourself (~16 GB VRAM).

From the command line:

```bash
python scripts/evaluate_directory.py \
    --video-dir demo/assets/gt_videos \
    --benchmark demo/assets/gt_cases.json \
    --output-dir outputs/gt
```

Note that these clips are 1080×1920, far larger than MEMS-1K's 512×320. The
detector downscales frames to 720px on the longest side before querying
(`--max-side`); without it, full-resolution frames exhaust a 24 GB GPU. The
bound is a no-op for MEMS-1K.
