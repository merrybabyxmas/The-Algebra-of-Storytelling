# SPAN / STEM — MEMS-1K benchmark and narrative-transition evaluation

Official benchmark and evaluation code for **The Algebra of Storytelling: A
State Transition Benchmark for Spanning Multi-Entity Multi-Shot Narratives**
(ACM Multimedia 2026).

<p>
  <a href="https://merrybabyxmas.github.io/The-Algebra-of-Storytelling/"><img alt="GitHub Pages" src="https://img.shields.io/badge/GitHub%20Pages-live-4c6dff?style=for-the-badge"></a>
  <a href="demo/stem_overview_demo.ipynb"><img alt="Demo Notebook" src="https://img.shields.io/badge/Demo-notebook-24a477?style=for-the-badge"></a>
  <a href="mems1k/mems1k.json"><img alt="MEMS-1K" src="https://img.shields.io/badge/MEMS--1K-1000%20scenarios-eb6660?style=for-the-badge"></a>
  <a href="scripts/evaluate_predictions.py"><img alt="STEM CLI" src="https://img.shields.io/badge/STEM-CLI-1c2635?style=for-the-badge"></a>
</p>

![STEM ground-truth diagnostic](docs/assets/stem-gt-diagnostic.gif)

Multi-shot video generators are getting very good at keeping a character
*looking* the same. They remain bad at making that character **arrive** and
**leave** when the story says so. Identity-consistency metrics cannot see this
failure — a model that renders the same two people in all three shots scores
well on consistency while telling no story at all.

This repository provides the two pieces needed to measure that directly:

- **MEMS-1K** — 1,000 three-shot scenarios across 7 narrative patterns, each
  prescribing exactly who must be on screen in each shot.
- **STEM** — the evaluation pipeline that turns a video into a presence matrix,
  differences it into transitions, and reports **PTM**, **EPA** and a four-type
  failure diagnosis.

```
story prompt ──> generated shots ──> VLM presence detection ──> S_obs
                                                                  │
                        ΔS_obs = S_obs[1:] − S_obs[:−1]  ◄─────────┘
                        E      = ΔS_obs − ΔS*
                                     │
                        PTM · EPA · Type I–IV failures
```

**Start here:** [`demo/stem_overview_demo.ipynb`](demo/stem_overview_demo.ipynb)
runs that entire diagram on real video, with no GPU.

---

## Install

```bash
git clone https://github.com/merrybabyxmas/The-Algebra-of-Storytelling.git
cd SPAN-STEM
pip install -r requirements.txt          # metrics only: numpy, scipy
pip install -r requirements-detector.txt # optional: the Qwen2-VL detector (GPU)
```

The metric layer is deliberately dependency-light. `torch` and `transformers`
are needed **only** if you want us to run the detector for you.

## Use

### You already have presence predictions (no GPU)

The recommended entry point. Bring presence from any detector, any VLM, or
human annotation; STEM scores it.

```bash
# your own predictions, scored against MEMS-1K
python scripts/evaluate_predictions.py --observed my_preds.json --method my_model
```

```json
{
  "DM1K_777_Relay_001": [[1, 0], [1, 1], [1, 1]]
}
```

Rows are shots in order, columns follow the scenario's `entities` list.

The bundled example uses the ground-truth cases below, so point `--benchmark`
at their case file:

```bash
python scripts/evaluate_predictions.py \
    --observed examples/predicted_presence.json \
    --benchmark demo/assets/gt_cases.json
```

### You have generated videos (GPU)

```bash
# one video
python scripts/evaluate_video.py --video my_video.mp4 \
    --scenario-id DM1K_777_Relay_001

# a directory, sharded over 4 GPUs
for i in 0 1 2 3; do
  python scripts/evaluate_directory.py --video-dir videos/my_model \
      --output-dir outputs/my_model --method my_model \
      --gpu $i --num-shards 4 --shard-id $i &
done; wait
cat outputs/my_model/results.shard*.jsonl > outputs/my_model/results.jsonl
python scripts/aggregate_results.py --input outputs/my_model/results.jsonl
```

Videos are matched to scenarios by filename stem, so name them
`<scenario_id>.mp4`. Each must contain the scenario's three shots concatenated
in order, all of equal length.

### As a library

```python
from stem import load_benchmark, evaluate_presence

bench = load_benchmark()
record = evaluate_presence(bench["DM1K_777_Relay_001"], [[1, 0], [1, 1], [1, 1]])

record["ptm"]            # 0.5
record["epa"]            # 0.833...
record["failure_types"]  # ['I']  -- Context Bleeding: the first entity never left
```

## The metrics

| | what it asks | scored over |
|---|---|---|
| **PTM** | Did the prescribed entrances and exits actually happen? | only cells where `ΔS* != 0` |
| **EPA** | Was the right cast on screen? | all `T × N` presence cells |
| `‖E‖_F` | How far off was the transition matrix overall? | the whole error matrix |

PTM's sparsity is the design decision that matters. A scenario prescribing two
transitions should not be graded on cells where nothing was supposed to happen;
those belong to EPA and to Type III.

Every non-zero cell of `E = ΔS_obs − ΔS*` gets exactly one failure type:

| type | name | condition | reads as |
|---|---|---|---|
| I | Context Bleeding | `ΔS* = −1` | the entity refuses to leave |
| II | Amnesia | `ΔS* = +1` | the entity never arrives |
| III | Spurious Transition | `ΔS* = 0` | it changes state unprompted |
| IV | Reversal | `\|E\| = 2` | the transition ran backwards |

## MEMS-1K

1,000 scenarios over 7 patterns. **Relay** and **Split** are the algebraic
basis — Theorem 3.1 proves that column permutations and integer combinations of
those two span every transition topology in `Z^{2×3}`; the other five are
derived.

| pattern | class | structure | entities | prescribed | scenarios |
|---|---|---|---|---|---|
| Relay | Basis | A → AB → B | 2 | 2 | 200 |
| Split | Basis | AB → A → B | 2 | 3 | 200 |
| Accumulation | Derived | A → AB → ABC | 3 | 2 | 200 |
| Convergence | Derived | A → B → AB | 2 | 3 | 100 |
| Sliding Window | Derived | AB → BC → C | 3 | 3 | 100 |
| Reduction | Derived | ABC → AB → A | 3 | 2 | 100 |
| Reverse Relay | Derived | B → AB → A | 2 | 2 | 100 |

Each scenario ships the generation prompt for every shot, the entity list, and
the prescribed `S*` / `ΔS*`. Field-by-field documentation is in
[`mems1k/README.md`](mems1k/README.md); the machine-readable contract is
[`mems1k/schema.json`](mems1k/schema.json).

```bash
python scripts/validate_mems1k.py    # 13 structural + consistency checks
```

## Ground-truth validation

A metric built on a VLM invites one obvious objection: are the failures real,
or is the detector wrong?

[`demo/assets/gt_videos/`](demo/assets/gt_videos) answers it with real footage.
Three clips of two cats — one alone, both together, the other alone — are recut
into five three-shot videos. All five are supposed to tell the same Relay
story; one does, and the other four each break it in exactly one way. Because
the edit is known, the correct answer is known *independently of any model*:

| case | shots cut from | PTM | EPA | STEM reports |
|---|---|---|---|---|
| `GT_Perfect` | black, both, orange | 1.00 | 1.00 | no failure |
| `Type_I_ContextBleeding` | black, both, **both** | 0.50 | 0.83 | I |
| `Type_II_Amnesia` | black, black, black | 0.00 | 0.50 | I, II |
| `Type_III_Spurious` | **orange**, both, orange | 0.50 | 0.67 | II, III |
| `Type_IV_Reversal` | **both**, black, orange | 0.50 | 0.67 | III, IV |

Every row matches what the edit actually did, and reproduces the paper's GT
validation run exactly. `tests/test_gt_cases.py` pins these as golden values,
so a change that would break the metric fails the suite.

```bash
python scripts/evaluate_directory.py --video-dir demo/assets/gt_videos \
    --benchmark demo/assets/gt_cases.json --output-dir outputs/gt
```

## Scope

This repository releases the benchmark and the evaluation pipeline. It contains
no video-generation code and does not redistribute the systems evaluated in the
paper — generate videos with any multi-shot system and score them through the
interface above.

The demo videos are real footage. No generated videos are included.

## Layout

```
mems1k/      benchmark JSON, per-pattern splits, schema
stem/        evaluation pipeline (detection | matrices | metrics | taxonomy)
span/        the algebra: basis matrices, operators, completeness proof
scripts/     CLIs for evaluation, validation, aggregation
demo/        the overview notebook, the ground-truth videos and their cases
examples/    sample input and output files
tests/       72 tests, including the ground-truth golden regression
```

`stem/` splits into a GPU half (`frame_extraction`, `presence_detection`) and a
pure-arithmetic half (`matrices`, `metrics`, `failure_taxonomy`). The second
never imports the first, which is what lets you swap in your own detector.

```bash
python -m unittest discover -s tests
```

## Reproducibility notes

- The pipeline is deterministic: re-running the same videos, on a different
  GPU, reproduces its own output bit for bit.
- Detector settings are pinned in [`configs/qwen2_vl_7b.yaml`](configs/qwen2_vl_7b.yaml).
  Changing the model, the query wording or the frame sampling changes the
  benchmark — report it if you do.
- PTM and EPA inherit the detector's error rate. Read them as **comparative
  diagnostics** across systems scored by the same detector, not as absolute
  quality scores. `aggregate_results.py` always prints the detector's own
  TPR/TNR against `S*` alongside the scores.
- A verified dependency combination is recorded at the bottom of
  [`requirements-detector.txt`](requirements-detector.txt).

## Citation

```bibtex
@inproceedings{span_stem_2026,
  title     = {The Algebra of Storytelling: A State Transition Benchmark
               for Spanning Multi-Entity Multi-Shot Narratives},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
```

## License

MIT for the code; see [LICENSE](LICENSE). MEMS-1K prompts and annotations are
released under CC BY 4.0.
