# MEMS-1K

**M**ulti-**E**ntity **M**ulti-**S**hot **1K**: 1,000 three-shot narrative
scenarios across 7 patterns, each prescribing exactly which entities must be
visible in each shot.

```
mems1k.json          all 1,000 scenarios + the pattern table
splits/*.json        the same scenarios, one file per pattern
schema.json          JSON Schema for both (field-by-field contract)
```

```bash
python scripts/validate_mems1k.py
```

## What a scenario looks like

```json
{
  "scenario_id": "DM1K_913_Convergence_079",
  "pattern": "Convergence",
  "pattern_class": "Derived",
  "notation": "A→B→AB",
  "theme": "Nature_Forest",
  "num_shots": 3,
  "num_entities": 2,
  "entities": [
    {"entity_id": "A", "name": "rabbit", "eval_name": "rabbit", "entity_type": "A"},
    {"entity_id": "B", "name": "fox",    "eval_name": "fox",    "entity_type": "A"}
  ],
  "shot_structure": ["A", "B", "AB"],
  "presence_matrix":   [[1, 0], [0, 1], [1, 1]],
  "transition_matrix": [[-1, 1], [1, 0]],
  "gen_prompt":  {"shot_1": "A rabbit rests in a dense pine forest., masterpiece, ..."},
  "eval_prompt": {"shot_1": "A rabbit rests in a dense pine forest."},
  "diag_prompt": {"shot_1": "a rabbit"},
  "generation_settings": {"camera": "...", "positive_suffix": "...", "negative_prompt": "..."},
  "metadata": {"basis_type": "derived", "num_active_transitions": 3}
}
```

## The two matrices

`presence_matrix` is **S\***, shape `T × N`, binary.

- **Row `t`** is shot `t+1`. The top row is the first shot.
- **Column `j`** is `entities[j]`. Column order is always the `entity_id`
  order A, B, C — this is the single alignment convention everything depends
  on, and `validate_mems1k.py` enforces it.
- `S*[t][j] = 1` means entity `j` must be visible in shot `t+1`.

`transition_matrix` is **ΔS\* = S\*[1:] − S\*[:−1]**, shape `(T−1) × N`, values
in `{−1, 0, +1}`.

- **Row `t`** is the transition from shot `t+1` to shot `t+2`.
- `+1` prescribed entrance, `−1` prescribed exit, `0` no change prescribed.

**PTM is scored only on the non-zero cells of ΔS\*.** The zero cells still
matter — they feed EPA and produce Type III (spurious) failures — but they
never enter the PTM fraction. `metadata.num_active_transitions` is that
denominator: 2 or 3 in MEMS-1K, so per-scenario PTM lands in `{0, ½, 1}` or
`{0, ⅓, ⅔, 1}`.

`shot_structure` is a human-readable rendering of the same information
(`["A", "B", "AB"]`); it is derived, not authoritative.

## The three prompt fields

| field | used by | content |
|---|---|---|
| `gen_prompt` | the video generator | full prompt including the shared style suffix |
| `eval_prompt` | humans, text-conditioned baselines | `gen_prompt` without the style suffix |
| `diag_prompt` | the CLIP-based baseline metric | short entity-only phrase, e.g. `"a rabbit and a fox together"` |

**None of these is the detector query.** STEM asks about entities, not
sentences:

> `Is there a {entities[j].eval_name} visible in this image? Answer only Yes or No.`

`name` and `eval_name` are identical in v1.0. They are kept as separate fields
so a detector can use different wording without perturbing the generation
prompts.

## Patterns

| pattern | class | structure | entities | prescribed cells | scenarios |
|---|---|---|---|---|---|
| Relay | Basis | A → AB → B | 2 | 2 | 200 |
| Split | Basis | AB → A → B | 2 | 3 | 200 |
| Accumulation | Derived | A → AB → ABC | 3 | 2 | 200 |
| Convergence | Derived | A → B → AB | 2 | 3 | 100 |
| Sliding_Window | Derived | AB → BC → C | 3 | 3 | 100 |
| Reduction | Derived | ABC → AB → A | 3 | 2 | 100 |
| Reverse_Relay | Derived | B → AB → A | 2 | 2 | 100 |

Relay and Split are the algebraic basis of Theorem 3.1; the canonical matrices
live in [`span/pattern_library.py`](../span/pattern_library.py), and the
benchmark file is checked against them on every load.

Scenarios are diversified over 10 themes (`Domestic_Kitchen`, `Urban_Library`,
`Nature_Forest`, …) and three entity categories: `H` generic person, `P`
professional/role-defined person, `A` animal.

## Videos

The 4,000 generated videos evaluated in the paper are **not** part of this
repository. It releases the benchmark and the evaluation code; generate videos
with any multi-shot system and score them with `scripts/evaluate_directory.py`.

Name each file `<scenario_id>.mp4` and concatenate the three shots in order,
equal length.

## Known issues

**`eval_prompt` was regenerated for this release.** In the internal prompt file
the `eval_prompt` field had never been re-permuted when the five derived
patterns were added, so for the 800 non-Relay scenarios it still held the
Relay-template wording and contradicted the scenario's own pattern.
`gen_prompt` — the field every generator actually read, and therefore the field
that produced the released videos — was correct throughout. The published
`eval_prompt` is derived as `gen_prompt` minus the shared style suffix.

This never touched any published number: STEM queries entity names, not shot
sentences, and PTM/EPA are computed from `S*`. Only the CLIP-based baseline
analysis ever consumed `eval_prompt`.

**Style suffix coverage.** 2,500 of 3,000 generation prompts carry the shared
style suffix. The remaining 500 are the shorter continuation shots of the
3-entity patterns, which use a different template; for those, `eval_prompt`
equals `gen_prompt`.

## License

CC BY 4.0.
