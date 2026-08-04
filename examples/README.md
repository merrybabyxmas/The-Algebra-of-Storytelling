# Examples

Sample input and output for the evaluation CLIs. All of it comes from the five
ground-truth cat videos in [`../demo/assets/gt_videos`](../demo/assets/gt_videos),
so the numbers are real detections, not invented ones.

```
predicted_presence.json   input:  scenario_id -> observed presence matrix
evaluation_result.json    output: one full record (Type_IV_Reversal)
batch_result.csv          output: the flat per-video summary table
```

## Reproduce them

```bash
python scripts/evaluate_predictions.py \
    --observed examples/predicted_presence.json \
    --benchmark demo/assets/gt_cases.json
```

`--benchmark` is needed here because these ids belong to the demo case set
rather than to MEMS-1K. For your own MEMS-1K predictions, omit it:

```bash
python scripts/evaluate_predictions.py --observed my_preds.json --method my_model
```

## Input format

```json
{
  "predictions": {
    "DM1K_777_Relay_001": [[1, 0], [1, 1], [1, 1]]
  }
}
```

A bare `{scenario_id: matrix}` object works too, as does a list of
`{"scenario_id": ..., "S_obs": ...}` records. Rows are shots in order; columns
follow the scenario's `entities` list, which is always `entity_id` order
(A, B, C).
