# Recorded R3 configuration and selection evidence

These five JSON files are unchanged English-language records supplied with the V1–V3 outputs:

- `protocol_P0.json`: exploratory design, candidate sets, task counts, seeds, selection rule and resource settings;
- `selection_V2_B.json`: tree/linear choices from the B evaluation;
- `selection_V3_B.json`: local/global/mixing choices from the B evaluation;
- `frozen_R3.json`: recorded freeze and selected configurations;
- `sampling_policy.json`: shared legal-anchor sampling policy.

`source_manifest.json` records both original and released hashes. They are identical because these files required no translation or path edits.

The records identify code hash `c2bda610dcdc739a08c45b499d969d738142056d900e1bdc8d3cc50ac55a151e`. The implementation that generated that hash was **not present in the supplied R3 output folder**. These configuration snapshots do not replace that missing source. The original lightweight code published under `extensions/lightweight_legacy/` belongs to the earlier NL01/GL03/LL01 runs and must not be labelled as R3 code.

The protocol explicitly identifies the analysis as exploratory after existing outer results. A B-group freeze within that later exploration does not erase the prior exposure to the earlier outer results. The original NL01 champion is retained.

The companion task scores and summaries are in `paper/data/r3_task_metrics.csv` and `paper/data/r3_summary.csv`. Their aggregation can be independently reconstructed with `paper/scripts/rebuild_tables.py`.
