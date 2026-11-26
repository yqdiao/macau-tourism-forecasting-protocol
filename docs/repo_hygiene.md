# Repository hygiene

What stays, what is archival, and what was removed in the 2026-11-26 English / RQ sync.

## Keep (manuscript path)

| Path | Role |
|---|---|
| `RQ1/`, `RQ1_outputs/` | Section 4.1 builders and sealed tables |
| `RQ2/`, `RQ2_outputs/`, `makeup-RQ2/` | Section 4.2 panels; makeup is the final figure/holiday/WIS package |
| `RQ3/`, `RQ3_outputs/` | Section 4.3; cite DeepSeek caches only |
| `shared/`, `track1/`, `track2/` | Protocol, training/inference code, frozen configs |
| `paper/`, `docs/` | Audited companion materials and English narrative |
| `track1/docs/MODEL_TRAINING_AND_TUNING.md` | Trainable / tunable model freeze map |

## Archival but not deleted

| Path | Note |
|---|---|
| `extensions/lightweight_legacy/` | Older packaging of lightweight code; prefer `track1/code/light_models.py` |
| `track1/docs/UNIFIED_EXPERIMENT_PROTOCOL.md` and related GPU guides | Historical design records; pending checklists are not completion claims |
| `track1/artifacts/` | Packaged prediction archives already in Git (not the ignored `results/` tree) |

## Removed as redundant (2026-11-26)

Chinese working dumps that duplicated English protocols or sealed RQ summaries:

- Historical Track 1 working dumps and manuscript-index notes
- Historical Track 2 execution manuals and working-output notes
- Chinese RQ2/RQ3/makeup verification filenames (replaced by English `*_verification_summary.md`)

Local-only archive outside Git (may be deleted after you confirm): `/root/DYQ/_macau_unused_archive/2026-11-26` (old track copies, weights, B-search dumps).

## Intentionally non-English tokens

Holiday calendar filters keep Chinese source labels (compensatory workdays, replacement holidays, rest days, and extended holidays) because they match mainland calendar row names in the data.
