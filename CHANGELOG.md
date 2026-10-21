# Change log

## 2026.10.21 — Manuscript evidence and English documentation

- Reorganize the repository entry points around forecast-tool adoption, demand
  conditions, and the incremental value of additional components.
- Provide English documentation and notebook interfaces, including English file
  names for the historical protocols. Historical commit messages are unchanged.
- Preserve the historical baseline selection rules and distinguish archived
  execution plans from completed experimental evidence.
- Release audited model, market, task, coverage, and resource summaries under
  `paper/`, together with provenance, file hashes, and reconstruction scripts.
- Document MASE scale and coverage, exploratory lightweight revisions, complete
  and sensitivity samples for direct LLM forecasts, and same-candidate controls
  for language-based weighting.
- Express deployment costs through resource quantities and user-supplied prices.
  No common currency estimate is treated as a universal deployment cost.
- Retain the original protocol and data separation interfaces. English workbook
  documentation does not change the source-column names accepted by the reader.

The release does not assert that every later server experiment can be rerun from
this repository. Consult `docs/reproducibility_status.md` and `paper/README.md`
for the executable, configuration-only, and results-only components. Model
weights and third-party articles are not included. No new license is granted by
this documentation update; upstream data and model terms continue to apply.
