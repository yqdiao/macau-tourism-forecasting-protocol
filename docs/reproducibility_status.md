# Reproducibility status and release boundaries

Updated for the 2026-10-21 paper-companion release. This document separates historical implementation, exported evidence, and rerun readiness.

## Availability layers

| Layer | Present evidence | What it supports |
|---|---|---|
| Historical data/task implementation | `shared/`, Track 1 pipeline, protocol calendar, origin sharding | Inspection and execution of the existing task construction and isolation design |
| Historical baseline selection | Frozen YAML and September audit artifacts | Verification of recorded candidate selection and completion hashes |
| Seven-model integration | Adapters, Dockerfiles, launchers, revision registry | Inspection of the registered integration; not automatic proof of GPU acceptance or final checkpoint identity |
| Track 2 historical design | English protocol | Inspection of proposed selection, combination, residual, and coordination rules |
| Current paper companion | `paper/` and its explicit inventory | Inspection/reproduction at the level supported by the released files |
| Full later server reruns | Require matching final code, versions, data, configurations, and logs | Not established merely by a protocol, summary table, or manuscript figure |

No completion status is inferred from a pending historical checklist. Conversely, an old pending status does not establish that a later server run never occurred. Use the actual released evidence for each experiment.

## Counts and temporal scope

- Primary Macau outer evaluation: 27 markets × 33 origins = **891 tasks**, each with a 28-day path.
- Macau B selection: 27 × 12 = **324 tasks**.
- Macau A stress analysis: 27 × 13 = **351 tasks** in selected windows.
- Hong Kong extension: two series, reported separately.
- Primary scored dates: 2024-01-01 through 2026-07-12; available observations extend to 2026-07-31.

Historical protocols also calculate workloads for 29 series, 20 planned configurations, three histories, or 916 daily origins. These are design workloads, not the current paper's completed common-sample model count. Forecast points and forecast tasks are different units. Tasks also share market and temporal dependence.

## Interpretation requirements

1. Preserve the original frozen selection and distinguish post-outer exploratory revisions.
2. Compare strategies on the actual shared task/candidate/history set.
3. Identify MASE's scale and undefined coverage; do not replace undefined MASE with zero.
4. Keep complete direct-generation results alongside labeled sensitivity analyses.
5. Separate weight judgment, direct numerical forecasting, and deterministic fallback.
6. Report recorded resource quantities and accounting boundaries before assigning prices.
7. Treat simulated faults as the tested fault construction, not a measured service-level agreement.
8. Frozen target-task weights do not prove the absence of upstream pretraining overlap.

## Cost scope

Use a common price vector `p` and cost function `C_j(p)`. Record the resource units and what is included. Device occupancy, per-call elapsed time, shared allocation estimates, and total deployment costs are not interchangeable. Prices depend on provider, region, hardware, billing model, and date. Missing labor, loading, maintenance, or implementation costs remain unmeasured, not zero.

Any improvement-per-cost ratio needs an identified comparator, the same error definition, and a defensible positive improvement. A paired interval crossing zero does not support a stable ranking by such a ratio. Operating profit and avoided service losses require additional decision data.

## Rights and archival integrity

This release retains existing tracked data and audit bytes, and translates current documentation to English. It does not rewrite historical Git commits or grant new rights to source data, third-party source, or model weights. Original historical files remain available in Git history. The repository should be cited with the exact commit used; no publication DOI or journal acceptance is implied.
