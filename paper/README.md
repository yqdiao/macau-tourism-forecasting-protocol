# Paper evidence release

This directory accompanies **Are large models more cost-effective for tourism demand forecasting? Evidence from different demand periods in Macao**. It provides audited derived scores, comparison metadata, selected English figures, and a local aggregation check. The manuscript is organized around forecasting-tool choice, demand conditions, and the incremental value of additional components.

## What can be reproduced here?

The release supports independent reconstruction of the **reported summary calculations from the supplied derived scores**. It does not yet provide a complete end-to-end rerun of every server experiment. The task-level predictions, all execution environments, model checkpoints, and complete prompts/responses for the later Qwen and lightweight revisions are not bundled. The original NL01/GL03/LL01 implementation snapshot is available under [`extensions/lightweight_legacy/`](../extensions/lightweight_legacy/), with its dependency and missing-calendar boundaries documented. The existing `track1/`, `track2/`, and `shared/` directories retain the implementation/protocol scope documented in their own READMEs.

No GPU, API key, or additional Python package is needed for the summary checks:

```bash
python3 paper/scripts/rebuild_tables.py
```

The command verifies the release checksums, checks task coverage and aggregation, reconstructs main-model, R3, and LLM tables, and writes an English report under `paper/rebuilt/`. The output directory is excluded from Git. It does not fit models, regenerate forecasts, or revise the recorded model-selection decisions.

To render the joint accuracy-resource figure from its released plot data, use the optional Matplotlib script:

```bash
python3 paper/scripts/plot_accuracy_resources.py
```

This writes a PDF and PNG to `paper/rebuilt/`. The numerical inputs are the same as the archived figure; font and rendering details can vary by platform.

## Research questions and evidence

| Question | Main evidence | Relevant manuscript section |
|---|---|---|
| Which tools offer useful accuracy-resource combinations? | Main 15-configuration comparison; 27-market scores; direct 4B evaluation | 4.1 and 4.5 |
| How do demand periods and available history change the conclusion? | Stress-window coverage, common-scale history comparison, Hong Kong extension | 4.2 |
| Do additional components justify their resource requirements? | Equal weighting, Qwen weight decisions, strategy comparisons, delivery records | 4.3–4.5 |

Experiment identifiers are provenance labels, not manuscript chapter titles. The main Macao evaluation comprises **891 tasks**, each with a 28-day forecast path. A task is identified by its market and origin; tasks are not assumed to be independent observations.

## Included files

| File | Contents and scope |
|---|---|
| `data/main_model_market_metrics.csv` | 405 rows: 15 configurations × 27 markets, each averaged over 33 O origins, all-history setting |
| `data/main_accuracy_summary.csv` | Market-equal aggregate accuracy for the same 15 configurations |
| `data/market_totals_27.csv` | Descriptive total arrivals across the 27 included markets, 2017-01-01 to 2026-07-31 |
| `data/top10_model_market_metrics.csv` | 150 model-market rows selected by the descriptive full-period top-ten market totals |
| `data/r3_task_metrics.csv` | 20,142 scored task/configuration rows for the exploratory V1–V3 revisions; seed errors already averaged |
| `data/r3_summary.csv` | 42 model/split summaries reconstructed from those task scores |
| `data/r3_partial_runtime.csv` | Partial CPU timing records; these do not establish complete deployment costs |
| `data/llm_task_metrics.csv` | 15,507 task/strategy score rows, including reconstructed comparators; repeat errors already averaged |
| `data/llm_summary.csv` | 30 full-sample, zero-scale, positive-scale, and matched-sample summaries |
| `data/llm_paired_differences.csv` | Stored paired differences and circular-origin-block bootstrap intervals |
| `data/llm_delivery_coverage.csv` | Aggregate delivered/native/fallback coverage by strategy and replicate |
| `data/llm_call_resources.csv` | Recorded request-duration coefficients, with absolute currency assumptions removed |
| `data/accuracy_resource_plot_data.csv` | Plot-ready relative registered charges, errors, and incremental LLM workload |
| `figures/` | Selected English figure exports used by the manuscript |
| `configs/qwen8b_manifest.json` | English transcription of the supplied actual 8B configuration note, with original protocol and sampling hashes |
| `configs/r3/` | Actual R3 protocol, sampling and selection JSON snapshots; the corresponding R3 implementation was not supplied |
| `evidence_manifest.json` | File checksums, upstream audit-file checksums, transformations and release scope |

See [data_dictionary.md](data_dictionary.md) for definitions and [evidence_notes.md](evidence_notes.md) for interpretation boundaries.

## Scoring conventions

The three main accuracy measures are MASE(7), MAE and **mean task RMSE**. A task RMSE is calculated across its 28 forecast days before averaging over origins and markets. This differs from pooling every squared error and taking a single square root.

For stochastic experiments, each delivered forecast is scored before averaging repeated-run errors within a task. This is not the same as averaging forecasts before scoring. The source audits retain fallback outputs.

MASE is left undefined when its historical scale is zero. The pressure-window evaluation has 29 such tasks out of 351 under the 180-day scale. MAE and RMSE retain all 351 tasks. Zero-scale cases are not repaired with an arbitrary epsilon. WAPE also has a coverage boundary when the future actual total is zero.

The released LLM table explicitly names `mase_evaluation_history`: M2 uses the all-history scale, while M1, M3, M4, and direct 4B use the 180-day scale. `mase_all_reference` reports the common all-history reference where available. M2's LLM prompt still uses recent 180-day observations; its numeric candidate forecasts use all history. These fields should not be collapsed into one generic “history” label.

## Model roles and selection identity

- Frozen time-series foundation models generate numeric candidate forecasts.
- Qwen3-8B Q4_K_M, with thinking disabled, makes candidate-weight decisions. A program applies those weights to generate the delivered forecast.
- Qwen3-4B-thinking directly generates a numeric path. It is labelled M5 in the manuscript; that label does not imply an additional 8B experiment.
- The original NL01 remains the B-selected lightweight configuration. LL01 and the subsequent R3 V1–V3 revisions have a post-outer exploratory identity and cannot replace that original selection retrospectively.
- Separate models may have identical predictions. R3V2_T3, R3V3_L_INF and R3V3_MIX_1 coincide in the audited daily outputs; R3V3_G_INF and R3V3_MIX_0 also coincide. They do not constitute independent replications.

## Resource and price conventions

Absolute dollar figures are not used in this release. A local price vector `p` converts recorded resources to a conditional cost function `C_j(p)`. Platform, region, hardware, billing unit and price date can change that conversion.

For LLM calls, `additional_call_hours_per_1000_deliveries` equals the sum of recorded request durations divided by delivered forecasts, multiplied by 1000/3600. It excludes candidate production and unmeasured service overhead. It is not a provider invoice, GPU kernel time, or a universal claim about physical GPU occupancy.

The same-GPU relative registered-charge ratios for five TSFMs originate from rounded registration entries. They are useful for a conditional within-group comparison, not new independent timing measurements. CPU/GPU comparisons require compatible resource boundaries and explicitly chosen prices. Partial R3 runtime values and zero-valued inference log entries do not mean free deployment.

## Relation to the two principal references

- Wu et al. (2025), [How well can ChatGPT forecast tourism demand?](https://doi.org/10.1016/j.tourman.2024.105119), motivates capability, usage and demand-period comparisons.
- Wu et al. (2027), [Multi-agent collaboration strategies for tourism demand forecasting](https://doi.org/10.1016/j.tourman.2026.105479), motivates comparisons of collaboration structures and agent composition.

The latter study principally asks LLM agents to produce forecasts directly and includes heterogeneous LLM systems. The local 8B experiments here primarily evaluate weight decisions over existing numerical forecasts with one shared LLM backbone. Their results therefore concern a different use of collaboration. The implementation does not establish a general finding that multi-agent collaboration is ineffective.

## Remaining reproducibility scope

The audit summaries were derived from supplied experimental outputs. Their hashes identify the upstream audit files, not a substitute public archive of all raw server outputs. The available Qwen model identity and R3 configuration snapshots are included. A future complete release would also need the final runnable later-stage code, complete machine-generated configuration and environment manifests, model forecasts, request/failure logs with sensitive content removed, and complete resource accounting. The present release makes no claim that those missing components are already available.

No third-party research-paper PDFs, model weights, credentials, private collaboration links, or Chinese manuscript files are included in this directory. The selected figures are the study's own English exports. Reuse follows the repository's applicable rights and any separately stated upstream data/model terms; this release does not assign new rights to third-party data.
