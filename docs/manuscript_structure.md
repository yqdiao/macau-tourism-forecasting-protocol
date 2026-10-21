# Manuscript structure and evidence map

The current manuscript uses six sections. The directory names `track1` and `track2` remain engineering identifiers, not the paper's narrative structure.

## Central question

Under different demand periods, accuracy objectives, and resource-price conditions, which forecasting tools are worth adopting, and when do additional combination, language judgment, or fallback components justify their resources?

## Three research questions

- **RQ1 — Tool choice:** Which configurations offer useful accuracy–resource combinations under explicitly defined tasks, information sets, and accounting boundaries?
- **RQ2 — Applicability:** How do these comparisons change across stress and reopening periods, markets, horizons, and available history?
- **RQ3 — Additional components:** Do forecast combination, LLM collaboration, and fallback mechanisms deliver incremental accuracy or delivery gains commensurate with additional resources?

## Six sections

| Section | Purpose | Evidence and writing responsibility |
|---|---|---|
| 1. Introduction | Establish the adoption problem | Tourism decision needs; two reference studies; research questions; strongest findings |
| 2. Literature review and research positioning | Connect existing evidence to the study | Demand conditions; pretrained/local learning; combination/collaboration; adoption framework |
| 3. Methodology | Make comparisons interpretable | Data; model permissions; rolling selection/evaluation; component controls; metrics and resources |
| 4. Empirical results | Answer the questions | Accuracy; demand/history conditions; incremental collaboration; delivery; joint resource evaluation |
| 5. Discussion | Explain the contribution and application | Dialogue with both reference studies; conditional adoption; scope and future work |
| 6. Conclusion | State the supported takeaways | Accuracy–resource findings, applicability, and adoption sequence |

### Empirical-results subsections

1. Predictive accuracy and trade-offs between error objectives.
2. Applicability across demand periods and historical inputs.
3. Incremental performance of combination and language collaboration.
4. Output reliability and fallback mechanisms.
5. Computational resources and price-dependent adoption choices.

## Reference studies and comparison boundaries

[Wu et al. (2025), *How well can ChatGPT forecast tourism demand?*](https://doi.org/10.1016/j.tourman.2024.105119) provides an ability/use/context comparison. It uses monthly Macau data and distinguishes pre- and post-pandemic evaluation. The present study uses daily tasks, a reopening evaluation, and selected pandemic stress windows; these labels are not interchangeable.

[Wu et al. (2027), *Multi-agent collaboration strategies for tourism demand forecasting*](https://doi.org/10.1016/j.tourman.2026.105479) compares direct LLM forecasting under collaboration structures and model compositions. The present Qwen3-8B experiments mainly assign weights to existing numerical forecasts. Homogeneous role collaboration does not test heterogeneous-LLM benefits. Different output permissions and tasks preclude interpreting the present results as a direct refutation or full reproduction of that paper.

Both studies discuss practical value. The present contribution operationalizes recorded resource inputs and conditional adoption comparisons; it does not claim that earlier work never considered cost.

## Experiment-to-question mapping

| Experiment family | Main-text location | Appendix/reproduction detail |
|---|---|---|
| Original single-model evaluation | Results 4.1 and 4.5 | Full markets, metrics, settings |
| Stress and history analyses | Results 4.2 | Windows, scale definitions, coverage |
| Original and revised lightweight configurations | Representative results in 4.1 | Complete V1–V3 revisions and exploratory selection identity |
| Simple combinations | Results 4.3 | Weights and candidate availability |
| Qwen3-8B M1–M2 | Results 4.3 and 4.5 | Strategies, repeats, caches, calls |
| Qwen3-8B M3–M4 | Contextual extension in 4.2/4.3 | Coverage and common-scale analyses |
| Qwen3-4B-thinking M5 | Results 4.1 and 4.4 | Extreme values and shared-sample sensitivity |
| Original fault/coordination experiments | Results 4.4 and 4.5 | Fault types, fallback paths, and logs |

The original NL01 selection remains the frozen choice. V1–V3 revisions were designed after previous outer results had been inspected; their results remain exploratory. M2 has a fully linked same-candidate equal-weight comparison; that evidence is not automatically a chain-versus-equal-weight comparison for M1.

## Main text versus appendix

The main text retains evidence needed for adoption: common design, accuracy, representative markets, demand/history effects, component increments, reliability, and resource conditions. The appendix preserves full coverage, configurations, repeated runs, version selection, metric definitions, resource boundaries, and the repository reference.

Unfavorable or unstable findings that affect adoption remain summarized in the main text. Appendix placement does not remove their scientific relevance.
