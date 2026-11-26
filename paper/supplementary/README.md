# Supplementary long tables

[Download the Excel workbook](supplementary-long-tables.xlsx). The workbook contains **9 data sheets and 924 data rows**, plus a README index. These are reported manuscript results, not raw visitor records or newly recalculated estimates. CSV previews below preserve the underlying cell values; Excel additionally preserves formatting and reported decimal precision.

| Worksheet / CSV preview | Contents | Rows |
|---|---|---:|
| [`stress_phase_market`](csv/stress_phase_market.csv) | Error differences by stress subperiod and market group | 126 |
| [`interval_coverage`](csv/interval_coverage.csv) | Coverage and scores of central prediction intervals | 50 |
| [`holiday_risk`](csv/holiday_risk.csv) | Underprediction risk on holidays and non-holidays | 112 |
| [`forecast_week`](csv/forecast_week.csv) | Error ratios by forecast week | 70 |
| [`market_wins`](csv/market_wins.csv) | Model win frequencies in major source markets | 40 |
| [`market_size`](csv/market_size.csv) | Forecast errors by market size | 42 |
| [`test_conditions`](csv/test_conditions.csv) | Error ratios relative to ETS and confidence intervals across test conditions | 236 |
| [`paired_intervals`](csv/paired_intervals.csv) | Intervals for paired differences and error ratios with origin-level tests | 176 |
| [`G1_paired_intervals`](csv/G1_paired_intervals.csv) | Market-equal paired intervals relative to statistical baseline G1 and origin-level tests | 72 |

## How to cite and navigate

The manuscript refers to these files by worksheet name. Historical S-numbered builder outputs elsewhere in the repository are provenance identifiers, not current manuscript table numbers. Use this index for the Excel supplement and cite the exact repository commit.

## Source and scope

Exported from the manuscript appendix on 26 November 2026. Model results, confidence intervals and p-values were copied without re-estimation. Original data access and end-to-end reproduction boundaries remain documented in [reproducibility status](../../docs/reproducibility_status.md).
