# Frozen research findings — v0.4.2

This document records the research interpretation frozen before article drafting. It is not a claim of a profitable trading strategy.

## Weather forecast skill

Archived ECMWF/Open-Meteo deterministic forecasts matched to ERA5-Land show a clear deterioration in temperature forecast accuracy with lead horizon. In the audited v0.4.1 evidence bundle, mean-temperature MAE rose from about 1.16°C at a one-day lead to 2.28°C at seven days; maximum-temperature MAE rose from about 0.98°C to 2.76°C.

Absolute deterministic forecast revisions were only weakly associated with absolute subsequent forecast error (correlation about 0.077), reinforcing the distinction between a changing point forecast and genuine ensemble uncertainty.

## Yield channel

A state + technology/time-trend baseline was difficult to improve robustly. The best nonlinear stage/extreme-weather model produced only a small paired absolute-error improvement, with the year-clustered confidence interval crossing zero. Linear weather augmentation generally did not help.

The audited v0.4.1 stage table had complete coverage for several temperature / extreme-day features but zero coverage for some precipitation and water-balance fields. v0.4.2 therefore excludes zero-coverage fields from publication-facing tables rather than implying they were tested successfully.

## Market channel

Across the pre-specified weather-revision return grid, directional corn-return predictability was weak. Only one audited specification had positive OOS R² versus a zero-return benchmark and its block-bootstrap interval crossed zero. Richer extreme / stage-weighted variants generally did not improve the result.

Weather-revision models showed more promising associations with subsequent realized volatility at some longer horizons, but those results remain exploratory until dedicated inference is added.

## Interpretation

The frozen article thesis is:

> Predictability of a physical fundamental does not automatically imply predictability of its market price.

The project uses weather as a forecasting laboratory, corn as a physical-to-financial bridge, and the negative / inconclusive market evidence as a lesson in separating forecastability, economic relevance and tradable alpha.
