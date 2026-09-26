# WeatherCommodityLab v0.4.2 audit

## Research freeze

- v0.4.2 is a publication / GitHub release, not a new broad model-search iteration.
- The frozen article interpretation is recorded in `docs/FROZEN_FINDINGS.md`.
- Negative and inconclusive results remain part of the evidence base.

## Point-in-time safeguards

- Reanalysis is used only as ex-post realized-weather truth / physical data.
- Historical market experiments use point-in-time deterministic forecast vintages.
- Geography is aggregated with prior-year USDA production weights when available.
- Supply tightness uses lagged production only.
- Price-derived market controls are lagged by one observation.
- 2026 is excluded from finalized realized-yield research; `max_finalized_crop_year=2025` is enforced before baseline holdout construction.
- Genuine ensemble spread is prospective only.

## v0.4.1 hardening retained

- Empty-feature market folds are explicitly `NOT_EVALUABLE`.
- Feature availability is determined using training data only.
- Missing threshold forecasts remain missing rather than being coerced to zero events.
- Weather aliases are resolved explicitly across providers.
- Maximum-horizon revisions require an older valid forecast vintage.

## v0.4.2 publication hardening

- Crop-stage maximum-temperature aggregation uses the resolved temperature alias rather than one provider-specific field.
- Zero-coverage stage variables are excluded from article-facing tables automatically.
- Controls-only versus controls+weather market models are evaluated on identical OOS dates.
- Incremental weather value versus market controls receives block-bootstrap inference.
- Volatility results are labelled exploratory unless dedicated inference exists.
- Stale winner-style final-model configuration was removed; the release reports paired comparisons rather than asserting a champion.
- GitHub CI runs the unit / research-contract suite on Python 3.11, 3.12 and 3.13.
