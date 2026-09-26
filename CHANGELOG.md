# Changelog

## v0.4.2 — Publication / GitHub release

This release freezes the broad research programme and adds publication hygiene rather than new model searching.

- Corrects baseline finalized-yield governance: years after 2025 are excluded before holdout construction.
- Fixes crop-stage maximum-temperature aggregation to use the resolved weather alias rather than one provider-specific column name.
- Adds a pre-specified incremental market-control test: controls-only versus controls + point-in-time weather revisions on identical OOS dates.
- Adds block-bootstrap inference for the incremental weather-vs-controls comparison.
- Adds `outputs/publication/` generation with conservative article-facing tables and BSIC-style figures.
- Automatically excludes zero-coverage crop-stage variables from publication tables.
- Labels weather-revision / realized-volatility results exploratory unless dedicated inference exists.
- Replaces stale winner-style `configs/final.yaml` with a research-freeze / publication policy.
- Adds GitHub Actions CI, citation metadata, release documentation and a frozen findings note.

No new broad model family is selected in v0.4.2.
