# WeatherCommodityLab v0.4.2

Research-grade Python platform for studying how weather forecast uncertainty propagates through crop fundamentals into commodity markets, and what that process can teach quantitative finance about probabilistic forecasting.

The repository is built around a deliberately separated chain:

```text
weather state -> forecast distribution / revisions -> crop outcome -> market response
```

The project is **not** designed to manufacture a profitable corn strategy. It tests where predictability survives and where it disappears.

## Research questions

1. How quickly does deterministic weather forecast skill deteriorate with forecast horizon?
2. Are forecast revisions informative about subsequent forecast error?
3. Do stage-aware temperature / extreme-weather variables improve corn-yield prediction beyond state and technology trend?
4. Do point-in-time weather forecast revisions predict corn returns or realized volatility?
5. Does weather add incremental information beyond simple market-state controls?
6. How should genuine ensemble uncertainty be studied prospectively?

## Release status

`v0.4.2` is the **publication / GitHub freeze**. Broad model searching is closed. This version adds publication hygiene and one pre-specified robustness test rather than another model zoo.

See:

- [`docs/FROZEN_FINDINGS.md`](docs/FROZEN_FINDINGS.md) — frozen research interpretation
- [`RESEARCH_GOVERNANCE.md`](RESEARCH_GOVERNANCE.md) — point-in-time and evidence rules
- [`AUDIT.md`](AUDIT.md) — implementation audit
- [`CHANGELOG.md`](CHANGELOG.md) — release history
- [`ARTICLE_BLUEPRINT.md`](ARTICLE_BLUEPRINT.md) — article research structure

## Data sources

The public-data pipeline supports:

- **Open-Meteo / ECMWF previous runs** for point-in-time deterministic forecast vintages;
- **ERA5-Land via Open-Meteo archive** as an ex-post realized-weather proxy;
- **USDA NASS Quick Stats** for corn yield and production;
- **public continuous corn futures proxy data** for market experiments;
- **ECMWF AIFS-ENS open data** for prospective ensemble snapshot archiving.

Important limitations are explicit:

- Reanalysis is never substituted for a historical forecast vintage.
- Continuous futures data may contain roll artifacts and are not official CME settlement history.
- `2026` is not treated as a finalized realized corn-yield year.
- Deterministic run revisions are never labelled as ensemble spread.
- Zero-coverage variables are excluded from publication-facing claims.

## Repository layout

```text
configs/                 frozen research configuration
scripts/                 reproducible command-line entry points
src/weatherlab/data/     data adapters and storage
src/weatherlab/features/ weather and crop-stage feature engineering
src/weatherlab/models/   compact benchmark models
src/weatherlab/experiments/
                         weather, yield, market and inference studies
src/weatherlab/reporting/
                         BSIC-style figures and publication outputs
tests/                   regression and research-contract tests
docs/                    frozen findings / release documentation
data/                    local datalake (gitignored)
outputs/                 generated results (gitignored)
```

## Installation

Python 3.11+ is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,market,weather]"
python -m pytest -q
```

For a lightweight CI-style installation without market/weather optional dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Reusing the existing datalake

If you already ran v0.4.1, copy its raw datalake rather than downloading again:

```bash
mkdir -p data
cp -R ../WeatherCommodityLab_v0_4_1/data/raw data/
```

## Run the complete research pipeline

```bash
python scripts/run_article_pipeline.py
```

The pipeline runs the frozen baseline and mechanism-aware studies, then adds the v0.4.2 incremental-control test and builds conservative publication outputs.

Key raw result files include:

```text
outputs/results/weather_forecast_skill_by_lead.csv
outputs/results/stage_weather_feature_coverage.csv
outputs/results/yield_mechanism_metrics.csv
outputs/results/yield_mechanism_inference.csv
outputs/results/market_mechanism_grid.csv
outputs/results/market_mechanism_inference.csv
outputs/results/market_controls_grid.csv
outputs/results/market_controls_inference.csv
outputs/results/market_revision_vol_grid.csv
outputs/results/v0_4_summary.json
```

Article-facing outputs are generated separately under:

```text
outputs/publication/tables/
outputs/publication/figures/
outputs/publication/publication_manifest.json
```

That publication layer deliberately excludes zero-coverage features and labels unsupported exploratory evidence accordingly.

## Market-control robustness test

The v0.4.2 robustness experiment compares two models on **identical OOS dates**:

```text
controls only
= lagged return
+ lagged 5-day realized volatility
+ seasonal sine/cosine terms
+ lagged supply-tightness proxy (when available)
```

against:

```text
controls + point-in-time weather forecast revisions
```

The reported quantity is incremental OOS fit of weather versus the controls-only model, with block-bootstrap inference on the OOS squared-error improvement.

## Build publication outputs only

If the research result CSVs already exist:

```bash
python scripts/build_publication_outputs.py
```

## Export the article evidence bundle

```bash
python scripts/export_research_bundle.py
```

This writes:

```text
outputs/weather_article_research_v0_4_2.zip
```

## Prospective ensemble archive

The historical article does **not** pretend deterministic forecast revisions are ensemble uncertainty. To build genuine spread-skill evidence prospectively:

```bash
python scripts/archive_ensemble_now.py
```

Snapshots are written immutably under `data/snapshots/ecmwf/`.

## Reproducibility and research governance

Core rules:

- chronological / walk-forward evaluation only;
- train-only feature availability and imputation;
- identical paired state-years when comparing yield models;
- lagged production weights for market aggregation;
- no contemporaneous final production used as a market feature;
- fixed growing-season market windows;
- explicit `NOT_EVALUABLE` states rather than fabricated zero results;
- negative and null findings are preserved;
- publication tables are generated from frozen outputs rather than hand-selected winners.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The associated BSIC article should be cited once published.
