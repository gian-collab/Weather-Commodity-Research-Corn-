# BSIC Article Blueprint — WeatherCommodityLab v0.4.1

## Working title
**Forecasting Chaos: What Weather Prediction Can Teach Quantitative Finance**

## Core question
Can methods designed for a chaotic physical system — horizon-aware forecast verification, state estimation, revisions, ensembles and calibrated uncertainty — improve inference in a market whose fundamentals are physically exposed to weather?

## Why now
Machine-learning weather models and ensemble systems increasingly coexist with numerical weather prediction; climate/weather shocks remain economically material to agriculture; and financial ML often still emphasizes point estimates rather than calibrated distributions and state-dependent uncertainty.

## Evidence chain
1. **Forecast skill:** archived ECMWF fixed-lead forecasts vs ERA5-Land truth proxy by horizon and variable.
2. **Forecast difficulty:** run-to-run revision instability vs subsequent forecast error, clearly distinguished from ensemble spread.
3. **Physical mechanism:** state + technology trend baseline vs stage-specific weather means, extremes, hot/dry interactions and water balance.
4. **Market mechanism:** production-weighted point-in-time forecast revisions during the growing season, tested at 1/3/5/10-day return horizons and 3/5/10-day realized-volatility horizons.
5. **State dependence:** revision features interacted with a lagged public supply-tightness proxy. No contemporaneous final production is used.
6. **Distributional extension:** prospectively archive ECMWF AIFS-ENS. Ensemble-spread claims are made only after genuine ensemble snapshots exist.

## Pre-specified interpretation
A negative market result does not invalidate weather forecasting. It can mean physical information exists but is already priced, the free continuous futures proxy is too noisy, or the historical deterministic forecast archive is too short. Results are reported separately at the forecast, physical, market and execution levels.


## v0.4.2 publication freeze

The article should distinguish five layers: physical state estimation, probabilistic weather forecasting, crop-fundamental mapping, market information aggregation, and tradeability. Corn is the bridge rather than the sole forecasting objective. Publication-facing tables are generated from `outputs/publication/`; zero-coverage variables are excluded and realized-volatility evidence remains exploratory unless dedicated inference is added. The final robustness test asks whether weather revisions add OOS information beyond lagged return, lagged realized volatility, seasonality and lagged supply tightness.
