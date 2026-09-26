from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from weatherlab.reporting.bsic import apply_bsic, export_figure


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_publication_outputs(results_dir: str | Path, publication_dir: str | Path) -> dict:
    """Create a conservative, article-facing subset of tables and figures.

    This layer intentionally excludes zero-coverage features and labels volatility evidence
    exploratory when no dedicated inference file exists.
    """
    results = Path(results_dir)
    pub = Path(publication_dir)
    tables = pub / 'tables'; figs = pub / 'figures'
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {'publication_policy': 'conservative_article_subset'}

    skill = _read(results / 'weather_forecast_skill_by_lead.csv')
    if not skill.empty:
        skill.to_csv(tables / 'weather_forecast_skill.csv', index=False)
        s = skill[skill['variable'].isin(['temperature_mean', 'temperature_max'])].copy()
        if not s.empty:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
            for variable, g in s.groupby('variable'):
                ax.plot(g['lead_days'], g['mae'], marker='o', label=variable.replace('_', ' ').title())
            ax.set_xlabel('Forecast lead (days)'); ax.set_ylabel('MAE (degrees C)')
            ax.legend(frameon=False)
            apply_bsic(fig, ax, title='Weather forecast error rises with horizon', sources=('Open-Meteo / ECMWF previous runs', 'ERA5-Land', 'BSIC'))
            export_figure(fig, figs / 'weather_skill_decay')
        manifest['weather_skill_rows'] = len(skill)

    coverage = _read(results / 'stage_weather_feature_coverage.csv')
    if not coverage.empty:
        observed = coverage[coverage['n_nonmissing'] > 0].copy()
        excluded = coverage[coverage['n_nonmissing'] <= 0].copy()
        observed.to_csv(tables / 'stage_weather_observed_features.csv', index=False)
        excluded.to_csv(tables / 'stage_weather_excluded_zero_coverage.csv', index=False)
        manifest['stage_features_observed'] = int(len(observed))
        manifest['stage_features_zero_coverage_excluded'] = int(len(excluded))

    ymet = _read(results / 'yield_mechanism_metrics.csv')
    yinf = _read(results / 'yield_mechanism_inference.csv')
    if not ymet.empty:
        ymet.to_csv(tables / 'yield_model_performance.csv', index=False)
    if not yinf.empty:
        yinf.to_csv(tables / 'yield_paired_inference.csv', index=False)
        top = yinf.head(8).sort_values('year_clustered_estimate')
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        y = np.arange(len(top))
        est = top['year_clustered_estimate'].to_numpy(float)
        lo = top['ci_low'].to_numpy(float); hi = top['ci_high'].to_numpy(float)
        ax.errorbar(est, y, xerr=[est-lo, hi-est], fmt='o', capsize=3)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels(top['model'].str.replace('_', ' '))
        ax.set_xlabel('Absolute-error improvement vs state + time trend (bu/acre)')
        apply_bsic(fig, ax, title='Incremental yield value of weather is uncertain', sources=('USDA NASS', 'ERA5-Land', 'BSIC'))
        export_figure(fig, figs / 'yield_incremental_inference')

    mgrid = _read(results / 'market_mechanism_grid.csv')
    minf = _read(results / 'market_mechanism_inference.csv')
    if not mgrid.empty:
        mgrid.to_csv(tables / 'corn_return_model_grid.csv', index=False)
        avail = mgrid[mgrid.get('status', 'AVAILABLE').eq('AVAILABLE')].copy() if 'status' in mgrid.columns else mgrid.copy()
        manifest['return_specs_total'] = int(len(avail))
        manifest['return_specs_positive_oos_r2'] = int((pd.to_numeric(avail['oos_r2_vs_zero'], errors='coerce') > 0).sum()) if 'oos_r2_vs_zero' in avail else None
    if not minf.empty:
        minf.to_csv(tables / 'corn_return_inference.csv', index=False)

    controls = _read(results / 'market_controls_grid.csv')
    controls_inf = _read(results / 'market_controls_inference.csv')
    if not controls.empty:
        controls.to_csv(tables / 'weather_incremental_vs_market_controls.csv', index=False)
    if not controls_inf.empty:
        controls_inf.to_csv(tables / 'weather_incremental_vs_market_controls_inference.csv', index=False)

    vol = _read(results / 'market_revision_vol_grid.csv')
    if not vol.empty:
        v = vol.copy(); v['evidence_label'] = 'EXPLORATORY_NO_DEDICATED_BOOTSTRAP_INFERENCE'
        v.to_csv(tables / 'corn_volatility_exploratory.csv', index=False)

    summary_path = results / 'v0_4_summary.json'
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            manifest['research_summary'] = summary
        except Exception:
            pass
    (pub / 'publication_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return manifest
