"""Deprecated compatibility entry point.

v0.4.2 intentionally freezes the research without selecting a single yield-model winner.
Use `scripts/run_article_pipeline.py` for the frozen evidence pipeline.
"""
raise SystemExit(
    'v0.4.2 has no one-shot winner model by design. '
    'Run `python scripts/run_article_pipeline.py` and report the paired OOS comparisons instead.'
)
