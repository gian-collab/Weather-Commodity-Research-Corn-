.PHONY: setup test smoke research publication bundle clean

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -e '.[dev,market,weather]'

test:
	.venv/bin/python -m pytest -q

smoke:
	.venv/bin/python scripts/run_smoke.py

research:
	.venv/bin/python scripts/run_article_pipeline.py

publication:
	.venv/bin/python scripts/build_publication_outputs.py

bundle:
	.venv/bin/python scripts/export_research_bundle.py

clean:
	rm -rf .pytest_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
