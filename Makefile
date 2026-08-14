.PHONY: install verify lint test run migrate seed clean

install:
	uv venv --python 3.12 .venv
	uv pip install -e '.[dev,api]'

lint:
	.venv/bin/python -m ruff check callytics tests

test:
	.venv/bin/python -m pytest tests/ -q

verify: lint test

run:
	.venv/bin/uvicorn callytics.api.app:app --host 127.0.0.1 --port 8080 --reload

migrate:
	.venv/bin/alembic upgrade head

seed:
	.venv/bin/python -m callytics seed

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
