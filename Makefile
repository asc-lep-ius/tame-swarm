.PHONY: lint test test-gpu fmt

lint:
	uv run ruff check . && uv run ruff format --check .

test:
	uv run pytest --tb=short -q

test-gpu:
	uv run pytest --tb=short -q -m gpu

fmt:
	uv run ruff format .
