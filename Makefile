.PHONY: lint test test-gpu fmt

lint:
	uv run ruff check . && uv run ruff format --check .

# -n auto here for the same reason CI has it (#51): tests/conftest.py pins each
# worker to one torch thread, without which `-n auto` is slower than serial.
test:
	uv run pytest --tb=short -q -n auto

# Serial, and it must stay that way: the GPU suite shares `resource_group: gpu`
# with the train jobs and asserts against a measured 300s budget.
test-gpu:
	uv run pytest --tb=short -q -m gpu

fmt:
	uv run ruff format .
