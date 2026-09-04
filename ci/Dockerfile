# syntax=docker/dockerfile:1

# One CI image for lint, the CPU tests and the GPU tests.
#
# The venv is the lock, exactly: the same torch build the developers run (the
# CUDA 12.8 wheels from PyPI), no CPU swap, no CUDA base image. A pip-installed
# torch carries every CUDA library it needs as nvidia-*-cu12 wheels and takes
# the driver from the host through the container toolkit, so a plain Debian
# Python image runs CUDA on the workstation runner exactly as the 2 GB
# nvidia/cuda base did -- measured on the RTX 5070 Ti before this was written.
#
# What the previous pair of images got wrong, and why a GPU job cost 20 minutes:
# their builder and runtime stages installed Python at different paths, so the
# copied venv pointed at an interpreter that did not exist, and uv threw it away
# at job time and re-downloaded ~3 GB of wheels -- 19 of the job's 20 minutes.
# The CPU image swapped torch for a CPU build that the job's `uv sync` swapped
# straight back. Here both stages share one base, so the venv is valid on
# arrival and the job's `uv sync --locked` is an audit that takes milliseconds.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy rather than hardlink: the wheel cache below is a build mount that is
# gone at runtime.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-system

COPY pyproject.toml uv.lock README.md ./
RUN mkdir -p tame && touch tame/__init__.py

# The cache mount keeps downloaded wheels across builds on the same builder, so
# a lock change re-downloads only what changed. --locked refuses a lock that is
# out of date with pyproject.toml rather than silently re-resolving it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra dev --no-install-project

# ── Runtime: the same base, so the venv's interpreter path resolves ─────────
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-system \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=builder /app/.venv .venv
COPY pyproject.toml uv.lock README.md ./
