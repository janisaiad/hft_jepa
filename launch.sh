#!/bin/bash
# we install uv, create venv, install llmcode and run a quick sanity check
pip install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .
uv run tests/test_env.py