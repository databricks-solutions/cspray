#!/bin/bash
# PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider --show-capture=all --disable-warnings -v
DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider --show-capture=all --disable-warnings --import-mode=importlib -v "$DIR/tests"
