#!/bin/bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider --show-capture=all --disable-warnings -v
