"""Thin entry: run from repo root after ``pip install -e .``."""
from rl_pairs_trading.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
