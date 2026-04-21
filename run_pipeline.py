"""
Goal: Provide a thin root entrypoint that launches the package pipeline runner.

Inputs: CLI flags passed by the user; imports pipeline main from rl_pairs_trading.pipeline.

Processing: Delegates execution directly to the package main() function.

Outputs: Pipeline process exit status to the shell; downstream artifacts are produced by called modules.
"""
from rl_pairs_trading.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
