"""Compatibility shim; canonical code is ``rl_pairs_trading.extras.srrl_tuning``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("rl_pairs_trading.extras.srrl_tuning", run_name="__main__")
