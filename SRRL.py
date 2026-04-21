"""Compatibility shim; canonical code is ``rl_pairs_trading.srrl`` (pip install -e .)."""
import runpy

if __name__ == "__main__":
    runpy.run_module("rl_pairs_trading.srrl", run_name="__main__")
