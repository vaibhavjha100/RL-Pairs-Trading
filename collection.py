"""Compatibility shim; canonical code is ``rl_pairs_trading.collection``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("rl_pairs_trading.collection", run_name="__main__")
