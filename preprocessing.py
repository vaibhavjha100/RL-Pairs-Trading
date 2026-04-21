"""Compatibility shim; canonical code is ``rl_pairs_trading.preprocessing``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("rl_pairs_trading.preprocessing", run_name="__main__")
