"""Compatibility shim; canonical code is ``rl_pairs_trading.traditional``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("rl_pairs_trading.traditional", run_name="__main__")
