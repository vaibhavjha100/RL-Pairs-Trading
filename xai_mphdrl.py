"""Compatibility shim; canonical code is ``rl_pairs_trading.extras.xai_mphdrl``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("rl_pairs_trading.extras.xai_mphdrl", run_name="__main__")
