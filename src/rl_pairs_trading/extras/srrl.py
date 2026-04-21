"""SRRL moved to extras.

The essential pipeline does not depend on SRRL.
"""

SRRL_HPARAMS = {}
SRRL_MODEL_DIR = "models/srrl"


class SRRLTrader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("SRRL is quarantined to extras and not enabled in the essential pipeline.")

