"""
Goal: Hold SRRL placeholders/quarantined logic outside the essential pipeline path.

Inputs: Imports or calls from extras-only SRRL experiments.

Processing: Defines SRRL metadata and guarded runtime behavior for non-essential usage.

Outputs: Extras-only SRRL symbols and controlled runtime errors when unsupported flows are invoked.
"""

SRRL_HPARAMS = {}
SRRL_MODEL_DIR = "models/srrl"


class SRRLTrader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("SRRL is quarantined to extras and not enabled in the essential pipeline.")
