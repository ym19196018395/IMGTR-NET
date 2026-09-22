"""ENRICH-Aerial training entry point.

The project keeps one shared training loop in ``train_whu.py``.  This launcher
only supplies ENRICH-specific defaults so fixes to the model/loss do not drift
between two copied training files.  Every default can still be overridden from
the command line.
"""

import runpy
import sys
from pathlib import Path


def _has_option(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


defaults = {
    "--dataset": "enrich_aerial",
    "--nviews": "3",
    "--batch_size": "1",
    "--num_workers": "2",
}

injected = []
for option, value in defaults.items():
    if not _has_option(option):
        injected.extend([option, value])

sys.argv[1:1] = injected
runpy.run_path(str(Path(__file__).with_name("train_whu.py")), run_name="__main__")
