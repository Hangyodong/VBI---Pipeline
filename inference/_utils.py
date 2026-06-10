"""Shared internal helpers for the inference package.

Module-private (leading underscore). Public API does not include these.
"""
import time

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # tqdm not installed
    _tqdm = None


def _emit(msg):
    """Write a line without clobbering an active tqdm bar."""
    if _tqdm is not None:
        _tqdm.write(msg)
    else:
        print(msg, flush=True)


def _progress(msg):
    """Print a timestamped progress message (tqdm-safe)."""
    ts = time.strftime("%H:%M:%S")
    _emit(f"  [{ts}] {msg}")
