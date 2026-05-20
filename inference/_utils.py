"""Shared internal helpers for the inference package.

Module-private (leading underscore). Public API does not include these.
"""
import time


def _progress(msg):
    """Print a timestamped progress message and flush."""
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)
