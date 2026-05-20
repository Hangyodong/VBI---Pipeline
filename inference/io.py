"""Pickle persistence for pipeline artifacts.

Public API
----------
- save_artifacts(path, **kwargs)
- load_artifacts(path) -> dict
"""
import os
import pickle


def save_artifacts(path, **kwargs):
    """Pickle scalers, pipelines, and metadata to disk."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(kwargs, f)


def load_artifacts(path):
    """Load a previously saved artifacts dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"artifacts file not found: {path!r}. "
            "Run the pipeline first (Step 14 in main.ipynb) or pass "
            "an explicit path to load_artifacts(...)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
