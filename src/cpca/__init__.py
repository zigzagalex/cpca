# src/cpca/__init__.py
from ._cpca import pca as _pca

__all__ = ["pca"]
__version__ = "0.1.0"

def pca(*args, **kwargs):
    """Public alias with docstring control."""
    return _pca(*args, **kwargs)