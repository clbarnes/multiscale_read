"""
# multiscale_read package
"""
from .ngl_n5 import NglN5Multiscale
from .ome import OmeMultiscale
from importlib.metadata import version as _version

__version__ = _version(__package__)

__all__ = ["NglN5Multiscale", "OmeMultiscale"]
