"""Helpers for loading datasets before validating them with mlwp-data-specs."""

from importlib.metadata import version

from .api import load_and_validate_dataset

__all__ = ["__version__", "load_and_validate_dataset"]
__version__ = version("mlwp-data-loaders")
