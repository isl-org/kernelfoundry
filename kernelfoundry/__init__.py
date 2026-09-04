"""KernelFoundry - Python package for GPU kernel generation and evaluation."""

from pathlib import Path

from .test_base import TestBase

__version__ = "0.2.0"

#: The installed (or checked-out) ``kernelfoundry`` package directory.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: Hydra config tree. Ships as package data, so this exists in a wheel too.
CONFIG_DIR = PACKAGE_ROOT / "configs"

__all__ = ["TestBase", "__version__", "PACKAGE_ROOT", "CONFIG_DIR"]
