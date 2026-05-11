"""Compatibility shim for the old `kernelfoundry.custom_test` API.

The canonical base class is `TestBase` (``from kernelfoundry import TestBase``).
"""

from .test_base import TestBase

__all__ = ["CustomTest", "TestBase"]


class CustomTest(TestBase):
    """Deprecated alias of `TestBase` (`from kernelfoundry import TestBase`)."""
