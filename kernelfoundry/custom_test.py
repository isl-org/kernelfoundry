"""Compatibility shim for the old `kernelfoundry.custom_test` API.

The canonical base class is now `kernelfoundry.test_base.TestBase`.
"""

from .test_base import TestBase

__all__ = ["CustomTest", "TestBase"]


class CustomTest(TestBase):
    """Deprecated alias of `kernelfoundry.test_base.TestBase`."""
