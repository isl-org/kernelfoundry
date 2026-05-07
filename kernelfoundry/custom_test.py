"""Compatibility shim for the old `kernelfoundry.custom_test` API.

The canonical base class is now `kernelfoundry.test_base.TestBase`.
"""

from __future__ import annotations

import warnings

from .test_base import TestBase

__all__ = ["CustomTest", "TestBase"]


class _CustomTestDeprecationProxy(type):
    def __getattribute__(cls, name: str):
        warnings.warn(
            "`kernelfoundry.custom_test.CustomTest` is deprecated; use `kernelfoundry.test_base.TestBase` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


class CustomTest(TestBase, metaclass=_CustomTestDeprecationProxy):
    """Deprecated alias of `kernelfoundry.test_base.TestBase`."""

