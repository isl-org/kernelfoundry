"""System information utilities (deprecated).

.. deprecated::
    This module is deprecated. Use :mod:`kernelfoundry.eval_pipeline.utils.sysinfo` instead.
"""

import warnings

warnings.warn(
    "kernelfoundry.utils.sysinfo is deprecated, use kernelfoundry.eval_pipeline.utils.sysinfo instead",
    DeprecationWarning,
    stacklevel=2,
)

from kernelfoundry.eval_pipeline.utils.sysinfo import *  # noqa: F401, F403
