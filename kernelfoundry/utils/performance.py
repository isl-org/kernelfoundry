"""Performance utilities (deprecated).

.. deprecated::
    This module is deprecated. Use :mod:`kernelfoundry.eval_pipeline.utils.performance` instead.
"""

import warnings

warnings.warn(
    "kernelfoundry.utils.performance is deprecated, use kernelfoundry.eval_pipeline.utils.performance instead",
    DeprecationWarning,
    stacklevel=2,
)

from kernelfoundry.eval_pipeline.utils.performance import *  # noqa: F401, F403
