"""Expose the clops C++ extension (csrc)."""

from .csrc import *  # noqa: F401, F403

# ── helpers ──────────────────────────────────────────────────────────────────


def source(options=""):
    """Decorator that compiles a CM/OCL kernel from a function's docstring.

    Example::

        @cl.source(options="-cmc")
        def my_kernel():
            return r'''
            extern "C" _GENX_MAIN_ void add(...) { ... }
            '''
    """
    import inspect

    def _cl_kernel(f):
        frame = inspect.currentframe().f_back
        src_lines, line_no = inspect.getsourcelines(f)
        src = "\n" * (line_no + 1) + f()
        return kernels(src, options)

    return _cl_kernel
