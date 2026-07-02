"""Functions for managing environments for executing tasks."""

import os
from pathlib import Path


def safe_copy_env(
    add_vars: dict[str, str] | None = None,
    extend_pythonpath: Path | str | list[str | Path] | None = None,
    src: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create a copy of the current environment with optional additional variables."""

    if src is not None:
        env = src.copy()
    else:
        env = os.environ.copy()
    # remove passwords and other sensitive information from the environment
    rm_keys = []
    for key in env.keys():
        if (
            "PASSWORD" in key.upper()
            or "SECRET" in key.upper()
            or key.upper().endswith("_USERNAME")
            or key.upper().endswith("_TOKEN")
        ):
            rm_keys.append(key)
    for key in rm_keys:
        del env[key]
    if add_vars:
        env.update(add_vars)
    if extend_pythonpath:
        if isinstance(extend_pythonpath, (str, Path)):
            extend_pythonpath = [extend_pythonpath]
        extend_pythonpath_str = os.pathsep.join(str(p) for p in extend_pythonpath)
        env["PYTHONPATH"] = extend_pythonpath_str + os.pathsep + env.get("PYTHONPATH", "")
    return env
