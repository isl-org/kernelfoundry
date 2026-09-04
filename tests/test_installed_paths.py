"""Guard the installed-package path: KernelFoundry must run from a wheel, not only a checkout.

``configs/`` ships as package data and :mod:`kernelfoundry` resolves it directly, so
nothing has to search upward for a ``.project-root`` marker. Both properties are easy to undo by
accident, and the failure is invisible in a checkout:

* an ``import autoroot`` anywhere in the package raises for installed users, since ``autoroot``
  is not a dependency and searches upward for a marker that is not there;
* a config that stops shipping makes hydra composition fail only outside the repository.

These tests fail in either case. They need no GPU, no LLM key and no network.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kernelfoundry import CONFIG_DIR, PACKAGE_ROOT

# Config groups that ``run.yaml`` lists under ``defaults``. If any stops shipping, hydra
# composition fails for an installed user with a "Could not find" error.
REQUIRED_CONFIGS = (
    "run.yaml",
    "run_agentic.yaml",
    "base.yaml",
    "paths/default.yaml",
    "inference/server.yaml",
    "prompt/default.yaml",
    "prompt/agentic.yaml",
    "task_set/default.yaml",
    "controller/copilot.yaml",
    "skills/default.yaml",
)


def test_config_dir_is_inside_the_package():
    """The config tree must live in the package, not at the repository root."""
    assert CONFIG_DIR.is_dir(), f"{CONFIG_DIR} does not exist"
    assert CONFIG_DIR.parent == PACKAGE_ROOT, (
        f"configs must ship inside the package so they are present in a wheel; "
        f"found {CONFIG_DIR} outside {PACKAGE_ROOT}"
    )


@pytest.mark.parametrize("relative", REQUIRED_CONFIGS)
def test_required_config_ships(relative):
    assert (CONFIG_DIR / relative).is_file(), f"missing shipped config: {relative}"


def test_run_yaml_composes_without_the_repository_marker(tmp_path):
    """Compose ``run.yaml`` from a working directory with no ``.project-root`` above it.

    This is the check that fails for a wheel install when configs stop shipping, and it runs in a
    subprocess so hydra's global state cannot leak between tests.
    """
    pytest.importorskip("hydra")
    script = (
        "from hydra import compose, initialize_config_dir\n"
        "from kernelfoundry import CONFIG_DIR\n"
        "with initialize_config_dir(version_base='1.3', config_dir=str(CONFIG_DIR)):\n"
        "    cfg = compose(config_name='run')\n"
        "assert cfg.paths is not None, 'paths group did not resolve'\n"
        "assert 'branches_per_iteration' in cfg, 'run.yaml did not compose'\n"
        "print('composed')\n"
    )
    # Point the subprocess at this checkout explicitly. Without it, running from tmp_path makes
    # `import kernelfoundry` resolve to whatever happens to be installed in the environment, which
    # may be an older copy: the test would then report on that instead of on the code under test.
    # cwd stays in tmp_path, since having no .project-root above it is the whole point.
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        timeout=180,
        env=env,
    )
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, f"composition failed outside a checkout:\n{stderr}"
    assert "composed" in stdout


def test_run_agentic_yaml_composes_without_the_repository_marker(tmp_path):
    """Compose ``run_agentic.yaml`` the same way ``test_run_yaml_composes_without_the_repository_marker``
    does for ``run.yaml``.

    ``run_agentic.yaml`` shares its base fields with ``run.yaml`` via ``base.yaml`` and layers on
    the ``controller``/``skills`` groups, so this guards both that composition (defaults-list
    ordering must let its own overrides win over ``base.yaml``, not the other way round) and that
    the extra config groups ship with the wheel.
    """
    pytest.importorskip("hydra")
    script = (
        "from hydra import compose, initialize_config_dir\n"
        "from kernelfoundry import CONFIG_DIR\n"
        "with initialize_config_dir(version_base='1.3', config_dir=str(CONFIG_DIR)):\n"
        "    cfg = compose(config_name='run_agentic')\n"
        "assert cfg.controller is not None, 'controller group did not resolve'\n"
        "assert cfg.skills is not None, 'skills group did not resolve'\n"
        "assert cfg.agentic_workflow is True, 'run_agentic.yaml did not compose'\n"
        "print('composed')\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        timeout=180,
        env=env,
    )
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, f"composition failed outside a checkout:\n{stderr}"
    assert "composed" in stdout


def _module_scope_autoroot_imports(path: Path) -> list[int]:
    """Line numbers of ``import autoroot`` statements at module scope in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "autoroot" for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "autoroot":
                lines.append(node.lineno)
    return lines


def test_no_module_scope_autoroot_import_in_the_package():
    """``autoroot`` is not a dependency and must not be imported anywhere in the package.

    Regression guard: this exact import was reintroduced into ``mcp_server/server.py`` days after
    it was first removed, and would break every installed user's entry point.
    """
    offenders = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found = _module_scope_autoroot_imports(path)
        if found:
            offenders[path.relative_to(PACKAGE_ROOT).as_posix()] = found
    assert not offenders, f"module-scope `import autoroot` found: {offenders!r}"
