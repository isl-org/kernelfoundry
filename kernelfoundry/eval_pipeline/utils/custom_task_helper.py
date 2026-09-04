"""Functions for analyzing and manipulating custom tasks."""

import os
import sys
from pathlib import Path
import subprocess
from functools import partial

from kernelfoundry import PACKAGE_ROOT
from kernelfoundry.eval_pipeline.utils.memory_file_map import MemoryFileMap

__all__ = [
    "allowed_text_file_extensions",
    "get_test_names",
    "get_block",
    "has_build_step",
    "has_reference_build_step",
    "get_config",
    "update_block",
    "blocks_to_str",
]


def allowed_text_file_extensions() -> list[str]:
    """Returns a list of allowed text file extensions for custom tasks."""
    return [
        ".sycl",
        ".cu",
        ".cpp",
        ".cxx",
        ".cc",
        ".c",
        ".hpp",
        ".hxx",
        ".hh",
        ".h",
        ".inl",
        ".py",
        ".md",
        ".txt",
        ".cl",
        ".yml",
        ".yaml",
        ".sh",
        ".ps1",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
        ".cmake",
        ".bat",
        ".cmd",
    ]


def get_test_names(task_root: Path | str) -> dict[str, list[str]]:
    """Returns the names of the correctnes and profile tests defined in task.py.

    Args:
        task_root (Path | str): Path to the root directory of the custom task.

    Returns:
        dict[str,list[str]]: A dictionary with three keys: 'all_tests', 'correctness_tests' and 'profile_tests'.
            Each key maps to a list of test names.
    """
    task_root = Path(task_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = PACKAGE_ROOT.parent.as_posix() + os.pathsep + env.get("PYTHONPATH", "")

    # Get all tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            f"--rootdir={str(task_root)}",
            str(task_root / "task.py"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    all_tests = []
    for line in result.stdout.splitlines():
        if "task.py::" in line:
            test_name = line.split("task.py::")[1].strip()
            all_tests.append(test_name)

    # Get profile tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--collect-only",
            f"--rootdir={str(task_root)}",
            "-m",
            "performance",
            str(task_root / "task.py"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to collect profile tests:\n{result.stdout}\n{result.stderr}")
    profile_tests = []
    for line in result.stdout.splitlines():
        if "task.py::" in line:
            test_name = line.split("task.py::")[1].strip()
            profile_tests.append(test_name)

    # Correctness tests are all tests minus profile tests
    correctness_tests = [test for test in all_tests if test not in profile_tests]

    return {
        "all_tests": all_tests,
        "correctness_tests": correctness_tests,
        "profile_tests": profile_tests,
    }


def get_block(task_root: Path | str | MemoryFileMap, key: str) -> dict[str, str]:
    """Returns the text block annotated with the given key.

    Args:
        task_root (Path | str | MemoryFileMap): Path to the root directory of the custom task or the in-memory file map.
        key (str): The key used to annotate the block (e.g., 'REFERENCE', 'EVOLVE',..)

    Returns:
        dict[str,str]: A dictionary with the annotated blocks as values and keys
            as the relative path to the file where the block was found.
    """
    result = {}
    extenstions = allowed_text_file_extensions()
    files = []
    if isinstance(task_root, MemoryFileMap):
        files = task_root.list_files()
        files = sorted([f for f in files if any(f.endswith(ext) for ext in extenstions)])
    else:
        task_root = Path(task_root)
        for ext in extenstions:
            files += list(task_root.rglob(f"*{ext}"))

    for file in sorted(files):
        if isinstance(task_root, MemoryFileMap):
            relative_path = file
            open_fn = partial(task_root.open, file_path=file)
        else:
            relative_path = file.relative_to(task_root)
            open_fn = partial(file.open, encoding="utf-8")

        with open_fn(mode="r") as f:
            lines = f.readlines()

        in_block = False
        current_block_lines = []
        block_contents = []  # list of line-lists, one per block found in this file
        for line in lines:
            if f"[{key}_START]" in line:
                if in_block:
                    raise ValueError(f"Nested [{key}_START] found in {file}")
                in_block = True
                current_block_lines = []
                continue
            if f"[{key}_END]" in line:
                if not in_block:
                    raise ValueError(f"[{key}_END] found without matching [{key}_START] in {file}")
                in_block = False
                block_contents.append(current_block_lines)
                continue
            if in_block:
                current_block_lines.append(line)

        contents = []
        for block_lines in block_contents:
            # Remove leading empty lines
            while block_lines and block_lines[0].strip() == "":
                block_lines.pop(0)
            # Remove trailing empty lines
            while block_lines and block_lines[-1].strip() == "":
                block_lines.pop()
            contents.append("".join(block_lines))
        if contents:
            result[str(relative_path)] = contents
    return result


def update_block(
    task: MemoryFileMap,
    key: str,
    path: str | Path,
    content: str,
    block_index: int = 0,
) -> None:
    """Updates the block annotated with the given key in the specified file.

    This updates the block in-place within the in-memory file map.

    Args:
        task (MemoryFileMap): The in-memory file map of the task.
        key (str): The key used to annotate the block (e.g., 'REFERENCE', 'EVOLVE',..)
        path (str | Path): The path to the file within the task.
        content (str): The new content to insert into the block.
    """
    path_str = str(path)
    lines = []
    with task.open(file_path=path_str, mode="r") as f:
        lines = f.readlines()

    # Collect all (start, end) line-index pairs for occurrences of the block
    occurrences = []
    i = 0
    while i < len(lines):
        if f"[{key}_START]" in lines[i]:
            start = i
            for j in range(i + 1, len(lines)):
                if f"[{key}_END]" in lines[j]:
                    occurrences.append((start, j))
                    i = j + 1
                    break
            else:
                raise ValueError(f"[{key}_START] without matching [{key}_END] in {path_str}")
        else:
            i += 1

    assert len(occurrences) > 0, f"Block with key '{key}' not found in {path_str}"
    assert block_index < len(
        occurrences
    ), f"Block index {block_index} out of range; only {len(occurrences)} '{key}' block(s) in {path_str}"

    start_idx, end_idx = occurrences[block_index]

    # Build the new lines: everything before start, the markers with new content, everything after end
    new_lines = lines[: start_idx + 1]
    new_lines.append(content)
    if not content.endswith("\n"):
        new_lines.append("\n")
    new_lines.extend(lines[end_idx:])

    # Write back to the task
    with task.open(file_path=path_str, mode="w") as f:
        f.writelines(new_lines)


def find_path(task: MemoryFileMap, key: str) -> str:
    """
    Find the path to the file containing the block annotated with the given key.
    """
    extenstions = allowed_text_file_extensions()
    files = task.list_files()
    files = sorted([f for f in files if any(f.endswith(ext) for ext in extenstions)])

    for file in sorted(files):
        open_fn = partial(task.open, file_path=file)
        with open_fn(mode="r") as f:
            file_content = f.read()
        if f"[{key}_START]" in file_content:
            return file


def get_config(task_root: Path | str | MemoryFileMap, config_stem: str = "config") -> dict:
    """Reads the configuration yaml with the specified name in the task root directory.

    Args:
        task_root (Path | str | MemoryFileMap): Path to the root directory of
            the custom task or a MemoryFileMap.
        config_name (str, optional): Name of the config file without extension.
            Defaults to "config".

    Returns:
        dict: The task configuration as a dictionary.
    """
    from omegaconf import OmegaConf

    config_name_variations = [config_stem + ".yaml", config_stem + ".yml"]

    config_files = []
    if isinstance(task_root, MemoryFileMap):
        file_list = task_root.list_files()
        for file_name in file_list:
            for config_name in config_name_variations:
                if config_name == os.path.basename(file_name):
                    config_files.append(file_name)
    else:
        task_root = Path(task_root)
        # Find config file with .yaml or .yml extension
        config_files = []
        for config_name in config_name_variations:
            config_files += list(task_root.glob(f"{config_name}"))

    if len(config_files) == 0:
        raise FileNotFoundError(f"No {' or '.join(config_name_variations)} found in {task_root}")
    if len(config_files) > 1:
        raise ValueError(f"Multiple config files found in {task_root}: {config_files}")

    if isinstance(task_root, MemoryFileMap):
        with task_root.open(file_path=config_files[0], mode="r") as f:
            config = OmegaConf.create(f.read())
    else:
        config = OmegaConf.load(config_files[0])
    return OmegaConf.to_container(config, resolve=True)


def blocks_to_str(blocks: dict[str, list[str]]) -> str:
    """Converts a blocks dict (file_path -> list[content]) to a single string.

    For a single block total, returns the content directly. For multiple blocks,
    combines them with ### Block: file_path ### separator lines. Split back with:
        re.split(r'^### Block: (.+?) ###$', text, flags=re.MULTILINE)

    Args:
        blocks: A dictionary mapping file paths to lists of block contents.

    Returns:
        str: The combined string representation of the blocks.
    """
    all_blocks = [(fp, c) for fp, contents in blocks.items() for c in contents]
    if len(all_blocks) == 1:
        return all_blocks[0][1]
    parts = [f"### Block: {fp} ###\n{c}" for fp, c in all_blocks]
    return "\n".join(parts)


def dict_to_yaml_str(data: dict, indent: int = 0) -> str:
    """Converts a dict to a yaml string.

    This function can be used to print dicts in a yaml format for better readability.

    Args:
        data (dict): The task configuration as a dictionary.
        indent (int): Number of spaces to indent the yaml string.
    """
    from omegaconf import OmegaConf

    omegaconf_config = OmegaConf.create(data)
    yaml_str = OmegaConf.to_yaml(omegaconf_config, resolve=False, sort_keys=False)

    if indent > 0:
        indent_str = " " * indent
        yaml_str = "\n".join(indent_str + line for line in yaml_str.splitlines())

    return yaml_str
