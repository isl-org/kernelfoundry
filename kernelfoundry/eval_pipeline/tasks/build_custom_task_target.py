"""Utility script for invoking custom build functions on task instances.

This standalone script invokes the specified build function of the task that is required to be a subclass of
kernelfoundry.TestBase. The resulting artifacts are serialized to JSON.
"""

import sys
import inspect
import argparse
import json
from pathlib import Path
from kernelfoundry import TestBase
from kernelfoundry.custom_test import CustomTest


def main(config: dict, build_function: str, output_path: Path):
    import task

    # search a subclass of TestBase in the task module (exclude the base classes themselves)
    instance = None
    for _, obj in inspect.getmembers(task):
        if (
            inspect.isclass(obj)
            and issubclass(obj, TestBase)
            and obj is not TestBase
            and obj is not CustomTest
            and hasattr(obj, build_function)
        ):
            instance = obj()
            break
    if instance is None:
        raise RuntimeError(
            f"Could not find a task test class in {task.__file__!r} that "
            f"derives from TestBase and defines '{build_function}()'"
        )
    artifacts = getattr(instance, build_function)(gpu_arch=config["gpu_arch"])
    with open(output_path, "w") as f:
        json.dump({"artifacts": artifacts}, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to input JSON file")
    parser.add_argument(
        "--build_function", required=True, help="Name of the build function to call on the CustomTest subclass"
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to the output JSON file for the artifacts")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)
    main(config, build_function=args.build_function, output_path=args.output)
