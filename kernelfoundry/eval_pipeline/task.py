"""Defines the Task class and related classes for managing kernel generation tasks, including build and test results."""

from typing import Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from tempfile import TemporaryDirectory
from omegaconf import DictConfig, OmegaConf
from kernelfoundry.eval_pipeline.utils.memory_file_map import MemoryFileMap
from kernelfoundry.eval_pipeline.database.tables import Task as DB_Task
from kernelfoundry.eval_pipeline.utils.custom_task_helper import blocks_to_str
from kernelfoundry.eval_pipeline.utils.formatting import strip_terminal_escapes


@dataclass
class ProcessResult:
    """This collects the output of a subprocess execution with a custom message and output data."""

    returncode: int
    stdout: str
    stderr: str
    # Message we set to capture more info like process timeouts etc.
    message: str | None = None
    output_data: dict | None = None

    def __post_init__(self):
        if isinstance(self.stdout, bytes):
            self.stdout = self.stdout.decode("utf-8", errors="replace")
        if isinstance(self.stderr, bytes):
            self.stderr = self.stderr.decode("utf-8", errors="replace")
        self.stdout = strip_terminal_escapes(self.stdout)
        self.stderr = strip_terminal_escapes(self.stderr)
        if self.message is not None:
            # Ours, but it interpolates captured child output in several places.
            self.message = strip_terminal_escapes(self.message)

    def combine_output(self, include_message: bool = True) -> str:
        """Combines stdout and stderr into a single string."""
        if include_message and self.message:
            return self.message + "\n" + self.stdout + "\n" + self.stderr
        return self.stdout + "\n" + self.stderr

    @classmethod
    def create(cls, subprocess_result, message: str | None = None, output_data: str | None = None) -> "ProcessResult":
        """Convenience method to create a ProcessResult from a subprocess.CompletedProcess instance."""
        return cls(
            returncode=subprocess_result.returncode,
            stdout=subprocess_result.stdout,
            stderr=subprocess_result.stderr,
            message=message,
            output_data=output_data,
        )

    @classmethod
    def merge(cls, results: dict[str, "ProcessResult"]) -> "ProcessResult":
        """Merges multiple ProcessResult instances into a single ProcessResult by concatenating their outputs."""
        if len(results) == 0:
            return None
        combined_stdout = "\n".join([res.stdout for res in results.values()])
        combined_stderr = "\n".join([res.stderr for res in results.values()])
        combined_returncode = max(res.returncode for res in results.values())
        combined_message = "\n".join([res.message for res in results.values() if res.message])
        return cls(
            returncode=combined_returncode,
            stdout=combined_stdout,
            stderr=combined_stderr,
            message=combined_message,
        )


@dataclass
class BuildResult:
    """The result of a build process."""

    worker_info: dict | None = None
    result: ProcessResult | None = None
    artifacts: MemoryFileMap | None = None

    @classmethod
    def decode(cls, data: dict) -> "BuildResult":
        """Decodes a BuildResult from a dictionary."""
        if data is None:
            return None
        artifacts = None
        if data.get("artifacts") is not None:
            artifacts = MemoryFileMap()
            artifacts.decode(data["artifacts"])
        return cls(
            worker_info=data.get("worker_info"),
            result=ProcessResult(**data["result"]) if data.get("result") is not None else None,
            artifacts=artifacts,
        )


@dataclass
class TestResult:
    """The result of executing the correctness and performance tests."""

    worker_info: dict | None = None
    # The output of the tests without any marker are the "correctness" tests
    correctness_result: ProcessResult | None = None
    # The output of the tests with the "profile" marker
    performance_result: ProcessResult | None = None
    # This includes the output of all profilers, e.g., {'unitrace': {...}, 'other_profiler': {...} }
    trace_results: dict[str, ProcessResult] = field(default_factory=dict)
    # To store the results per gpu architecture in case of a multi-gpu job
    results_per_gpu: dict[str, "TestResult"] | None = None

    @classmethod
    def decode(cls, data: dict) -> "TestResult":
        """Decodes a TestResult from a dictionary."""
        if data is None:
            return None

        # Decode trace_results as dict[str, ProcessResult]
        trace_results = {}
        if data.get("trace_results") is not None:
            trace_results = {key: ProcessResult(**value) for key, value in data["trace_results"].items()}

        return cls(
            worker_info=data.get("worker_info"),
            correctness_result=(
                ProcessResult(**data["correctness_result"]) if data.get("correctness_result") is not None else None
            ),
            performance_result=(
                ProcessResult(**data["performance_result"]) if data.get("performance_result") is not None else None
            ),
            trace_results=trace_results,
        )

    @classmethod
    def merge(cls, results: dict[str, "TestResult"]) -> "TestResult":
        """Merges multiple TestResult instances into a single TestResult."""
        if len(results) == 1:
            return list(results.values())[0]
        merged_worker_info = {gpu: res.worker_info for gpu, res in results.items() if res.worker_info is not None}
        correctness_results = ProcessResult.merge(
            {gpu: res.correctness_result for gpu, res in results.items() if res.correctness_result}
        )
        performance_results = ProcessResult.merge(
            {gpu: res.performance_result for gpu, res in results.items() if res.performance_result}
        )
        merged_trace_results = {}
        for gpu_arch, res in results.items():
            for key, trace_result in res.trace_results.items():
                merged_trace_results[f"{key}_{gpu_arch}"] = trace_result
        return cls(
            worker_info=merged_worker_info,
            correctness_result=correctness_results,
            performance_result=performance_results,
            trace_results=merged_trace_results,
            results_per_gpu=results,
        )


@dataclass
class Task:
    """A Task represents a custom kernel generation problem, including the input code, expected output code,
    and associated tests.
    """

    # The data that defines the task as in-memory fs
    task_data: MemoryFileMap
    # If True the task will be build every time before evaluation
    has_build_step: bool
    has_reference_build_step: bool
    # The names of the correctness tests
    correctness_tests: list[str]
    # The names of the profile tests
    profile_tests: list[str]
    # The config that comes with the task
    config: dict

    # The detected annotated blocks with the mapping key->file_path->content
    # For example:   {'EVOLVE': {'src/kernel.sycl': ['block1', 'block2']}}
    blocks: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    # Not an actual result of a subprocess but used to capture the result of code extraction
    # and handle it in the same manner as the result of the build and test steps.
    extract_code_result: ProcessResult | None = None

    # build artifacts
    build_result: BuildResult | None = None
    build_result_reference: BuildResult | None = None

    test_result: TestResult | None = None
    test_result_reference: TestResult | None = None

    hyperparameters: dict | None = None

    def print_info(self, print_fn: Callable | None = None) -> None:
        """Prints information about the Task."""
        from kernelfoundry.eval_pipeline.utils.custom_task_helper import dict_to_yaml_str

        if print_fn is None:
            print_fn = print
        print_fn("Task Info:")
        print_fn(f"  Has build step: {self.has_build_step}")
        print_fn(f"  Correctness tests: \n{dict_to_yaml_str(self.correctness_tests,4)}")
        print_fn(f"  Profile tests: \n{dict_to_yaml_str(self.profile_tests,4)}")
        print_fn(f"  Config: \n{dict_to_yaml_str(self.config,4)}")
        print_fn(f"  Blocks: \n{dict_to_yaml_str(list(self.blocks.keys()),4)}")

    def apply_config(self, config: DictConfig) -> "Task":
        """Attaches an execution config to the Task and transfers config-driven attributes."""
        self.config = OmegaConf.to_container(config, resolve=True)
        self.has_build_step = self.config.get("has_build_step", self.has_build_step)
        self.has_reference_build_step = self.config.get("has_reference_build_step", self.has_reference_build_step)
        self.hyperparameters = self.config.get("hyperparameters", self.hyperparameters)
        return self

    def validate(self):
        """Checks if the Task is valid."""
        assert self.task_data is not None and len(self.task_data.file_map) > 0, "Task task_data is empty"
        assert "task.py" in self.task_data.file_map, "Task is missing task.py"
        assert "conftest.py" in self.task_data.file_map, "Task is missing conftest.py"
        assert len(self.correctness_tests) > 0, "Task has no correctness tests"
        assert len(self.profile_tests) > 0, "Task has no profile tests"
        assert "EVOLVE" in self.blocks, "Task has no EVOLVE block"
        assert "task_name" in self.config, "Task config is missing task_name"
        assert "job_name" in self.config, "Task config is missing job_name"

    @classmethod
    def create(cls, path_or_bytes: str | Path | bytes) -> tuple["Task", dict]:
        """Creates a Task from a path to a directory, a tar file, zip file or from bytes of a zip/tar file.

        Args:
            path_or_bytes (str | Path | bytes): Path to the task directory, tar file, zip file or bytes of a tar/zip file.

        Returns:
            Task: The created Task instance.
            dict: Dictionary with additional information such as overrides.
        """
        from kernelfoundry.eval_pipeline.utils.custom_task_helper import (
            get_block,
            get_test_names,
            get_config,
        )

        metadata = {}

        # Initialize the original task from the given input
        task_data = MemoryFileMap()
        if isinstance(path_or_bytes, bytes):
            task_data.from_archive(archive_bytes=path_or_bytes)
        else:
            path = Path(path_or_bytes)
            task_data.from_path(path)
        # remove potential root directory. We expect task.py and config.yml at the top level
        task_data.remove_root_dir()

        # Extract information about the task
        config = get_config(task_data)
        # correct task name and job name (for backward compatability to old naming scheme)
        if "job_name" not in config and "run_name" in config:
            config["job_name"] = config["run_name"]
            del config["run_name"]
        if "task_name" not in config and "Op_Name" in config:
            config["task_name"] = config["Op_Name"]
            del config["Op_Name"]
        try:
            # overrides are not part of the task itself, return them as metadata and remove from task_data
            overrides = get_config(task_data, config_stem="overrides")
            metadata["overrides"] = overrides
        except Exception:
            pass
        if "overrides.yml" in task_data:
            del task_data.file_map["overrides.yml"]
            task_data.meta_map.pop("overrides.yml", None)
        elif "overrides.yaml" in task_data:
            del task_data.file_map["overrides.yaml"]
            task_data.meta_map.pop("overrides.yaml", None)

        keywords = ["REFERENCE", "EVOLVE", "USER_INSTRUCTIONS"]
        blocks = {}
        for key in keywords:
            block = get_block(task_data, key)
            if block:
                blocks[key] = block
        # We need to write the task to disk to analyze the tests and build step
        with TemporaryDirectory(prefix="custom_task_create_", delete=True) as temp_dir:
            temp_path = Path(temp_dir)
            task_data.to_disk(output_dir=temp_path)
            analyze_task_root = temp_path

            has_build_step = config.get("has_build_step", True)
            has_reference_build_step = config.get("has_reference_build_step", True)
            test_names = get_test_names(analyze_task_root)
            correctness_tests = test_names.get("correctness_tests", [])
            profile_tests = test_names.get("profile_tests", [])

        return (
            cls(
                task_data=task_data,
                has_build_step=has_build_step,
                has_reference_build_step=has_reference_build_step,
                correctness_tests=correctness_tests,
                profile_tests=profile_tests,
                config=config,
                blocks=blocks,
                hyperparameters=config.get("hyperparameters"),
            ),
            metadata,
        )

    def with_blocks(
        self,
        blocks: dict[str, dict[str, list[str] | None]] | dict[str, str | None],
        keep_test_result_reference: bool = False,
    ) -> "Task":
        """Returns a new Task with the given blocks updated.

        Note that the newly created Task will not contain build or test results as those are not valid
        anymore after changing the blocks! If you want to keep the reference test result, set
        keep_test_result_reference=True.

        Args:
            blocks (dict[str, dict[str, list[str] | None]] | dict[str, str | None]): The blocks to update.
                If the task has only a single block for a key, a simple dict can be provided
                mapping key->content. If multiple blocks exist for a key, a dict mapping
                key->file_path->content must be provided.
                Use a value of None to indicate that the block could not be generated, e.g.,
                {'EVOLVE': None} or {'EVOLVE': {'src/kernel.sycl': None}}
            keep_test_result_reference (bool): If True, the reference test result will be kept in the new task.

        Returns:
            Task: A new Task with the updated blocks.
        """
        from kernelfoundry.eval_pipeline.utils.custom_task_helper import update_block, get_block

        # check if any of the blocks updates is None
        def find_none_blocks():
            for key, value in blocks.items():
                if value is None:
                    return True
                elif isinstance(value, dict):
                    for path, content in value.items():
                        if content is None:
                            return True

        if find_none_blocks():
            return Task(
                task_data=MemoryFileMap(),  # pass empty memory file map to indicate invalid task
                has_build_step=self.has_build_step,
                has_reference_build_step=self.has_reference_build_step,
                correctness_tests=self.correctness_tests,
                profile_tests=self.profile_tests,
                config=self.config.copy(),
                blocks=self.blocks,
                extract_code_result=ProcessResult(
                    returncode=1,
                    stdout="",
                    stderr="SyntaxError. \n Your output was not in the required format. The code could not be extracted.",
                    message="One or more blocks to update were None",
                ),
                test_result_reference=self.test_result_reference if keep_test_result_reference else None,
                build_result_reference=self.build_result_reference if keep_test_result_reference else None,
                hyperparameters=self.hyperparameters,
            )

        new_task_data = MemoryFileMap()
        # Copy existing data
        for file_key in self.task_data.file_map:
            new_task_data.file_map[file_key] = self.task_data.file_map[file_key]
            if file_key in self.task_data.meta_map:
                new_task_data.meta_map[file_key] = self.task_data.meta_map[file_key]

        # update blocks
        for key, value in blocks.items():
            if isinstance(value, str):
                file_paths = list(self.blocks[key].keys())
                if len(file_paths) != 1 or len(self.blocks[key][file_paths[0]]) != 1:
                    raise ValueError(
                        f"Multiple blocks exist for key '{key}'. Provide a dict mapping key->file_path->list[content]."
                    )
                update_block(new_task_data, key=key, path=file_paths[0], content=value, block_index=0)
            else:
                for path, content_list in value.items():
                    for idx, content in enumerate(content_list):
                        update_block(new_task_data, key=key, path=path, content=content, block_index=idx)

        # rebuild blocks from file map
        keywords = ["REFERENCE", "EVOLVE", "USER_INSTRUCTIONS"]
        new_blocks = {}
        for key in keywords:
            block = get_block(new_task_data, key)
            if block:
                new_blocks[key] = block

        return Task(
            task_data=new_task_data,
            has_build_step=self.has_build_step,
            has_reference_build_step=self.has_reference_build_step,
            correctness_tests=self.correctness_tests,
            profile_tests=self.profile_tests,
            config=self.config.copy(),
            blocks=new_blocks,
            extract_code_result=ProcessResult(
                returncode=0,
                stdout="",
                stderr="",
                message="Code blocks have been modified successfully.",
            ),
            test_result_reference=self.test_result_reference if keep_test_result_reference else None,
            build_result_reference=self.build_result_reference if keep_test_result_reference else None,
            hyperparameters=self.hyperparameters,
        )

    def encode(self) -> dict:
        """Encodes the Task to a serializable dictionary."""

        # Helper to encode BuildResult with nested objects
        def encode_build_result(build_result):
            if build_result is None:
                return None
            return {
                "worker_info": build_result.worker_info,
                "result": asdict(build_result.result) if build_result.result is not None else None,
                "artifacts": build_result.artifacts.encode() if build_result.artifacts is not None else None,
            }

        return {
            "task_data": self.task_data.encode(),
            "has_build_step": self.has_build_step,
            "has_reference_build_step": self.has_reference_build_step,
            "correctness_tests": self.correctness_tests,
            "profile_tests": self.profile_tests,
            "config": self.config,
            "blocks": self.blocks,
            "extract_code_result": asdict(self.extract_code_result) if self.extract_code_result is not None else None,
            "build_result": encode_build_result(self.build_result),
            "build_result_reference": encode_build_result(self.build_result_reference),
            "test_result": asdict(self.test_result) if self.test_result is not None else None,
            "test_result_reference": (
                asdict(self.test_result_reference) if self.test_result_reference is not None else None
            ),
            "hyperparameters": self.hyperparameters,
        }

    @classmethod
    def decode(cls, data: dict) -> "Task":
        """Decodes a Task from a serialized dictionary."""

        task_data = MemoryFileMap()
        task_data.decode(data["task_data"])

        return Task(
            task_data=task_data,
            has_build_step=data["has_build_step"],
            has_reference_build_step=data.get("has_reference_build_step", False),
            correctness_tests=data["correctness_tests"],
            profile_tests=data["profile_tests"],
            config=data["config"],
            blocks=data["blocks"],
            extract_code_result=(
                ProcessResult(**data["extract_code_result"]) if data.get("extract_code_result") else None
            ),
            build_result=BuildResult.decode(data.get("build_result")),
            build_result_reference=BuildResult.decode(data.get("build_result_reference")),
            test_result=TestResult.decode(data.get("test_result")),
            test_result_reference=TestResult.decode(data.get("test_result_reference")),
            hyperparameters=data.get("hyperparameters"),
        )

    def to_database_task(self):
        """Convert this Task object into a db.Task object that can be written to the detabase"""
        # Extract task_name, etc. from config
        task_name = self.config["task_name"]
        task_origin = self.config["task_origin"]
        assert task_origin is not None, "task_origin must be specified"

        # get task data
        task_data_dict = self.task_data.encode()

        def blocks_for_db(block_indicator: str) -> str | None:
            b = self.blocks.get(block_indicator, {})
            return blocks_to_str(b) if b else None

        # Create the Task object
        db_task = DB_Task(
            task_name=task_name,
            task_origin=task_origin,
            has_build_step=self.has_build_step,
            config=self.config,
            task_data=task_data_dict,
            correctness_tests=self.correctness_tests,
            profile_tests=self.profile_tests,
            evolve_block=blocks_for_db("EVOLVE"),
            reference_block=blocks_for_db("REFERENCE"),
            user_instructions_block=blocks_for_db("USER_INSTRUCTIONS"),
            hyperparameters_buildtime=(self.hyperparameters or {}).get("buildtime"),
            hyperparameters_runtime=(self.hyperparameters or {}).get("runtime"),
        )
        # generate hash
        db_task.generate_hash_id(task_data_dict)
        return db_task
