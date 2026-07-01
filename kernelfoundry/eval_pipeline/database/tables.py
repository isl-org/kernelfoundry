from typing import Any
import hashlib
import getpass
from datetime import datetime
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.types import JSON, DateTime, Text
from sqlalchemy import String


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


def _default_db_user() -> str:
    # Use a portable Python-side default instead of DB-specific session_user().
    return getpass.getuser()


class Kernel(Base):
    __tablename__ = "kernels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Creation date of this entry"
    )
    uuid: Mapped[str | None] = mapped_column(comment="uuid of the generated kernel")
    parent_uuid: Mapped[str | None] = mapped_column(comment="uuid of the parent generated kernel")
    task_name: Mapped[str | None] = mapped_column(
        comment='Human-readable task identifier (e.g., "1_add" or "vector_addition")'
    )
    Level_ID: Mapped[int | None] = mapped_column(comment="Task or difficulty level identifier")
    input_code: Mapped[str | None] = mapped_column(Text, comment="The input code")
    input_language: Mapped[str | None] = mapped_column(comment="one of (pytorch, sycl, cuda, ocl, ..)")
    output_code: Mapped[str | None] = mapped_column(Text, comment="The generated code extracted from the answer")
    output_language: Mapped[str | None] = mapped_column(comment="one of (pytorch, sycl, cuda, ocl, ..)")
    Op_ID: Mapped[str | None] = mapped_column(comment="Operation identifier (possibly numeric or short code)")
    job_name: Mapped[str | None] = mapped_column(
        comment="Job name or run directory (which experiment produced the result)"
    )
    prompt: Mapped[str | None] = mapped_column(Text, comment="Input prompt passed to the language model")
    answer: Mapped[str | None] = mapped_column(Text, comment="raw answer received")
    status: Mapped[str | None] = mapped_column(comment='Result status (e.g., "error", "compiled", "correct")')
    trial: Mapped[int | None] = mapped_column(comment="Trial index or repetition number for the job")
    runtime: Mapped[float | None] = mapped_column(
        comment="Measured execution time in seconds (negative or NaN for failures)"
    )
    runtime_stats: Mapped[dict[str, Any] | None] = mapped_column(comment="This is a json object")
    improve_over_native: Mapped[float | None] = mapped_column(
        comment="Relative improvement over native baseline (e.g., speedup factor or delta)"
    )
    improve_over_compile: Mapped[float | None] = mapped_column(
        comment="Improvement relative to torch compiled baseline"
    )
    score: Mapped[int | None] = mapped_column(comment="Integer quality or ranking score assigned to the result")
    eval_log: Mapped[str | None] = mapped_column(Text, comment="Evaluation log")
    version: Mapped[int | None] = mapped_column(comment="id of the solution generated within one trial")
    index_level_0: Mapped[int | None] = mapped_column(comment="DataFrame internal index column (original row index)")
    language_model: Mapped[str | None] = mapped_column(comment="The language model that generated the code")
    profiler_data: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with basic profiler data"
    )
    profiler_data_reference: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with basic profiler data for the reference implementation"
    )
    profiler_data_detail: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with detailed profiler data like metric and event timelines or other large data"
    )
    profiler_data_detail_reference: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with detailed profiler data like metric and event timelines or other large data for the reference implementation"
    )
    template_results: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with the results for each parameter value that was used"
    )
    optimization_profile: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, comment="This is a json object with the results of the map elite evaluation"
    )
    gpu_arch: Mapped[str | None] = mapped_column(comment="This is a string for the GPU architecture")
    eval_worker_info: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with information about the eval worker"
    )
    compile_worker_info: Mapped[dict[str, Any] | None] = mapped_column(
        comment="This is a json object with information about the compile worker"
    )
    config: Mapped[dict[str, Any] | None] = mapped_column(comment="This is a json object with the config used")
    task_id: Mapped[str] = mapped_column(String(64), comment="The id of the task for which this kernel was generated")
    job_id: Mapped[int | None] = mapped_column(comment="The id of the job that generated this kernel")
    agent_session_id: Mapped[str | None] = mapped_column(String(64), comment="Optional agent session identifier")
    db_user: Mapped[str] = mapped_column(String(128), default=_default_db_user, nullable=False, comment="Database user")

    def __str__(self):
        def _abbrev(value: Any, max_len: int = 60) -> str:
            if value is None:
                return "None"
            text = "↩".join(str(value).splitlines())

            return text if len(text) <= max_len else f"{text[:max_len - 3]}..."

        fields = [(column.key, getattr(self, column.key)) for column in self.__table__.columns]

        lines = ["Kernel("]
        lines.extend(f"  {name}={_abbrev(value)}" for name, value in fields)
        lines.append(")")
        return "\n".join(lines)


class BaselineTime(Base):
    __tablename__ = "baseline_times"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(
        nullable=False, comment="Difficulty level of the operation (e.g., level1, level2)"
    )
    task_name: Mapped[str] = mapped_column(
        nullable=False, comment="Operation name (e.g., 1_Square_matrix_multiplication_.py)"
    )
    mean: Mapped[float | None] = mapped_column(comment="Mean runtime in miliseconds")
    std: Mapped[float | None] = mapped_column(comment="Standard deviation of runtime")
    min: Mapped[float | None] = mapped_column(comment="Minimum runtime in miliseconds")
    max: Mapped[float | None] = mapped_column(comment="Maximum runtime in miliseconds")
    num_trials: Mapped[int | None] = mapped_column(comment="Number of trials")
    hardware: Mapped[str | None] = mapped_column(comment="Hardware used for the operation")
    device: Mapped[str | None] = mapped_column(comment="Device used for the operation")

    platform: Mapped[str | None] = mapped_column(comment="Platform used (e.g., A100_modal, Intel_A770)")

    backend: Mapped[str | None] = mapped_column(
        comment="Backend used (e.g., torch, torch_compile_openvino, torch_compile_inductor)"
    )

    def __str__(self):
        return (
            f"BaselineTime(id={self.id}, level={self.level}, task_name={self.task_name}, mean={self.mean}, "
            f"std={self.std}, min={self.min}, max={self.max}, num_trials={self.num_trials}, "
            f"hardware={self.hardware}, device={self.device})"
        )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # extracted from Task config:
    task_name: Mapped[str] = mapped_column(
        comment='Human-readable operation identifier (e.g., "1_add" or "vector_addition")'
    )
    Op_ID: Mapped[str | None] = mapped_column(comment="Operation identifier (possibly numeric or short code)")
    task_origin: Mapped[str] = mapped_column(comment="The source of the task, e.g. KernelBench, custom, etc")
    config: Mapped[dict | None] = mapped_column(JSON, comment="The config that comes with the task")
    # only for KernelBench
    Level_ID: Mapped[int | None] = mapped_column(comment="Task or difficulty level identifier")
    # corresponding to Task object
    task_data: Mapped[dict | None] = mapped_column(JSON, comment="The data that defines the task as in-memory fs")
    has_build_step: Mapped[bool] = mapped_column(
        nullable=False, comment="If True, the task will be built every time before evaluation"
    )
    correctness_tests: Mapped[list[str] | None] = mapped_column(JSON, comment="The names of the correctness tests")
    profile_tests: Mapped[list[str] | None] = mapped_column(JSON, comment="The names of the profile tests")
    evolve_block: Mapped[str | None] = mapped_column(comment="Extracted EVOLVE block")
    reference_block: Mapped[str | None] = mapped_column(comment="Extracted REFERENCE block")
    user_instructions_block: Mapped[str | None] = mapped_column(comment="Extracted USER INSTRUCTIONS block")
    hyperparameters_buildtime: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, comment="Build-time hyperparameters"
    )
    hyperparameters_runtime: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="Runtime hyperparameters")

    def __str__(self):
        return (
            f"TaskData(id={self.id}, Op_ID={self.Op_ID}, task_name={self.task_name}, has_build_step={self.has_build_step}, "
            f"correctness_tests={self.correctness_tests}, profile_tests={self.profile_tests}, config={self.config}, "
            f"hyperparameters_buildtime={self.hyperparameters_buildtime}, "
            f"hyperparameters_runtime={self.hyperparameters_runtime})"
        )

    def generate_hash_id(self, data: str | dict) -> None:
        # Generate a hash using SHA256 or another algorithm
        if isinstance(data, str):
            self.id = hashlib.sha256(data.encode()).hexdigest()
        else:
            sha = hashlib.sha256()
            for path in sorted(data.keys()):
                if path in ["config.yaml", "config.yml"]:
                    continue
                # sha.update(path.encode('utf-8')) # include file path
                content = data[path]
                if isinstance(content, str):
                    content = content.encode("utf-8")
                sha.update(content)  # include the file content
            self.id = sha.hexdigest()


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[str | None] = mapped_column(String(64), comment="Associated task ID")
    task_origin: Mapped[str] = mapped_column(comment="The source of the task, e.g. KernelBench, custom, etc")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="Job status: INIT, RUN, COMPLETE, FAIL")
    progress: Mapped[float] = mapped_column(default=0.0, nullable=False, comment="Job progress from 0.0 to 1.0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Job creation time"
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When job execution started"
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When job completed or failed"
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="When job was archived"
    )
    config: Mapped[dict[str, Any] | None] = mapped_column(comment="This is a json object with the config used")

    def __str__(self):
        return (
            f"Job(id={self.id}, task_id={self.task_id}, status={self.status}, progress={self.progress}, "
            f"created_at={self.created_at}, started_at={self.started_at}, finished_at={self.finished_at}, "
            f"archived_at={self.archived_at}, config={self.config})"
        )


class JobLog(Base):
    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="Message ID")
    job_id: Mapped[int] = mapped_column(nullable=False, comment="Associated job ID")
    hostname: Mapped[str | None] = mapped_column(comment="Hostname of the process that emitted the log entry")
    kernel_uuid: Mapped[str | None] = mapped_column(comment="UUID of the kernel associated with the log entry")
    agent_session_id: Mapped[str | None] = mapped_column(String(64), comment="Optional agent session identifier")
    level: Mapped[str] = mapped_column(
        String(16), nullable=False, comment="Log level: INFO, DEBUG, WARNING, ERROR, etc."
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Time when the message is logged"
    )
    message: Mapped[str] = mapped_column(Text, nullable=False, comment="Log message")
    extra: Mapped[dict[str, Any] | None] = mapped_column(comment="This is an optional json object to hold extra data")

    def __str__(self):
        return (
            f"JobLog(id={self.id}, job_id={self.job_id}, hostname={self.hostname}, "
            f"kernel_uuid={self.kernel_uuid}, level={self.level}, "
            f"timestamp={self.timestamp}, message={self.message[:60]}...)"
        )


class Rag(Base):
    __tablename__ = "rag"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_name: Mapped[str | None] = mapped_column(
        comment='Human-readable operation identifier (e.g., "1_add" or "vector_addition")'
    )
    language: Mapped[str] = mapped_column(String, nullable=False, comment="Programming language of the kernel")
    descriptor: Mapped[str | None] = mapped_column(String, nullable=True, comment="Description for finding matches")
    kernel_code: Mapped[str | None] = mapped_column(String, nullable=True, comment="Source code of the kernel")
    file_path: Mapped[str | None] = mapped_column(String, nullable=False, comment="Path to original kernel file")
    category: Mapped[str] = mapped_column(String, nullable=False, comment="Category (e.g. sdpa, dpas, etc.)")
    keywords: Mapped[str | None] = mapped_column(String, nullable=True, comment="Keywords describing the topic")
    optimization_profile: Mapped[dict[str, float] | list[float] | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Optimization profile vector or coordinate dictionary",
    )
    embedded_text: Mapped[str] = mapped_column(String, nullable=False, comment="Text that was embedded")
    embedding: Mapped[list[float] | None] = mapped_column(
        JSON, nullable=True, comment="Embedding of the code as a list of floats"
    )
    origin: Mapped[str | None] = mapped_column(
        String, nullable=True, comment="Origin of the example (e.g. KernelBench)"
    )

    def __str__(self):
        return (
            f"Rag(id={self.id}, descriptor={self.descriptor}, language={self.language}, "
            f"file_path={self.file_path}, category={self.category}, keywords={self.keywords}, "
            f"optimization_profile={self.optimization_profile}, origin={self.origin})"
        )
