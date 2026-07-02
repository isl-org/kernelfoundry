"""Interfaces and schemas for kernel Program objects and evaluation results."""

import json
import time
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dataclasses import asdict, dataclass, field, fields

from kernelfoundry.eval_pipeline.database.tables import Kernel
from kernelfoundry.eval_pipeline.task import Task


class EvalResult(BaseModel):
    """Results from evaluating a single kernel.

    This class stores comprehensive evaluation metrics for a kernel execution,
    including compilation status, correctness results, performance data, and profiling
    information. It also provides unified performance scoring and status reporting.
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance
    runtime_improvement: float = -1.0  # in us, only recorded if we decide to measure performance
    improve_over_compile: float = -1.0
    perf_score: int = -1  # 0: syntax error, 1: not compiled, 2: compiled but runtime error,
    # 3: shape mismatch, 4: value mismatch, 5: correctness pass
    profiler_data: dict = {}  # raw data from profiler, if collected
    template_results: dict = {}
    eval_log: str = ""  # The log produced during evaluation

    @staticmethod
    def compute_performance_score(eval_result: "EvalResult") -> float:
        """
        Compute unified performance score for a kernel evaluation result.

        This is the single source of truth for performance scoring,
        used consistently across:
        - MAP-Elites elite selection (combined_score in metrics)
        - Best kernel selection for next iteration
        - Prompt evolution fitness tracking

        Fitness calculation:

        - Base: perf_score (0-5, discrete quality indicator)
        - If runtime_improvement > 0: add actual speedup (when test_reference=True)
        - Else if runtime_improvement == -1 (test_reference=False) and kernel is correct:
          add 1/runtime to prefer faster kernels

        Args:
            eval_result: EvalResult object with perf_score, runtime_improvement, correctness, runtime

        Returns:
            float: Composite performance score (higher is better)
        """
        assert eval_result.perf_score != -1, "perf_score is still -1, program must be evaluated first"
        score = eval_result.perf_score

        # Add runtime component if available
        if eval_result.runtime_improvement > 0:
            # Use actual speedup if test_reference=True
            score += eval_result.runtime_improvement
        elif eval_result.runtime_improvement == -1 and eval_result.correctness and eval_result.runtime > 0:
            # Fallback when test_reference=False: use reciprocal of runtime to prefer faster kernels
            score += 1.0 / eval_result.runtime

        return score

    @staticmethod
    def get_eval_status(compiled: bool, correctness: bool, for_prompt=False) -> str:
        """Convert correctness and compilation variables into a string status variable"""
        if correctness:
            status = "correct"
            correct_info = "compiles and correct"
        elif compiled:
            status = "compiled"
            correct_info = "compiled but incorrect"
        else:
            status = "error"
            correct_info = "compilation error"
        if for_prompt:
            return correct_info
        else:
            return status

    def get_status(self):
        return self.get_eval_status(self.compiled, self.correctness)

    def format_for_prompt(self):
        # whether it was correct
        correct_info = self.get_eval_status(self.compiled, self.correctness, for_prompt=True)
        # which hardware it ran on
        hardware_name = self.runtime_stats.get("hardware", None)
        hardware = ", on a " + hardware_name if hardware_name else ""
        # together
        info_for_prompt = f"Correctness score: {self.perf_score} / 5 ({correct_info})"
        info_for_prompt += f", runtime in ms: {self.runtime:.3f}{hardware}"
        # same for all the template results
        if len(self.template_results) > 0:
            info_for_prompt += " This templated kernel was tested with the following template parameters: "
            per_param_results = []
            for params, res_for_params in self.template_results.items():
                score, runtime = res_for_params["perf_score"], res_for_params["runtime"]
                per_param_results.append(f"{params} (score: {score}/5, runtime: {runtime:.3f}{hardware})")
            info_for_prompt += ", ".join(per_param_results)
        return info_for_prompt

    def to_dict(self):
        """Convert exec result to dictionary"""
        return {
            "compiled": self.compiled,
            "correctness": self.correctness,
            "metadata": self.metadata,
            "runtime": self.runtime,
            "runtime_stats": self.runtime_stats,
            "runtime_improvement": self.runtime_improvement,
            "improve_over_compile": self.improve_over_compile,
            "perf_score": self.perf_score,
            "profiler_data": self.profiler_data,
            "template_results": self.template_results,
        }

    def __str__(self):
        tmp = self.to_dict()
        if self.profiler_data:
            # profiler data can be very large, so just indicate if collected
            tmp["profiler_data"] = "collected"
        if self.metadata.get("eval_worker_info"):
            worker_info = self.metadata.get("eval_worker_info")
            hostname = (
                self.metadata["eval_worker_info"].get("hostname", "unknown")
                if isinstance(worker_info, dict)
                else worker_info
            )
            tmp["metadata"]["eval_worker_info"] = f"present [{hostname}]"
        if self.metadata.get("compile_worker_info"):
            if isinstance(self.metadata["compile_worker_info"], str):
                hostname = self.metadata["compile_worker_info"]
            else:
                hostname = self.metadata["compile_worker_info"].get("hostname", "unknown")
            tmp["metadata"]["compile_worker_info"] = f"present [{hostname}]"
        return "EvalResult\n" + json.dumps(tmp, indent=2)


@dataclass
class Program:
    """Represents a kernel program candidate in the optimization database.

    This class encapsulates a kernel implementation along with its metadata,
    evaluation results, and evolutionary history. Programs are the core units
    tracked through the kernel optimization pipeline.
    """

    # Program identification
    id: str
    code: str
    is_program0: bool = False
    raw_llm_code: Optional[str] = None  # Original LLM output if available
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Template
    is_templated: bool = False
    template_parameter_combinations: list = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Eval results
    kernel_exec_result: EvalResult = None
    feedback: str = None

    task: Task | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    def get_artifact(self) -> str:
        """Get artifact from here or EvalResult"""
        assert self.kernel_exec_result.eval_log, "exec result must have non-empty eval log"
        return self.kernel_exec_result.eval_log

    def add_eval_results(self, exec_result: EvalResult, artifact_path: str = None, artifact_str: str = None):
        """Add evaluation results to the program"""
        # either the log is already part of the exec result, or it is provided as path or str
        assert exec_result.eval_log or artifact_path or artifact_str, "Eval result requires path or string to eval log"
        # load eval log and add to exec result
        if not exec_result.eval_log:
            if not artifact_str:
                with open(artifact_path, "r") as inf:
                    artifact_str = inf.read()
            exec_result.eval_log = artifact_str

        # add exec result to program
        self.kernel_exec_result = exec_result

        # Compute combined score using unified performance scoring function
        comb_score = EvalResult.compute_performance_score(exec_result)
        metrics = {
            "combined_score": comb_score,
            "runtime_improvement": exec_result.runtime_improvement,
            "performance_score": exec_result.perf_score,
            "compilation_success": exec_result.compiled,
            "correctness_success": exec_result.correctness,
        }
        self.metrics.update(metrics)

        # add templated information:
        if len(exec_result.template_results) > 0:
            self.is_templated = True
            self.template_parameter_combinations = [
                eval(param_str) for param_str in exec_result.template_results.keys()
            ]

    @staticmethod
    def populate_kernel_from_exec_result(kernel: Kernel, exec_result: EvalResult):
        kernel.eval_log = exec_result.eval_log
        kernel.status = exec_result.get_status()
        kernel.runtime = exec_result.runtime
        kernel.score = exec_result.perf_score
        kernel.improve_over_native = exec_result.runtime_improvement
        kernel.improve_over_compile = exec_result.improve_over_compile
        kernel.runtime_stats = exec_result.runtime_stats
        if exec_result.profiler_data:
            custom_profiler_data = {}
            custom_profiler_data_detail = {}
            reference_profiler_data = {}
            reference_profiler_data_detail = {}
            for k, v in exec_result.profiler_data.items():
                if k.endswith("_ref"):
                    reference_profiler_data[k[:-4]] = v
                else:
                    custom_profiler_data[k] = v
            # Separate large profiler data
            for small_data, large_data in [
                (custom_profiler_data, custom_profiler_data_detail),
                (reference_profiler_data, reference_profiler_data_detail),
            ]:
                for profiler_k, data_dict in small_data.items():
                    # The timeline and lineinfo can be very large, so separate them
                    for key in ("timeline", "lineinfo"):
                        if key in data_dict and data_dict[key] is not None:
                            if large_data.get(profiler_k) is None:
                                large_data[profiler_k] = {}
                            large_data[profiler_k][key] = data_dict[key]
                            # Remove large data from small_data
                            del data_dict[key]
            kernel.profiler_data = custom_profiler_data if custom_profiler_data else None
            kernel.profiler_data_detail = custom_profiler_data_detail if custom_profiler_data_detail else None
            kernel.profiler_data_reference = reference_profiler_data if reference_profiler_data else None
            kernel.profiler_data_detail_reference = (
                reference_profiler_data_detail if reference_profiler_data_detail else None
            )

        kernel.template_results = exec_result.template_results
        arch_compiler_worker_infos = {
            k: v.get("compile_worker_info") for k, v in exec_result.metadata.items() if v.get("compile_worker_info")
        }
        arch_eval_worker_infos = {
            k: v.get("eval_worker_info") for k, v in exec_result.metadata.items() if v.get("eval_worker_info")
        }
        kernel.compile_worker_info = arch_compiler_worker_infos
        kernel.eval_worker_info = arch_eval_worker_infos

    def update_Kernel(self, kernel: Kernel):
        """Populate a Kernel database object from this Program"""
        kernel.uuid = self.id
        kernel.parent_uuid = self.parent_id if self.parent_id else None

        kernel.answer = self.raw_llm_code

        if self.metadata.get("feature_coords") is not None:
            kernel.optimization_profile = {"map_elite_cell": self.metadata.get("feature_coords")}
        else:
            # Local import avoids circular dependency with evolve_database_optimization_aware.
            from kernelfoundry.algorithm.evolve_database_optimization_aware import OptimizationFeatureClassifier

            optim_profile = OptimizationFeatureClassifier.classify_from_code(self.code)
            kernel.optimization_profile = {"map_elite_cell": list(optim_profile)}

        kernel.output_code = self.code
        kernel.output_language = self.language

        if self.kernel_exec_result is not None:
            self.populate_kernel_from_exec_result(kernel, self.kernel_exec_result)

        return kernel

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())

        return cls(**filtered_data)


@dataclass
class CompilationResult:
    """Results from compiling a kernel."""

    idx: int
    binary: Optional[object]
    kernel_exec_result: Optional[object]
    compile_success: bool
    error: Optional[str] = None
    worker_info: Optional[dict] = None
