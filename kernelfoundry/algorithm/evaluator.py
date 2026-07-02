"""Evaluator for kernel performance (build and test kernel and compare to reference)"""

import logging
from io import StringIO
import numpy as np
from scipy.stats import gmean
from kernelfoundry.algorithm.problem_logger import ProblemLogger
from kernelfoundry.algorithm.schemas import EvalResult
from kernelfoundry.eval_pipeline.task import Task, BuildResult, TestResult, ProcessResult
from kernelfoundry.algorithm.utils.score import get_timing_stats, compute_runtime_improvement, combine_gpu_arch_results
from kernelfoundry.algorithm.utils.eval_helper import *
from kernelfoundry.eval_pipeline.profiler_feedback import (
    get_profiler_feedback_class,
    get_reference_language_for_profiling,
)
from kernelfoundry.eval_pipeline.tasks.task_runner import TaskRunner
from kernelfoundry.algorithm.utils.score import combine_template_results

# if there was a timeout during performance benchmarking, we set the speedup to a very low value and runtime high
SPEEDUP_IF_TIMEOUT: float = 9.999e-10
RUNTIME_IF_TIMEOUT: float = 9.999e10


class Evaluator:
    """Class for evaluating kernel performance (build and test kernel and compare to reference)"""

    def __init__(
        self,
        config,
        problem_logger: ProblemLogger,
        version: int,
        dump_raw_outputs: bool = True,
        kernel_uuid: str | None = None,
        agent_session_id: str | None = None,
    ):
        """Initialize the Evaluator for a kernel task.

        Args:
            config: Configuration for the evaluation job.
            problem_logger (ProblemLogger): Helper object for logging individual outputs.
            version (int): Version (=branch) of the program being evaluated.
            dump_raw_outputs (bool): Whether to dump raw outputs of each evaluation step. Defaults to True.
            kernel_uuid (str | None): Optional kernel UUID to log with. Defaults to None.
            agent_session_id (str | None): Optional agent session ID to log with. Defaults to None.
        """
        self.config = config
        self.trial = problem_logger.trial
        self.version = version
        self.kernel_uuid = kernel_uuid
        self.agent_session_id = agent_session_id
        self._eval_log = StringIO()
        self._eval_log_file = problem_logger.stdout_path_part + f"_v{version}.txt"
        self.dump_raw_outputs = dump_raw_outputs

    def log(self, msg: str) -> None:
        """Log a message to the evaluation log. The evaluation log is a condense log of the evaluation process.

        Args:
            msg (str): message to log
        """
        self._eval_log.write(msg + "\n")
        # TODO do not open and close file every time
        with open(self._eval_log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _log_process_result(self, name: str, result: ProcessResult, worker_info: dict | None = None) -> None:
        if not self.dump_raw_outputs or not result:
            return
        output_string = ""
        if result.returncode is not None:
            output_string += f"Return code: {result.returncode}\n"
        if result.message is not None:
            output_string += f"Message: {result.message}\n"
        if result.stdout is not None and result.stdout.strip() != "":
            output_string += "------ Standard Output ------\n"
            output_string += result.stdout + "\n"
        if result.stderr is not None and result.stderr.strip() != "":
            output_string += "------ Standard Error ------\n"
            output_string += result.stderr + "\n"

        print(output_string)  # also print to console for easier debugging
        log_extra = {
            "data": {
                "log": output_string,
                "output_data": result.output_data,
                "worker_info": worker_info,
                "trial": self.trial,
                "version": self.version,
            },
            "kernel_uuid": self.kernel_uuid,
            "agent_session_id": self.agent_session_id,
        }

        logging.raw(name, extra=log_extra)

    def log_raw_result(self, result: BuildResult | TestResult | None, prefix: str | None = None) -> None:
        if not self.dump_raw_outputs:
            return

        try:
            worker_info = result.worker_info
        except AttributeError:
            worker_info = {}
        if isinstance(result, BuildResult):
            prefix = prefix or "build"
            self._log_process_result(prefix, result.result, worker_info)
        elif isinstance(result, TestResult):
            prefix = prefix or "test"
            self._log_process_result(f"{prefix}_correctness", result.correctness_result, worker_info)
            self._log_process_result(f"{prefix}_performance", result.performance_result, worker_info)
            for k, v in result.trace_results.items():
                self._log_process_result(f"{prefix}_trace_{k}", v, worker_info)

    def run(
        self,
        task: Task,
    ) -> tuple[EvalResult, Task]:
        """Evaluate a single task

        Args:
            task (Task): custom task to evaluate

        Returns:
            EvalResult: evaluation result
        """

        assert task.extract_code_result is not None, "Evaluator requires extract_code_result to be set"
        if task.extract_code_result.returncode != 0:
            self.log(task.extract_code_result.stderr)
            return (
                EvalResult(
                    compiled=False, correctness=False, perf_score=0, metadata={}, eval_log=self._eval_log.getvalue()
                ),
                task,
            )

        metadata = {}  # collect worker info here

        # alter the task to skip reference build and test if reference test result is already provided
        _test_result_reference, _build_result_reference = None, None
        if task.test_result_reference is not None:
            task.config["test_reference"] = False  # skip reference test if already provided
            _test_result_reference = task.test_result_reference  # create a backup
            _build_result_reference = task.build_result_reference  # backup reference build result if available
            task.test_result_reference = None  # save some bandwith by not sending the reference test result data

        # check if we need to run reference and custom separately due to different gpu_arch requirements
        gpu_arch = task.config.get("gpu_arch")
        ref_gpu_arch = task.config.get("reference_gpu_arch", gpu_arch)
        if task.config.get("test_reference", True) and ref_gpu_arch != gpu_arch:
            _test_result_reference = self.build_run_reference_separately(task)

        should_run_reference = task.config.get("test_reference", True)
        should_run_custom = task.config.get("test_custom", True)

        # Build if reference needs to be build or if custom needs to be built for testing
        if (should_run_reference and task.has_reference_build_step) or (should_run_custom and task.has_build_step):
            # build
            task = TaskRunner.build_custom_task(task)

        # save reference build result if available and throw error if compile error in reference
        if should_run_reference and task.has_reference_build_step and task.build_result_reference:
            self.log_raw_result(task.build_result_reference, prefix="build_reference")
            assert (
                task.build_result_reference.result.returncode == 0
            ), "Reference build failed, cannot proceed to testing"

        # save build result
        if task.build_result:
            self.log_raw_result(task.build_result, prefix="build")
            metadata["compile_worker_info"] = task.build_result.worker_info

        # restore backup of the reference build result (may be needed for testing custom to compare to reference)
        if _build_result_reference is not None:
            task.build_result_reference = _build_result_reference

        # run tests on custom and reference kernels
        task_build_ok = (not task.has_build_step) or (task.build_result and task.build_result.result.returncode == 0)
        test_results = {}  # empty dict if no testing to be done
        if task.config.get("test_reference", True) or (should_run_custom and task_build_ok):
            gpu_arch = task.config["gpu_arch"].split(",")
            # run for every gpu architecture
            test_results = {}
            for arch in gpu_arch:
                test_result_arch = TaskRunner.test_custom_task(task, gpu_arch=arch)
                test_results[arch] = test_result_arch

        # get reference result
        if _test_result_reference is not None:
            task.test_result_reference = _test_result_reference  # restore backup of the reference test result
        elif task.config["test_reference"]:
            ref_results = {arch: arch_result["reference"] for arch, arch_result in test_results.items()}
            task.test_result_reference = TestResult.merge(ref_results)
            for arch, ref_result in ref_results.items():
                self.log_raw_result(ref_result, prefix="test_reference")
            assert (
                task.test_result_reference.correctness_result.returncode == 0
            ), "Reference test failed, cannot proceed to testing generated kernel"

        # if custom build failed, return EvalResult with compile error
        if task.build_result and task.build_result.result.returncode != 0:
            src_file_paths = set(task.task_data.list_files())
            self.log(postprocess_compiler_output(task.build_result.result, src_file_paths))
            return (
                EvalResult(
                    compiled=False,
                    correctness=False,
                    perf_score=1,
                    metadata=metadata,
                    eval_log=self._eval_log.getvalue(),
                ),
                task,
            )

        # get all custom results (usually just one, only more if kernel is templated
        custom_result_keys = [k for k in test_results[gpu_arch[0]].keys() if k.startswith("custom")]
        if len(custom_result_keys) == 0:
            # only reference was tested
            return

        arch_exec_results, arch_test_results = {}, {}
        for arch in gpu_arch:
            exec_result_list, template_parameters = [], []
            for custom_key in custom_result_keys:
                # dump raw test results
                self.log_raw_result(test_results[arch][custom_key], prefix=f"test_{custom_key}")
                # get reference result
                if task.test_result_reference is None:
                    ref_result_arch = None
                elif task.test_result_reference.results_per_gpu is None:
                    ref_result_arch = task.test_result_reference
                else:
                    ref_result_arch = task.test_result_reference.results_per_gpu.get(arch, None)
                # transform to kernelExecResult
                exec_result = self.convert_test_result_to_exec_result(
                    test_result=test_results[arch][custom_key],
                    test_result_reference=ref_result_arch,
                    metadata=metadata,
                    language=task.config.get("language", ""),
                    ref_language=get_reference_language_for_profiling(task.config),
                    gpu_arch=arch,
                )
                exec_result_list.append(exec_result)
                # add list to template parameters
                template_parameters.append(custom_key.split("_")[1:])

            # combine template results if there a multiple
            if len(exec_result_list) > 1:
                # find best kernel exec results, store results for other parameters in metadata
                combined_exec_result, index_of_best = combine_template_results(
                    exec_result_list, template_parameters, return_index=True
                )
                best_test_result = test_results[arch][custom_result_keys[index_of_best]]
            else:
                combined_exec_result, best_test_result = exec_result_list[0], test_results[arch][custom_result_keys[0]]
            arch_exec_results[arch] = combined_exec_result
            arch_test_results[arch] = best_test_result

        # merge the test results (simply concatenates all stdout and stderr etc)
        task.test_result = TestResult.merge(arch_test_results)

        # Combine all gpu-specific results. Use current eval log because it contains all results.
        mean_arch_exec_result = combine_gpu_arch_results(arch_exec_results, combined_eval_log=self._eval_log.getvalue())
        return mean_arch_exec_result, task

    def build_run_reference_separately(self, task: Task) -> TestResult:
        """
        Build and run reference on its own by overwriting language and gpu_arch.

        Args:
            task (Task): custom task to evaluate

        Returns:
            TestResult: reference test result
        """
        logging.info("Reference arch is different from custom gpu arch - test separately")
        # backup original config
        original_config = task.config.copy()
        # alter config to run reference test only
        task.config["has_build_step"] = False
        task.config["language"] = get_reference_language_for_profiling(task.config)
        task.config["gpu_arch"] = task.config.get("reference_gpu_arch", original_config["gpu_arch"])
        assert (  # multi-gpu not supported since it's unclear which reference-gpu-result to compare to which custom result
            len(task.config["gpu_arch"].split(",")) == 1
        ), "Only one reference gpu arch is supported if reference runs on other hardware"
        # build
        logging.info(
            f"Building reference for gpu_arch {task.config['gpu_arch']} with language {task.config['language']}"
        )
        if task.has_reference_build_step:
            built_task = TaskRunner.build_custom_task(task)
            task.build_result_reference = built_task.build_result_reference
            assert task.build_result_reference.result.returncode == 0, f"Reference build failed, cannot proceed"
        self.log_raw_result(task.build_result_reference, prefix="build_reference")
        # run reference on the specified gpu
        test_output_reference = TaskRunner.test_custom_task(task)
        test_result_reference = test_output_reference.get("reference", None)
        # restore attributes
        task.config["language"] = original_config["language"]  # restore language config
        task.config["test_custom"] = original_config.get("test_custom", True)  # restore custom test config
        task.config["has_build_step"] = original_config.get("has_build_step", True)  # restore custom build config
        task.config["gpu_arch"] = original_config["gpu_arch"]  # restore gpu arch
        # disable reference build and test for later since we already did it here
        task.config["test_reference"] = False
        self.log_raw_result(test_result_reference, prefix="test_reference")
        return test_result_reference

    def convert_test_result_to_exec_result(
        self,
        test_result: TestResult,
        test_result_reference: TestResult,
        metadata: dict,
        language: str,
        ref_language: str,
        gpu_arch: str,
    ) -> EvalResult:
        """Convert correctness and benchmarking results into EvalResult"""

        metadata["eval_worker_info"] = test_result.worker_info

        # -------- Part 1: Correctness evaluation --------
        if test_result.correctness_result is None:
            self.log("No test results found for custom task.")
            return EvalResult(
                compiled=True,
                correctness=False,
                perf_score=2,
                metadata=metadata,
                eval_log=self._eval_log.getvalue(),
            )
        self.log(f"\nTest result on platform {test_result.worker_info['gpu_name']}:")
        if test_result.correctness_result.returncode != 0:
            self.log(postprocess_pytest_output(test_result.correctness_result))
            if test_result.correctness_result.message:
                self.log(test_result.correctness_result.message)
            if test_result.correctness_result.returncode != 1:
                # pytest returns 1 if there are test failures
                # if there are segfaults or other errors, return code is different
                perf_score = 2  # compiled but runtime error
            elif check_shape_test_result(test_result.correctness_result):
                perf_score = 4  # value mismatch
            else:
                # Note that this will be the score for tasks that do not have shape tests too
                perf_score = 3  # shape mismatch

            return EvalResult(
                compiled=True,
                correctness=False,
                perf_score=perf_score,
                metadata=metadata,
                eval_log=self._eval_log.getvalue(),
            )

        self.log(postprocess_pytest_output(test_result.correctness_result))
        self.log("The kernel compiles and is correct, great job!")

        ker = EvalResult(compiled=True, correctness=True, perf_score=5, metadata=metadata)

        # -------- Part 2: runtimes (without speedup so far) --------
        # runtime stats is a dictionary of the form {"benchmark1": {"mean": ..., "std": ...}, "benchmark2": {...}, ...}
        runtime_stats = {}
        if (
            test_result.performance_result is not None
            and test_result.performance_result.output_data is not None
            and test_result.performance_result.output_data.get("runtimes") is not None
            and len(test_result.performance_result.output_data["runtimes"]) > 0
        ):
            for benchmark_name, runtimes in test_result.performance_result.output_data["runtimes"].items():
                runtime_stats[benchmark_name] = get_timing_stats(runtimes)
        else:
            self.log("\nNo runtime measurement found for custom. Setting runtime to high value.")
            perf_result = test_result.performance_result
            if perf_result is not None and perf_result.message is not None and "timed out" in perf_result.message:
                self.log("There was a timeout error during benchmarking.\n")
            logging.warning("No performance measurement found (e.g. due to timeout). Setting runtime to high value.")
            runtime_stats["timed_out_test"] = get_timing_stats([RUNTIME_IF_TIMEOUT])  # set to very high runtime
        ker.runtime_stats = runtime_stats
        # use average runtime over all tested inputs
        ker.runtime = np.mean([stats["mean"] for stats in ker.runtime_stats.values()])

        # -------- Part 3: profiler feedback --------
        try:
            # Select profiler based on language
            profiler_feedback_class = get_profiler_feedback_class(
                language, gpu_arch, self.config.get("profiler_kernel")
            )
            profiler_feedback = profiler_feedback_class()
            profiler_name = profiler_feedback.name
            profiler_collated_data, profiler_feedback_str = profiler_feedback.collate_and_create_feedback(
                {k: v.output_data for k, v in test_result.trace_results.items() if v.output_data},
                worker_info=test_result.worker_info,
            )
            if profiler_collated_data:
                ker.profiler_data[profiler_name] = profiler_collated_data
                if profiler_feedback_str:
                    self.log(f"\nProfiler output on platform {test_result.worker_info['gpu_name']}:")
                    self.log(profiler_feedback_str)

            if test_result_reference:
                ref_profiler_feedback_class = get_profiler_feedback_class(
                    ref_language, gpu_arch, self.config.get("profiler_reference")
                )
                ref_profiler_feedback = ref_profiler_feedback_class()
                profiler_collated_data = ref_profiler_feedback.collate_data(
                    {k: v.output_data for k, v in test_result_reference.trace_results.items() if v.output_data},
                )
                if profiler_collated_data:
                    ker.profiler_data[f"{ref_profiler_feedback.name}_ref"] = profiler_collated_data

        except Exception as e:
            logging.error(f"Failed to process profiler feedback for language {language} with error: {e}")

        ker.eval_log = self._eval_log.getvalue()
        if test_result_reference is None:
            logging.warning("No reference performance result available, cannot compute speedup.")
            return ker

        # -------- Part 4: compute speedup --------
        perf_res = test_result_reference.performance_result
        if perf_res is None or perf_res.output_data is None or perf_res.output_data.get("runtimes") is None:
            logging.warning("No performance result or runtimes available for reference. Set speedup to low value.")
            ker.runtime_improvement = SPEEDUP_IF_TIMEOUT  # low speedup
            for benchmark_name in ker.runtime_stats.keys():
                ker.runtime_stats[benchmark_name]["speedup"] = SPEEDUP_IF_TIMEOUT
        else:
            # get reference runtimes
            ref_runtimes = test_result_reference.performance_result.output_data["runtimes"]
            # compute speedup per benchmark (per tested input)
            for benchmark_name in ker.runtime_stats.keys():
                if benchmark_name in ref_runtimes:
                    ref_runtime_stats = get_timing_stats(ref_runtimes[benchmark_name])
                    ref_runtime = ref_runtime_stats.get("mean", None)
                    custom_runtime = ker.runtime_stats[benchmark_name].get("mean", None)
                    if ref_runtime is not None and ref_runtime > 0 and benchmark_name != "timed_out_test":
                        # write speedpup to runtime stats dict
                        logging.debug(
                            "Reference runtime for benchmark {}: {}, custom runtime: {}".format(
                                benchmark_name, ref_runtime, custom_runtime
                            )
                        )
                        ker.runtime_stats[benchmark_name]["speedup"] = float(ref_runtime) / custom_runtime
                        ker.runtime_stats[benchmark_name]["ref_speed"] = ref_runtime
                    elif benchmark_name == "timed_out_test":
                        # low speedup if it timed out in the real one
                        ker.runtime_stats[benchmark_name]["speedup"] = SPEEDUP_IF_TIMEOUT
                    else:
                        logging.warning(
                            f"Reference runtime for benchmark {benchmark_name} is invalid. Set speedup to low value."
                        )
                        ker.runtime_stats[benchmark_name]["speedup"] = SPEEDUP_IF_TIMEOUT  # low speedup
                else:
                    logging.warning(
                        f"Benchmark {benchmark_name} not found in reference runtimes. Set speedup to low value."
                    )
                    ker.runtime_stats[benchmark_name]["speedup"] = SPEEDUP_IF_TIMEOUT  # low speedup

            speedups = [stats["speedup"] for stats in ker.runtime_stats.values() if stats["speedup"] > 0]
            ker.runtime_improvement = gmean(speedups) if speedups else -1

        return ker
