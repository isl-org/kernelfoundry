"""Logger for kernel generation and evaluation results."""

import os
import json
import re
import warnings
from collections import defaultdict
import warnings
import numpy as np
from kernelfoundry.algorithm.schemas import EvalResult


def extract_result_from_log(log: str) -> tuple[dict, EvalResult]:
    """Reverse engineer result from a log file
    Args:
        log: str, log file loaded as a string
    Returns:
        dict, exec result as dict
        EvalResult, the result inferred based on the log
    """
    result = EvalResult()

    if "could not be extracted" in log:
        result.perf_score = 0
    else:
        # at least 1 if it could be extracted
        result.perf_score = 1

    # Check if compilation works
    if "Correctness check" in log:
        result.compiled = True
        result.perf_score = 2

    if "shape_mismatch" in log:
        result.perf_score = 3

    if "Pass count" in log:
        result.perf_score = 4  # the correctness check did not fail

    # Check if correctness check has passed and performance measurement has started
    if "Performance measurement:" in log:
        result.correctness = True
        result.perf_score = 5

        # Extract runtime stats using regex
        runtime_stats_match = re.search(r"Runtime stats:\s*({.*?})", log)
        if runtime_stats_match:
            runtime_stats_str = runtime_stats_match.group(1)
            result.runtime_stats = eval(runtime_stats_str)  # Unsafe for untrusted input!

            # Extract runtime (mean of runtime stats)
            result.runtime = result.runtime_stats.get("mean", -1.0)

    result_dict = result.to_dict()
    result_dict["status"] = result.get_status()
    return result_dict, result


def initialize_or_load_results(logdir: str):
    """Load results from a previous job or initialize an empty results dictionary."""
    results = defaultdict(list)
    if os.path.exists(os.path.join(logdir, "results.json")):
        print("Results file already exists, loading previous results...")
        with open(os.path.join(logdir, "results.json"), "r", encoding="utf-8") as f:
            results.update(json.load(f))
    return results


class ProblemLogger:
    """Saves and loads the logged files for one problem in a single trial of one job"""

    def __init__(self, level: int, problem_id: int, logdir: str, trial: int):
        """Initialize the problem logger.

        Args:
            level (int): Optimization level.
            problem_id (int): Unique problem identifier.
            logdir (str): Directory for logging outputs.
            trial (int): Trial number for this evaluation.
        """
        self.level = level
        self.problem_id = problem_id
        self.logdir = logdir
        self.trial = trial
        self.gen_code_fn = f"generated_kernel_level_{self.level}_problem_{self.problem_id}_trial_{self.trial}.py"
        self.stdout_path_part = os.path.join(self.logdir, f"stdout_level_{level}_problem_{problem_id}_trial_{trial}")

        # for now, not using additional correctness tests for KernelBench
        self.tests_file = None

    def read_from_prior_run(self, fn) -> str | None:
        """Load prior stdout or gen from a prior trial."""
        prior_file_path = os.path.join(self.logdir, fn)
        if not os.path.exists(prior_file_path):
            if self.trial > 0:
                warnings.warn(f"Attention: Trial > 0 but could not find {fn}!")
            return None
        with open(prior_file_path, "r", encoding="utf-8") as f:
            stdout_or_kernel = f.read()
        return stdout_or_kernel

    def read_prior_stdout(self) -> str | None:
        return self.read_from_prior_run(
            f"stdout_level_{self.level}_problem_{self.problem_id}_trial_{self.trial-1}_best.txt"
        )

    def read_prior_gen_code(self):
        """Reads the generated code from the previous trial."""
        gen_code_fn = f"generated_kernel_level_{self.level}_problem_{self.problem_id}_trial_{self.trial-1}.py"
        return self.read_from_prior_run(gen_code_fn)

    def log_prompt_list(self, prompt_list: list[str]) -> None:
        """Write prompt to a file."""
        for i, prompt in enumerate(prompt_list):
            prompt_out_fn = f"prompt_level_{self.level}_problem_{self.problem_id}_trial_{self.trial}_v{i}.md"
            with open(os.path.join(self.logdir, prompt_out_fn), "w", encoding="utf-8") as f:
                f.write(prompt)

    def log_llm_messages(self, llm_messages: list) -> None:
        """Log whole conversation with LLM in one file."""
        with open(
            os.path.join(self.logdir, f"llm_messages_level_{self.level}_problem_{self.problem_id}.txt"),
            "w",
            encoding="utf-8",
        ) as ouf:
            for msg in llm_messages:
                if msg["role"] in ["user", "system"]:
                    ouf.write(f"User:\n\n{msg['content']}\n")
                else:
                    ouf.write(f"LLM:\n\n{msg['content']}\n")
                ouf.write("=" * 30)

    def log_eval_results(self, eval_results: list) -> None:
        """Write eval results to a file"""
        eval_out_path_part = f"eval_result_level_{self.level}_problem_{self.problem_id}_trial_{self.trial}"
        for version, exec_res in enumerate(eval_results):
            with open(os.path.join(self.logdir, eval_out_path_part + f"_v{version}.json"), "w", encoding="utf-8") as f:
                json.dump(exec_res.to_dict(), f, indent=4)

    def save_gen_kernel(self, custom_kernel: str) -> None:
        """Save the generated kernel code to a file."""
        with open(os.path.join(self.logdir, self.gen_code_fn), "w", encoding="utf-8") as f:
            f.write(custom_kernel)

    def save_stdout(self, console_output: str, version=None) -> None:
        """Save console log to a file."""
        fn_end = f"v{version}.txt" if version is not None else ".txt"
        with open(self.stdout_path_part + fn_end, "w", encoding="utf-8") as f:
            f.write(console_output)

    def save_gen_kernel_w_version(self, custom_kernel: str, language: str, version: int) -> str:
        """Save the generated kernel code with versioning."""
        fn_end = "cu" if language.lower() == "cuda" else "sycl"
        kernel_save_path = os.path.join(self.logdir, self.gen_code_fn[:-3] + f"_v{version}.{fn_end}")
        with open(kernel_save_path, "w", encoding="utf-8") as f:
            f.write(custom_kernel)
        return kernel_save_path

    def save_diffs(self, custom_kernel_out_list) -> None:
        """Save the diffs between the generated kernel and the previous one."""
        for i, kernel in enumerate(custom_kernel_out_list):
            diff_out_path = os.path.join(
                self.logdir, f"diff_level_{self.level}_problem_{self.problem_id}_trial_{self.trial}_v{i}.txt"
            )
            with open(diff_out_path, "w", encoding="utf-8") as ouf:
                ouf.write(kernel)

    def load_kernel_from_other_run(self, run_path: str) -> tuple[str, str]:
        """Find the best generated kernel for this problem in run_path"""
        warnings.warn(
            "load_kernel_from_other_run is deprecated and will be removed in future versions. ",
            DeprecationWarning,
            stacklevel=2,
        )
        if not os.path.isdir(run_path):
            with open(run_path, "r", encoding="utf-8") as file:
                kernel_code = file.read()
            stdout_path = run_path.replace("generated_kernel", "stdout").split(".")[0] + ".txt"
            eval_log = None
            if os.path.exists(stdout_path):
                with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                    eval_log = f.read()
            # prepare code to return
            wrapped_kernel_code = f"```\n{kernel_code}\n```" if "```" not in kernel_code else kernel_code
            return wrapped_kernel_code, eval_log

        result_fp = os.path.join(run_path, "results.json")
        assert os.path.exists(result_fp), f"kernels_iter_0_path requires results.json, not found at {run_path}"
        with open(result_fp, "r", encoding="utf-8") as f:
            results_prior_run = json.load(f)
        if str(self.problem_id) not in results_prior_run:
            raise ValueError(f"Problem ID {self.problem_id} not found in results of prior job at {run_path}")
        results_per_trial = results_prior_run[str(self.problem_id)]

        # select idx with min runtime or max score
        runtimes = np.array([res["runtime"] if res["runtime"] > 0 else np.nan for res in results_per_trial])
        if np.all(np.isnan(runtimes)):
            scores = [res["perf_score"] for res in results_per_trial]
            best_idx = np.argmax(scores)
        else:
            best_idx = np.nanargmin(runtimes)
        best_trial = results_per_trial[best_idx]["trial"]

        # now load the kernel code from the best trial
        gen_code_fn = f"generated_kernel_level_{self.level}_problem_{self.problem_id}_trial_{best_trial}.py"
        stdout_fn = f"stdout_level_{self.level}_problem_{self.problem_id}_trial_{best_trial}.txt"
        gen_code_fp, stdout_fp = os.path.join(run_path, gen_code_fn), os.path.join(run_path, stdout_fn)
        if not os.path.exists(stdout_fp):
            stdout_fp = os.path.join(run_path, stdout_fn[:-4] + "_best.txt")
            if not os.path.exists(stdout_fp):
                raise FileNotFoundError(f"Stdout file {stdout_fn} not found in {run_path}")
        if not os.path.exists(gen_code_fp):
            raise FileNotFoundError(f"Generated kernel file {gen_code_fn} not found in {run_path}")

        print("Loading previous solution from", gen_code_fp)
        with open(gen_code_fp, "r", encoding="utf-8") as f:
            # old version: f"level_{config.level}_problem_{problem_id}_sample_0_kernel.py";
            # custom_kernel = f.read(); custom_kernel_out = f"```python {custom_kernel}```"
            custom_kernel_out = f.read()
        with open(stdout_fp, "r", encoding="utf-8") as f:
            stdout = f.read()
        return custom_kernel_out, stdout

    def log_result(self, kernel_exec_result: EvalResult, results: dict, save: bool = True) -> dict:
        """Log the result of the kernel execution to a JSON file."""
        res_dict = {"trial": self.trial, "status": kernel_exec_result.get_status()}
        res_dict.update(kernel_exec_result.to_dict())

        # remove profiler data to avoid blowing up the results file
        if "profiler_data" in res_dict:
            del res_dict["profiler_data"]
            for _, templated in res_dict["template_results"].items():
                del templated["profiler_data"]
        if res_dict.get("metadata").get("eval_worker_info"):
            del res_dict["metadata"]["eval_worker_info"]
        if res_dict.get("metadata").get("compile_worker_info"):
            del res_dict["metadata"]["compile_worker_info"]

        # save results
        if save:
            # add to results
            results[self.problem_id].append(res_dict)
            with open(os.path.join(self.logdir, "results.json"), "w", encoding="utf-8") as ouf:
                json.dump(results, ouf, indent=4)
        return res_dict
