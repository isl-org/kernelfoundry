import numpy as np
import warnings
import logging
from scipy.stats import gmean

from kernelfoundry.eval_pipeline import database as db
from kernelfoundry.eval_pipeline.database.tables import BaselineTime
from kernelfoundry.eval_pipeline.utils.gpu_specs import GPU_ARCH_TO_BL_TIME
from kernelfoundry.algorithm.schemas import EvalResult


def compute_runtime_improvement(exec_result, level: int, task_name: str, gpu_arch: str):
    """
    Add improvement of kernel over baseline time to exec result, using ORM queries.

    Args:
        exec_result: Execution result containing runtime but missing runtime_improvement
        level: Level of the operation (e.g., 1)
        task_name: Operation name (with .py!)
        gpu_arch: gpu arch string
        db: Database object with SessionRO() context manager
    """
    assert db.SessionRO is not None, "First need to initialize database!"

    if exec_result.runtime < 0:
        return exec_result

    level_str = f"level{level}"

    if isinstance(gpu_arch, list) and len(gpu_arch) > 0:
        warnings.warn("Currently only single gpu_arch supported for computing runtime improvement!")
    arch = gpu_arch if isinstance(gpu_arch, str) else gpu_arch[0]
    if arch not in GPU_ARCH_TO_BL_TIME:
        warnings.warn(
            f"No baseline times recorded for architecture '{arch}'; runtime_improvement will not be "
            f"computed. Recorded architectures: {', '.join(sorted(GPU_ARCH_TO_BL_TIME))}."
        )
        return exec_result
    platform = GPU_ARCH_TO_BL_TIME[arch]

    with db.SessionRO() as session:
        # Query for eager baseline
        bl_eager = (
            session.query(BaselineTime)
            .filter_by(level=level_str, task_name=task_name, platform=platform, backend="torch")
            .first()
        )

        if bl_eager and bl_eager.mean and bl_eager.mean > 0:
            exec_result.runtime_improvement = bl_eager.mean / exec_result.runtime
        else:
            warnings.warn(
                f"Baseline time not found for level {level}, op {task_name}, platform {platform}, backend eager. Cannot compute runtime improvement."
            )

        # Query for torch_compile baseline
        bl_compile = (
            session.query(BaselineTime)
            .filter_by(
                level=level_str, task_name=task_name, platform=platform, backend="torch_compile_inductor_default"
            )
            .first()
        )
        if bl_compile and bl_compile.mean and bl_compile.mean > 0:
            exec_result.improve_over_compile = bl_compile.mean / exec_result.runtime

    return exec_result


def select_best_solution(eval_results: list[EvalResult]) -> int:
    """
    Select the best kernel based on composite fitness score.

    Uses EvalResult.compute_performance_score to ensure consistent scoring across the system.

    Args:
        eval_results: List of EvalResult
    Returns:
        int: index of best solution
    """
    scores = [EvalResult.compute_performance_score(res) for res in eval_results]
    best_score_ind = np.argmax(scores)
    return best_score_ind


def combine_gpu_arch_results(exec_result_dict: dict[str, EvalResult], combined_eval_log: str) -> EvalResult:
    """Combine results from different gpu archs by taking the worst correctness and average runtime."""
    # combine metadata
    all_meta = {gpu_key: res.metadata for gpu_key, res in exec_result_dict.items()}
    exec_result_list = list(exec_result_dict.values())
    # If one doesn't compile, return compile error
    compilation = [res.compiled for res in exec_result_list]
    if not all(compilation):
        return EvalResult(compiled=False, perf_score=1, eval_log=combined_eval_log, metadata=all_meta)
    # If one is incorrect, return incorrect
    correctness = [res.correctness for res in exec_result_list]
    perf_score = [res.perf_score for res in exec_result_list]
    if not all(correctness):
        return EvalResult(
            compiled=True, correctness=False, perf_score=min(perf_score), eval_log=combined_eval_log, metadata=all_meta
        )
    # If all compiled and correct, return combined runtime and improvement
    assert all([res.runtime > 0 for res in exec_result_list]), "All runtimes should be > 0 if none is incorrect"
    # mean runtime
    mean_runtime = float(np.mean([res.runtime for res in exec_result_list]))
    # store all runtime stats by adding a new level with the gpu arch
    all_runtime_stats = {gpu_key: res.runtime_stats for gpu_key, res in exec_result_dict.items()}
    # compute geometric mean for speedup - collect all speedups from runtime stats
    speedups = [
        all_runtime_stats[gpu][shape]["speedup"] for gpu in all_runtime_stats for shape in all_runtime_stats[gpu]
    ]
    # other option: just take geom mean over runtime improvements which are already aggregated over benchmarks
    # speedups = [res.runtime_improvement for res in exec_result_list if res.runtime_improvement > 0]
    mean_speedup = gmean(speedups) if len(speedups) > 0 else -1
    # profiler data
    first_profiler_data = next(iter(exec_result_dict.values())).profiler_data
    if len(exec_result_dict) > 1:
        logging.warning("Storing profiler data only for the first GPU, visualizing for more GPUs is not supported yet.")
    return EvalResult(
        compiled=True,
        correctness=True,
        runtime=mean_runtime,
        runtime_stats=all_runtime_stats,
        perf_score=5,
        eval_log=combined_eval_log,
        runtime_improvement=mean_speedup,
        profiler_data=first_profiler_data,
        metadata=all_meta,
    )


def combine_template_results(exec_result_list: list, parameter_options: list, return_index: bool = False):
    """Take the results for each parameter option and combine them into one result"""
    assert len(exec_result_list) == len(parameter_options), f"{len(exec_result_list)} != {len(parameter_options)}"
    index_of_best = select_best_solution(exec_result_list)
    base_result = exec_result_list[index_of_best]
    base_result.metadata["template_parameters"] = parameter_options[index_of_best]
    # add all other results to the template_results field of base_result
    for param, res in zip(parameter_options, exec_result_list):
        res_dict = res.to_dict()
        # remove unnecessary parts of result
        for k in ["improve_over_compile", "runtime_improvement", "template_results", "metadata"]:
            res_dict.pop(k, None)
        base_result.template_results[str(param)] = res_dict
    if return_index:
        return base_result, index_of_best
    return base_result


def geometric_mean_speed_ratio_correct_only(
    is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int
) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct)  # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0


def geometric_mean_speed_ratio_correct_and_faster_only(
    is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int
) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0


def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0


def get_timing_stats(elapsed_times: list[float], device=None, device_name: str = "unknown") -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "median": float(f"{np.median(elapsed_times):.3g}"),
        "speedup": -1,  # can be filled later when reference time is available
        "ref_speed": -1,  # can be filled later when reference time is available
        "num_trials": len(elapsed_times),
    }

    if device is not None:
        stats["hardware"] = device_name
        stats["device"] = str(device)  # for debugging

    return stats
