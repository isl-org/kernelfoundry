"""Caches reference benchmark results in the database so repeated `build_and_test`/`submit_task`
invocations can reuse a reference result instead of rebuilding/rebenchmarking it every time.
"""

import logging
from dataclasses import asdict

from kernelfoundry.eval_pipeline.task import BuildResult, Task, TestResult
import kernelfoundry.eval_pipeline.database as db


def _reference_archs(config) -> list[str]:
    gpu_arch = config.get("reference_gpu_arch", config.get("gpu_arch"))
    return [arch.strip() for arch in gpu_arch.split(",")]


def load_cached_reference(task: Task, config) -> tuple[TestResult, BuildResult | None] | None:
    """Looks up a cached reference benchmark result for this task's reference content + gpu_arch(es).

    Args:
        task (Task): Task whose reference-relevant files are hashed to form the cache key.
        config: Execution config (dict or DictConfig), used to read `use_cached_reference` and
            the gpu_arch(es) to key the lookup on.

    Returns:
        tuple[TestResult, BuildResult | None] | None: The cached (test_result_reference,
            build_result_reference), or None if caching is disabled, any requested architecture is
            missing from the cache, or the lookup failed.
    """
    if not config.get("use_cached_reference", True):
        return None
    archs = _reference_archs(config)
    try:
        task_hash = task.reference_hash()
        rows = {}
        for arch in archs:
            row = db.get_reference_performance(task_hash, arch)
            if row is None:
                return None  # any missing arch is a full cache miss -- all archs are re-benchmarked together
            rows[arch] = row

        per_arch_test_results = {arch: TestResult.decode(row.test_result_reference) for arch, row in rows.items()}
        if any(result is None for result in per_arch_test_results.values()):
            return None

        test_result_reference = (
            next(iter(per_arch_test_results.values())) if len(archs) == 1 else TestResult.merge(per_arch_test_results)
        )
        build_result_reference = BuildResult.decode(next(iter(rows.values())).build_result_reference)
        logging.info(f"Reusing cached reference result for task_hash={task_hash[:12]}…, gpu_arch={archs}")
        if config.get("profiler_reference") and not test_result_reference.trace_results:
            # trace_results is empty iff no profiler ran for any arch when this entry was cached
            logging.warning(
                f"profiler_reference={config.get('profiler_reference')!r} is set, but the cached reference "
                f"result for task_hash={task_hash[:12]}… has no profiler data (it was cached before "
                "profiling was enabled). Set use_cached_reference: false to force a fresh, profiled "
                "reference run."
            )
        return test_result_reference, build_result_reference
    except Exception as e:
        logging.warning(f"Failed to load cached reference result, falling back to a fresh run. Error: {e}")
        return None


def store_reference(
    task: Task,
    config,
    test_result_reference: TestResult | None,
    build_result_reference: BuildResult | None,
    job_id: int | None = None,
) -> None:
    """Persists a freshly-computed reference result so later invocations can reuse it.

    One row is stored per individual gpu_arch (see module docstring). The build result, which isn't
    arch-specific, is duplicated across each arch's row.

    Args:
        task (Task): Task whose reference-relevant files are hashed to form the cache key.
        config: Execution config (dict or DictConfig), used to read `use_cached_reference` and
            the gpu_arch(es) to key the entry on.
        test_result_reference (TestResult | None): Freshly-computed reference test result. No-op
            if None (nothing to cache).
        build_result_reference (BuildResult | None): Freshly-computed reference build result, if any.
        job_id (int | None): Job that produced this result, stored for traceability.
    """
    if not config.get("use_cached_reference", True) or test_result_reference is None:
        return

    archs = _reference_archs(config)
    try:
        task_hash = task.reference_hash()
        if len(archs) == 1:
            per_arch_test_results = {archs[0]: test_result_reference}
        else:
            per_arch_test_results = test_result_reference.results_per_gpu or {}
            missing = set(archs) - set(per_arch_test_results)
            if missing:
                logging.warning(f"Cannot cache reference result: no per-arch result for {missing}")
                return

        encoded_build_result = build_result_reference.encode() if build_result_reference is not None else None
        task_name = config.get("task_name")
        for arch in archs:
            saved = db.save_reference_performance(
                task_name=task_name,
                task_hash=task_hash,
                gpu_arch=arch,
                build_result_reference=encoded_build_result,
                test_result_reference=asdict(per_arch_test_results[arch]),
                job_id=job_id,
            )
            if saved:
                logging.info(f"Cached reference result for task_hash={task_hash[:12]}…, gpu_arch={arch}")
    except Exception as e:
        logging.warning(f"Failed to store reference result in cache. Error: {e}")
