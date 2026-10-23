"""Structured, injection-safe search helpers over the ``kernels`` table.

Shared by the KernelFoundry MCP server (local execution) and the ``gui`` app's REST API
(remote execution), so both modes run the exact same query and return identical results.
"""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.database.tables import Kernel

#: Max characters of output_code included in kernel_to_preview_dict before truncating.
SEARCH_PREVIEW_LEN = 300

#: Baseline floor: kernels below this speedup are not meaningfully better than native and
#: are not useful as examples, so it is always applied rather than exposed as a filter.
MIN_IMPROVE_OVER_NATIVE = 0.5

#: Exact-match filter fields -> Kernel column.
_EQUALITY_FIELDS = ("output_language", "task_name", "gpu_arch")

#: Substring filter fields (``<field>_contains`` -> Kernel column to ILIKE against).
_CONTAINS_FIELDS = {
    "task_name_contains": "task_name",
    "output_code_contains": "output_code",
    "gpu_arch_contains": "gpu_arch",  # to allow that gpu_arch is sometimes a comma-separated combo (e.g. "bmg,lnl")
}

DEFAULT_LIMIT = 20
MAX_LIMIT = 100


def _apply_filters(query, filters: dict):
    # Always-on baseline: only correct kernels with a meaningful speedup are useful examples.
    query = query.where(
        Kernel.status == "correct",
        Kernel.improve_over_native > MIN_IMPROVE_OVER_NATIVE,
        Kernel.improve_over_native < float("inf"),
    )

    for field in _EQUALITY_FIELDS:
        value = filters.get(field)
        if value is not None:
            query = query.where(func.lower(getattr(Kernel, field)) == value.lower())

    for contains_field, column_name in _CONTAINS_FIELDS.items():
        value = filters.get(contains_field)
        if value:
            query = query.where(getattr(Kernel, column_name).ilike(f"%{value}%"))

    output_code_terms = filters.get("output_code_terms") or []
    for term in output_code_terms:
        if term.strip():
            query = query.where(Kernel.output_code.ilike(f"%{term.strip()}%"))

    max_runtime = filters.get("max_runtime")
    if max_runtime is not None:
        query = query.where(Kernel.runtime <= max_runtime)

    min_speedup = filters.get("min_speedup")
    if min_speedup is not None:
        query = query.where(Kernel.improve_over_native >= min_speedup)

    max_speedup = filters.get("max_speedup")
    if max_speedup is not None:
        query = query.where(Kernel.improve_over_native <= max_speedup)

    return query


def search_kernels(filters: dict, limit: int = DEFAULT_LIMIT, offset: int = 0) -> tuple[list[Kernel], int]:
    """Search the ``kernels`` table with a fixed set of safe, parameterized filters.

    Always restricted to ``status == "correct"`` and ``improve_over_native >
    MIN_IMPROVE_OVER_NATIVE`` -- incorrect or non-improving kernels aren't useful examples.

    Results are diversified across ``task_name`` rather than just sorted by speedup: Each task's single
    best kernel is returned first (that first batch itself ordered by speedup, best task first),
    then each task's second-best, and so on.

    Args:
        filters: Optional keys — case-insensitive exact match: output_language, task_name,
            gpu_arch; substring match: task_name_contains, output_code_contains,
            gpu_arch_contains; all-term code match: output_code_terms; numeric range:
            max_runtime, min_speedup/max_speedup (further narrows above the baseline floor;
            ``speedup`` is the ratio stored in ``Kernel.improve_over_native``).
        limit: Max rows to return, clamped to [1, MAX_LIMIT].
        offset: Row offset for pagination.

    Returns:
        Tuple of (matching Kernel rows for this page, total matching row count).
    """
    limit = max(1, min(limit, MAX_LIMIT))

    with db.SessionRO() as session:
        count_query = _apply_filters(select(func.count(Kernel.id)), filters)
        total_count = session.execute(count_query).scalar() or 0

        # Rank each task's kernels by speedup so the outer query can pull rank 1 for every task
        # first, then rank 2, etc. -- computed in SQL so it scales to large match counts without
        # pulling every row into Python.
        rank_within_task = (
            func.row_number()
            .over(partition_by=Kernel.task_name, order_by=Kernel.improve_over_native.desc())
            .label("rank_within_task")
        )
        ranked = _apply_filters(select(Kernel, rank_within_task), filters).subquery()
        RankedKernel = aliased(Kernel, ranked)

        query = (
            select(RankedKernel)
            .order_by(ranked.c.rank_within_task.asc(), ranked.c.improve_over_native.desc())
            .limit(limit)
            .offset(max(0, offset))
        )
        rows = session.execute(query).scalars().all()

    return rows, total_count


def get_kernel(kernel_id: int) -> Kernel | None:
    """Fetch a single kernel row by primary key, or None if it doesn't exist."""
    with db.SessionRO() as session:
        return session.get(Kernel, kernel_id)


def _safe_float(value: float | None) -> float | None:
    """Return None for non-JSON-compliant floats (NaN, Inf)."""
    if value is None:
        return None
    try:
        if math.isnan(value) or math.isinf(value):
            return None
    except (TypeError, ValueError):
        return None
    return value


def kernel_to_preview_dict(kernel: Kernel) -> dict[str, Any]:
    """Compact search_kernels result entry: metadata plus a truncated code preview."""
    output_code = kernel.output_code or ""
    preview = output_code[:SEARCH_PREVIEW_LEN]
    if len(output_code) > SEARCH_PREVIEW_LEN:
        preview += f"...(truncated, {len(output_code)} chars total; call get_kernel_from_db for the full code)"
    return {
        "id": kernel.id,
        "uuid": kernel.uuid,
        "task_name": kernel.task_name,
        "job_name": kernel.job_name,
        "input_language": kernel.input_language,
        "output_language": kernel.output_language,
        "runtime": _safe_float(kernel.runtime),
        "speedup": _safe_float(kernel.improve_over_native),
        "gpu_arch": kernel.gpu_arch,
        "timestamp": kernel.timestamp.isoformat() if kernel.timestamp else None,
        "output_code_preview": preview,
    }


def kernel_to_full_dict(kernel: Kernel, include_profiler_data: bool = False) -> dict[str, Any]:
    """Full get_kernel result entry: every field an agent would plausibly want."""
    result = {
        "id": kernel.id,
        "uuid": kernel.uuid,
        "parent_uuid": kernel.parent_uuid,
        "task_id": kernel.task_id,
        "task_name": kernel.task_name,
        "job_name": kernel.job_name,
        "job_id": kernel.job_id,
        "input_language": kernel.input_language,
        "output_language": kernel.output_language,
        "output_code": kernel.output_code,
        "status": kernel.status,
        "trial": kernel.trial,
        "version": kernel.version,
        "runtime": _safe_float(kernel.runtime),
        "runtime_stats": kernel.runtime_stats,
        "speedup": _safe_float(kernel.improve_over_native),
        "score": kernel.score,
        "eval_log": kernel.eval_log,
        "gpu_arch": kernel.gpu_arch,
        "language_model": kernel.language_model,
        "optimization_profile": kernel.optimization_profile,
        "timestamp": kernel.timestamp.isoformat() if kernel.timestamp else None,
    }
    if include_profiler_data:
        result["profiler_data"] = kernel.profiler_data
        result["profiler_data_reference"] = kernel.profiler_data_reference
        result["profiler_data_detail"] = kernel.profiler_data_detail
        result["profiler_data_detail_reference"] = kernel.profiler_data_detail_reference
    return result
