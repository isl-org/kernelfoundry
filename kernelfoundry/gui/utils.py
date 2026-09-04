"""Utility functions for the NiceGUI frontend, including database queries for jobs, kernels, and tasks."""

import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import desc, func, select

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.database.tables import Job, JobLog, Kernel, Task


def normalize_runtime_stats(runtime_stats: dict, gpu_arch: Optional[str] = None) -> dict:
    """Normalize the three legacy shapes of `Kernel.runtime_stats` into one.

    The database still contains rows in each of these shapes:
      - {gpu: {benchmark: {mean, std, speedup, ...}}}  (current format)
      - {benchmark: {mean, std, speedup, ...}}          (missing the GPU level)
      - {mean, std, ...}                                 (flat, single benchmark/gpu)

    Returns the current, 3-level shape in all cases; `{}` if `runtime_stats` is
    empty or unrecognized. `gpu_arch` (kernel.gpu_arch, possibly comma-separated)
    fills in the GPU level for the two older shapes; falls back to "unknown".
    """
    if not runtime_stats:
        return {}

    gpu = gpu_arch.split(",")[0] if gpu_arch else "unknown"

    # Flat format: the top-level dict is itself a single metrics dict.
    if "mean" in runtime_stats or "speedup" in runtime_stats:
        return {gpu: {"test1": runtime_stats}}

    first_value = next(iter(runtime_stats.values()), None)
    assert isinstance(first_value, dict), "Expected runtime stats entry"
    if not first_value:
        # Already 3-level, just with no benchmarks recorded for that GPU yet.
        return runtime_stats

    inner_value = next(iter(first_value.values()))
    if isinstance(inner_value, dict):
        # Already 3-level: {gpu: {benchmark: {metrics}}}.
        return runtime_stats

    # 2-level format: {benchmark: {metrics}} - missing the GPU level.
    return {gpu: runtime_stats}


def _get_job_sort_column(sort_by: Optional[str]):
    """Get the database column to sort jobs by."""
    if not sort_by:
        return Job.created_at
    return getattr(Job, sort_by, Job.created_at)


def get_jobs(
    statuses: list[str] | None = None,
    offset: int = 0,
    limit: int = 10,
    sort_by: Optional[str] = None,
    descending: bool = True,
) -> list[Job]:
    """Retrieve paginated jobs with optional filtering by status and sorting."""
    stmt = select(Job).where(Job.archived_at.is_(None))
    if statuses:
        stmt = stmt.where(Job.status.in_(statuses))

    sort_column = _get_job_sort_column(sort_by)
    stmt = stmt.order_by(desc(sort_column) if descending else sort_column)
    stmt = stmt.offset(offset).limit(limit)

    with db.SessionRO() as session:
        return session.execute(stmt).scalars().all()


def get_total_jobs(statuses: list[str] | None = None) -> int:
    """Count total non-archived jobs, optionally filtered by status."""
    stmt = select(func.count(Job.id)).where(Job.archived_at.is_(None))
    if statuses:
        stmt = stmt.where(Job.status.in_(statuses))

    with db.SessionRO() as session:
        result = session.execute(stmt).scalar()
        return result or 0


def get_jobs_by_user(
    user_id: str,
    statuses: list[str] | None = None,
    offset: int = 0,
    limit: int = 10,
    sort_by: Optional[str] = None,
    descending: bool = True,
) -> list[Job]:
    """Backward-compatible wrapper: returns all jobs when user_id is empty."""
    stmt = select(Job).where(Job.archived_at.is_(None))
    if user_id:
        stmt = stmt.where(Job.task_origin == user_id)
    if statuses:
        stmt = stmt.where(Job.status.in_(statuses))

    sort_column = _get_job_sort_column(sort_by)
    stmt = stmt.order_by(desc(sort_column) if descending else sort_column)
    stmt = stmt.offset(offset).limit(limit)

    with db.SessionRO() as session:
        return session.execute(stmt).scalars().all()


def get_total_jobs_by_user(user_id: str, statuses: list[str] | None = None) -> int:
    """Count total jobs for a user, optionally filtered by status."""
    stmt = select(func.count(Job.id)).where(Job.archived_at.is_(None))
    if user_id:
        stmt = stmt.where(Job.task_origin == user_id)
    if statuses:
        stmt = stmt.where(Job.status.in_(statuses))

    with db.SessionRO() as session:
        result = session.execute(stmt).scalar()
        return result or 0


def archive_job_by_id(job_id: int, requesting_username: str | None = None) -> bool | None | str:
    """Archive a job by marking it with a timestamp (soft delete)."""
    try:
        with db.SessionInsert() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job is None:
                return None
            if requesting_username is not None and job.task_origin != requesting_username:
                return "forbidden"
            if job.archived_at is not None:
                return True
            job.archived_at = datetime.now(timezone.utc)
            session.commit()
            return True
    except Exception as e:
        print(f"Warning: Could not archive job {job_id}: {e}")
    return False


def cancel_job_by_id(job_id: int, requesting_username: str | None = None) -> bool | None | str:
    """
    Set a job's status to CANCELED.

    Args:
        job_id: The ID of the job to cancel.
        requesting_username: If provided, only cancel if job's task_origin matches this username.

    Returns:
        True if the job was found and canceled (or already canceled),
        None if the job was not found,
        "forbidden" if requesting_username does not own the job,
        or False on a database error.
    """
    try:
        with db.SessionInsert() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job is None:
                return None
            if requesting_username is not None and job.task_origin != requesting_username:
                return "forbidden"
            if job.status == "CANCELED":
                return True
            job.status = "CANCELED"
            if job.finished_at is None:
                # add finished at
                job.finished_at = datetime.now(timezone.utc)
            session.commit()
            return True
    except Exception as e:
        print(f"Warning: Could not cancel job {job_id}: {e}")
    return False


def get_job_is_validate(job_id: int) -> bool:
    """Check if a job has status VALIDATED."""
    with db.SessionRO() as session:
        job = session.query(Job.status).filter_by(id=job_id).first()
    return job is not None and job.status == "VALIDATED"


def get_job_logs_by_job_id(job_id: int) -> list[JobLog]:
    """Retrieve all logs for a job in chronological order."""
    stmt = select(JobLog).where(JobLog.job_id == job_id).order_by(JobLog.timestamp.asc())
    with db.SessionRO() as session:
        return session.execute(stmt).scalars().all()


def get_id_by_uuid(uuid: str) -> int | None:
    """Get the kernel ID from its UUID."""
    with db.SessionRO() as session:
        result = session.execute(select(Kernel.id).where(Kernel.uuid == uuid))
        row = result.fetchone()
        return row[0] if row else None


def get_kernels_by_op_and_run(op: str, name: str, user_id: Optional[str] = None):
    """Retrieve kernels for a specific operation and run name."""
    query = select(
        Kernel.id,
        Kernel.uuid,
        Kernel.status,
        Kernel.trial,
        Kernel.version,
        Kernel.runtime,
        Kernel.improve_over_native,
        Kernel.eval_log,
        Kernel.parent_uuid,
        Kernel.timestamp,
        Kernel.runtime_stats,
        Kernel.agent_session_id,
        Kernel.gpu_arch,
    ).where(Kernel.task_name == op, Kernel.job_name == name)

    # Optional user filter for backwards compatibility.
    if user_id:
        query = query.join(Job, Kernel.job_id == Job.id).where(Job.task_origin == user_id)

    query = query.order_by(Kernel.trial.asc())
    with db.SessionRO() as session:
        return session.execute(query).all()


def get_op_names(user_id: Optional[str] = None) -> list[str]:
    """Retrieve all unique operation/task names, optionally filtered by user."""
    query = select(Kernel.task_name).where(Kernel.task_name.is_not(None)).distinct().order_by(Kernel.task_name.asc())
    if user_id:
        query = query.join(Job, Kernel.job_id == Job.id).where(Job.task_origin == user_id)

    with db.SessionRO() as session:
        return [row[0] for row in session.execute(query).fetchall() if row[0] is not None]


def get_jobs_by_op(task_name: str, user_id: Optional[str] = None) -> list[str]:
    """Retrieve all unique job names for a given operation/task."""
    query = (
        select(Kernel.job_name)
        .where(Kernel.task_name == task_name, Kernel.job_name.is_not(None))
        .distinct()
        .order_by(Kernel.job_name.asc())
    )
    if user_id:
        query = query.join(Job, Kernel.job_id == Job.id).where(Job.task_origin == user_id)

    with db.SessionRO() as session:
        return [row[0] for row in session.execute(query).fetchall() if row[0] is not None]


def get_kernels_by_job_id(job_id: int):
    """Retrieve all kernels for a specific job."""
    query = select(
        Kernel.id,
        Kernel.uuid,
        Kernel.status,
        Kernel.trial,
        Kernel.version,
        Kernel.runtime,
        Kernel.eval_log,
        Kernel.parent_uuid,
        Kernel.timestamp,
        Kernel.score,
        Kernel.improve_over_native,
        Kernel.runtime_stats,
        Kernel.output_code,
        Kernel.agent_session_id,
        Kernel.gpu_arch,
    ).where(Kernel.job_id == job_id)

    with db.SessionRO() as session:
        return session.execute(query).all()


def get_kernel_by_id(kernel_id: int) -> Kernel | None:
    """Retrieve a kernel by its ID."""
    with db.SessionRO() as session:
        return session.get(Kernel, kernel_id)


def get_job_by_id(job_id: int) -> Job | None:
    """Retrieve a job by its ID."""
    with db.SessionRO() as session:
        return session.get(Job, job_id)


def get_task_by_id(task_id: str) -> Task | None:
    """Retrieve a task by its ID."""
    with db.SessionRO() as session:
        return session.get(Task, task_id)


def get_profiler_data_reference_by_job_id(job_id: int) -> tuple:
    """Retrieve reference profiler data and worker info for a job."""
    stmt = select(Kernel.profiler_data_reference, Kernel.eval_worker_info).where(
        Kernel.job_id == job_id, Kernel.profiler_data_reference.is_not(None)
    )
    with db.SessionRO() as session:
        row = session.execute(stmt).fetchone()
        return row if row else (None, None)
