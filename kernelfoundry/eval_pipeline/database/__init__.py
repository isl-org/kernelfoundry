"""Database initialization, utility functions and table definitions."""

from sqlalchemy import create_engine as _create_engine, inspect as _inspect, text as _text
from sqlalchemy.orm import sessionmaker as _sessionmaker
from kernelfoundry.eval_pipeline.database.tables import Base, Kernel, Task, Job, JobLog, Rag
import os
import sys
from urllib.parse import quote_plus
from pathlib import Path
from datetime import datetime, timezone
import logging

#: Statuses meaning the job is over. Reaching one implies a ``finished_at``; see update_job_status.
TERMINAL_JOB_STATUSES = ("COMPLETE", "FAIL", "CANCELED", "VALIDATED")
#: Statuses meaning the job is running right now.
ACTIVE_JOB_STATUSES = ("INIT", "RUN", "VALIDATING")
JOB_STATUSES = ACTIVE_JOB_STATUSES + TERMINAL_JOB_STATUSES

__all__ = [
    "init",
    "add_ignore_errors",
    "add_job",
    "update_job_status",
    "update_job_progress",
    "update_job_token_usage",
    "add_job_log",
    "add_job_log_batch",
    "Kernel",
    "Task",
    "Rag",
    "Job",
    "JobLog",
    "engine_readonly",
    "engine_insert",
    "SessionRO",
    "SessionInsert",
]

engine_readonly = None
engine_insert = None
SessionRO = None
SessionInsert = None


def _format_db_url(url: str | None) -> str | None:
    if url is None:
        return None
    try:
        return url.format_map(os.environ)
    except KeyError as e:
        print(f"Warning: Could not format database URL {url}.\nMissing environment variable: {e}")
        raise e


def _ensure_sqlite_parent_directory(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    if url.startswith("sqlite:///:memory:"):
        return

    # sqlite:///relative.db -> relative path
    # sqlite:////absolute/path.db -> absolute path
    db_path_str = url[len("sqlite:///") :]
    # Strip URL query/fragment parts (e.g. sqlite:///db.sqlite3?mode=ro).
    db_path_str = db_path_str.split("?", 1)[0].split("#", 1)[0]
    db_path = Path(db_path_str)
    if db_path.parent and str(db_path.parent) != "":
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _init_postgresql():
    """Initialize PostgreSQL readonly and insert-only database connections."""
    global engine_readonly, engine_insert, SessionRO, SessionInsert

    database_ip = os.environ.get("DB_IP")
    if "dbaas" in database_ip:
        logging.info("Set logging level to critical for celery+dbaas database (to avoid reconnect warnings).")
        logging.getLogger("celery.backends.database").setLevel(logging.CRITICAL)

    readonly_password = os.environ.get("DB_READONLY_PASSWORD")
    if not readonly_password:
        raise ValueError(
            "Missing required environment variable DB_READONLY_PASSWORD for kernel readonly database connection."
        )

    insertonly_password = os.environ.get("DB_INSERTONLY_PASSWORD")
    if not insertonly_password:
        raise ValueError(
            "Missing required environment variable DB_INSERTONLY_PASSWORD for kernel insert-only database connection."
        )

    database_port = int(os.environ.get("DATABASE_PORT", "5432"))
    database_name = os.environ.get("KERNELS_DB_NAME") or os.environ.get("DATABASE_NAME", "kernels")

    # Connection pool configuration to handle long-running tasks and server-side idle termination
    # pool_pre_ping: Test connections before using them to catch stale connections
    # pool_recycle: Recycle connections before idle timeout (default 300s = 5 min)
    # pool_size: Number of connections to keep in the pool
    # max_overflow: Maximum overflow connections beyond pool_size
    pool_config = {
        "pool_pre_ping": True,
        "pool_recycle": int(os.environ.get("DB_POOL_RECYCLE_SECONDS", "300")),
        "pool_size": int(os.environ.get("DB_POOL_SIZE", "10")),
        "max_overflow": int(os.environ.get("DB_MAX_OVERFLOW", "20")),
    }

    read_only_url = (
        f"postgresql+psycopg2://codegen_readonly_user:{quote_plus(readonly_password)}"
        f"@{database_ip}:{database_port}/{database_name}"
    )
    engine_readonly = _create_engine(read_only_url, **pool_config)
    SessionRO = _sessionmaker(engine_readonly)

    insert_only_url = (
        f"postgresql+psycopg2://codegen_insertonly_user:{quote_plus(insertonly_password)}"
        f"@{database_ip}:{database_port}/{database_name}"
    )
    engine_insert = _create_engine(insert_only_url, **pool_config)
    SessionInsert = _sessionmaker(engine_insert)


def _check_sqlite_schema(engine, url: str) -> None:
    """Fail loudly if an existing SQLite file predates the current schema.

    ``create_all`` only creates tables that don't exist yet; it never adds columns to ones
    that do, so an older database file otherwise causes a ``no such column`` error.
    """
    inspector = _inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        existing_columns = (
            {col["name"] for col in inspector.get_columns(table.name)} if table.name in existing_tables else set()
        )
        missing = [col.name for col in table.columns if col.name not in existing_columns]
        if missing:
            what = (
                f"table '{table.name}'"
                if table.name not in existing_tables
                else f"columns {missing} in table '{table.name}'"
            )
            raise RuntimeError(
                f"The SQLite database at {url!r} is missing {what}. It was likely created by an "
                "older version of KernelFoundry. Move or delete the file (or point "
                "paths.kernels_db_path at a new location) so "
                "KernelFoundry can create a fresh database with the current schema."
            )


def _init_sqlite(cfg):
    """Initialize SQLite database connection."""
    global engine_readonly, engine_insert, SessionRO, SessionInsert

    db_url = _format_db_url(cfg.paths.get("kernels_db_path", None))
    primary_url = db_url or "sqlite:///runs/kernels.sqlite3"

    _ensure_sqlite_parent_directory(primary_url)

    create_kwargs = {"connect_args": {"check_same_thread": False}}

    engine_insert = _create_engine(primary_url, **create_kwargs)
    SessionInsert = _sessionmaker(engine_insert)

    engine_readonly = _create_engine(primary_url, **create_kwargs)
    SessionRO = _sessionmaker(engine_readonly)

    # Bootstrap schema automatically for SQLite.
    Base.metadata.create_all(engine_insert)
    _check_sqlite_schema(engine_insert, primary_url)


def init(cfg):
    """Initialize database connections (PostgreSQL or SQLite based on DB_IP env var)."""
    if os.environ.get("DB_IP") is not None:
        _init_postgresql()
    else:
        _init_sqlite(cfg)


def add_ignore_errors(item):
    """Adds the item to the database, ignoring any errors to not stop the program flow."""
    try:
        with SessionInsert() as session:
            session.add(item)
            session.commit()
        return True
    except Exception as e:
        e_str = str(e).lower()
        if "unique constraint" in e_str and "task" in e_str:
            logging.info("Info: Task already in database (not inserting it again due to unique ID constraint).")
        else:
            logging.warning(f"Could not add item to database. Error: {e}")
    return False


def add_job(task_origin: str, status: str = "INIT") -> int | None:
    """Create and add a new Job to the database.

    Args:
        status (str): Initial status (default: "INIT")

    Returns:
        int | None: The job ID if successful, None otherwise
    """
    try:
        with SessionInsert() as session:
            job = Job(task_origin=task_origin, status=status)
            session.add(job)
            session.commit()
            # Access the ID before the session closes
            job_id = job.id
        return job_id
    except Exception as e:
        print(f"Warning: Could not add job to database. Error: {e}")
    return None


def update_job_status(
    job_id: int,
    status: str = None,
    started_at: datetime = None,
    finished_at: datetime = None,
    task_id: str = None,
    config: dict = None,
    progress: float = None,
) -> bool:
    """Update the status config, and timestamps of a job in the database.

    Args:
        job_id (int): The ID of the job to update
        status (str): New status (INIT, RUN, COMPLETE, FAIL)
        started_at (datetime, optional): Timestamp when job started
        finished_at (datetime, optional): Timestamp when job finished
        task_id (str, optional): The task ID to associate with this job
        config (dict, optional): Updated configuration to store in the database
        progress (float, optional): Updated progress value between 0.0 and 1.0

    Returns:
        bool: True if update was successful, False otherwise
    """
    assert status is None or status in JOB_STATUSES, f"Invalid status value {status}"
    try:
        with SessionInsert() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job:
                if job.status == "CANCELED":
                    logging.error(
                        f"Attempted to update job {job_id} which has been canceled. No updates will be applied."
                    )
                    sys.exit(1)
                if status is not None:
                    job.status = status
                if started_at is not None:
                    job.started_at = started_at
                if finished_at is not None:
                    job.finished_at = finished_at
                if task_id is not None:
                    job.task_id = task_id
                if config is not None:
                    job.config = config
                if progress is not None:
                    job.progress = max(0.0, min(1.0, progress))
                if status in TERMINAL_JOB_STATUSES and job.finished_at is None:
                    # add finished_at if the job has reached a terminal status but finished_at is not set
                    job.finished_at = finished_at or datetime.now(timezone.utc)
                if status == "COMPLETE" and job.finished_at is not None:
                    job.progress = 1.0
                session.commit()
                return True
    except Exception as e:
        print(f"Warning: Could not update job status: {e}")
    return False


def update_job_progress(job_id: int, progress: float) -> bool:
    """Update the progress of a job in the database.

    Args:
        job_id (int): The ID of the job to update
        progress (float): Progress value between 0.0 and 1.0

    Returns:
        bool: True if update was successful, False otherwise
    """
    return update_job_status(job_id, progress=progress)


def update_job_token_usage(
    job_id: int,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> bool:
    """Persist aggregated token usage for a job.

    Args:
        job_id (int): The ID of the job to update.
        input_tokens (int): Aggregated prompt/input tokens (excludes cache-read tokens).
        output_tokens (int): Aggregated completion/output tokens.
        cached_input_tokens (int): Aggregated cache-read input tokens. Defaults to 0 for
            providers with no caching concept.
    Returns:
        bool: True if update was successful, False otherwise.
    """
    try:
        with SessionInsert() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            if job is None:
                return False

            job.input_tokens = input_tokens
            job.output_tokens = output_tokens
            job.cached_input_tokens = cached_input_tokens
            session.commit()
            return True
    except Exception as e:
        print(f"Warning: Could not update job token usage: {e}")
    return False


def add_job_log(
    job_id: int,
    level: str,
    message: str,
    timestamp: datetime = None,
    hostname: str | None = None,
    kernel_uuid: str | None = None,
    agent_session_id: str | None = None,
) -> int | None:
    """Add a single log entry for a job.

    Args:
        job_id (int): The ID of the job this log belongs to
        level (str): Log level (INFO, DEBUG, WARNING, ERROR)
        message (str): The log message
        timestamp (datetime, optional): Custom timestamp (defaults to server time)
        hostname (str | None, optional): Hostname of the process that emitted the log entry
        kernel_uuid (str | None, optional): UUID of the kernel associated with the log entry
        agent_session_id (str | None, optional): Agent session identifier associated with the log entry

    Returns:
        int | None: The log entry ID if successful, None otherwise
    """
    try:
        with SessionInsert() as session:
            log_entry = JobLog(
                job_id=job_id,
                hostname=hostname,
                kernel_uuid=kernel_uuid,
                agent_session_id=agent_session_id,
                level=level,
                message=message,
            )
            if timestamp is not None:
                log_entry.timestamp = timestamp
            session.add(log_entry)
            session.commit()
            log_id = log_entry.id
        return log_id
    except Exception as e:
        print(f"Warning: Could not add job log to database. Error: {e}")
    return None


def add_job_log_batch(logs: list[dict]) -> bool:
    """Add multiple log entries in a single transaction for better performance.

    Args:
        logs (list[dict]): List of log dictionaries, each containing:
            - job_id (int): The ID of the job
            - hostname (str | None, optional): Hostname of the process that emitted the log entry
            - kernel_uuid (str | None, optional): UUID of the kernel associated with the log entry
            - agent_session_id (str | None, optional): Agent session identifier associated with the log entry
            - level (str): Log level (INFO, DEBUG, WARNING, ERROR)
            - message (str): The log message
            - timestamp (datetime, optional): Custom timestamp

    Returns:
        bool: True if all logs were added successfully, False otherwise

    Example::

        logs = [
            {"job_id": 1, "level": "INFO", "message": "Starting task"},
            {"job_id": 1, "level": "DEBUG", "message": "Processing step 1"},
            {"job_id": 1, "level": "ERROR", "message": "Failed at step 2"},
        ]
        add_job_log_batch(logs)
    """
    try:
        with SessionInsert() as session:
            for log_data in logs:
                log_entry = JobLog(
                    job_id=log_data["job_id"],
                    hostname=log_data.get("hostname"),
                    kernel_uuid=log_data.get("kernel_uuid"),
                    agent_session_id=log_data.get("agent_session_id"),
                    level=log_data["level"],
                    message=log_data["message"],
                    extra=log_data.get("extra") or None,
                )
                if "timestamp" in log_data and log_data["timestamp"] is not None:
                    log_entry.timestamp = log_data["timestamp"]
                session.add(log_entry)
            session.commit()
        return True
    except Exception as e:
        print(f"Warning: Could not add job logs batch to database. Error: {e}")
    return False
