"""
Celery application configuration for the kernelfoundry evaluation pipeline.

Configures a Celery worker app with RabbitMQ as broker and either PostgreSQL
or RPC as the result backend. Connection parameters are read from environment
variables with sensible defaults for local development.

Environment variables:
    - RABBITMQ_IP: RabbitMQ host (default: localhost)
    - RABBITMQ_USERNAME: RabbitMQ username (default: guest)
    - RABBITMQ_PASSWORD: RabbitMQ password (default: guest)
    - QUEUE_BACKEND_TYPE: Set to ``postgresql`` to use Postgres as result backend
    - QUEUE_BACKEND_IP: Postgres host (default: localhost)
    - QUEUE_BACKEND_USERNAME: Postgres username (default: admin)
    - QUEUE_BACKEND_PASSWORD: Postgres password (default: admin)
    - QUEUE_BACKEND_DB_NAME: Postgres database name (default: queue_metadata)
"""

import sys
import os

if "autoroot" not in sys.modules:
    from pyrootutils import setup_root

    root = setup_root(
        search_from=os.getcwd(),
        indicator=".project-root",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False,
    )
    sys.path.insert(0, f"{root}/kernelfoundry")

from celery import Celery
import os

RABBITMQ_IP = os.environ.get("RABBITMQ_IP", "localhost")
RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME", "guest")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "guest")
broker_url = f"pyamqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_IP}:5672//"

if os.environ.get("QUEUE_BACKEND_TYPE") == "postgresql":
    print("Using Postgres as Celery backend")
    POSTGRES_IP = os.environ.get("QUEUE_BACKEND_IP", "localhost")
    POSTGRES_USERNAME = os.environ.get("QUEUE_BACKEND_USERNAME", "admin")
    POSTGRES_PASSWORD = os.environ.get("QUEUE_BACKEND_PASSWORD", "admin")
    POSTGRES_DB_NAME = os.environ.get("QUEUE_BACKEND_DB_NAME", "queue_metadata")
    backend_url = f"db+postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_IP}/{POSTGRES_DB_NAME}"
else:
    backend_url = "rpc://"

celery_app = Celery("tasks", broker=broker_url, backend=backend_url)

celery_app.conf.update(
    task_serializer="msgpack",
    result_serializer="msgpack",
    accept_content=["msgpack", "json"],
)

# Enhanced Celery configuration for PostgreSQL backend with robust connection handling
if os.environ.get("QUEUE_BACKEND_TYPE") == "postgresql":
    backend_engine_options = {
        "pool_pre_ping": True,
        # Recycle connections before aggressive DB-side idle termination kicks in (default: 5 min)
        "pool_recycle": int(os.environ.get("QUEUE_BACKEND_POOL_RECYCLE_SECONDS", "300")),
        "pool_size": int(os.environ.get("QUEUE_BACKEND_POOL_SIZE", "10")),
    }
    celery_app.conf.update(
        database_engine_options=backend_engine_options,
        # Prevent long-lived stale sessions for SQLAlchemy result backend
        database_short_lived_sessions=True,
        # Retry transient backend I/O failures
        result_backend_always_retry=True,
        result_backend_max_retries=int(os.environ.get("QUEUE_BACKEND_RESULT_MAX_RETRIES", "5")),
        result_backend_base_sleep_between_retries_ms=int(os.environ.get("QUEUE_BACKEND_RETRY_SLEEP_MS", "100")),
        result_backend_max_sleep_between_retries_ms=int(os.environ.get("QUEUE_BACKEND_RETRY_MAX_SLEEP_MS", "2000")),
    )


celery_app.autodiscover_tasks(
    [
        "kernelfoundry.eval_pipeline.tasks.build_image",
        "kernelfoundry.eval_pipeline.tasks.pull_image",
        "kernelfoundry.eval_pipeline.tasks.build_custom_task",
        "kernelfoundry.eval_pipeline.tasks.test_custom_task",
        "kernelfoundry.eval_pipeline.tasks.test_sleep",
    ],
    force=True,
)
