import logging
import socket
from datetime import datetime, timezone
import kernelfoundry.eval_pipeline.database as db


class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that writes log records to the database."""

    def __init__(self, job_id: int, level=logging.NOTSET):
        super().__init__(level)
        self.job_id = job_id
        self.log_buffer = []
        self.buffer_size = 10  # Batch size for database writes
        self.main_thread_hostname = socket.gethostname()

    def emit(self, record):
        """Emit a log record to the database."""
        try:
            level_name = record.levelname

            record_data = record.__dict__.get("data", None)
            hostname = (
                record_data.get("worker_info", {}).get("hostname", self.main_thread_hostname)
                if record_data
                else self.main_thread_hostname
            )

            log_entry = {
                "job_id": self.job_id,
                "hostname": hostname,
                "kernel_uuid": record.__dict__.get("kernel_uuid", None),
                "agent_session_id": record.__dict__.get("agent_session_id", None),
                "level": level_name,
                "message": self.format(record),
                "extra": record_data or None,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc),
            }

            self.log_buffer.append(log_entry)

            # Flush buffer if it reaches the batch size
            if len(self.log_buffer) >= self.buffer_size:
                self.flush()

        except Exception:
            self.handleError(record)

    def flush(self):
        """Flush the log buffer to the database."""
        if self.log_buffer:
            db.add_job_log_batch(self.log_buffer)
            self.log_buffer = []

    def close(self):
        """Close the handler and flush any remaining logs."""
        self.flush()
        super().close()
