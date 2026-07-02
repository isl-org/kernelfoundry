"""Utilities for logging task-related messages to files and console."""


def log_task_msg(error_log_file, msg: str, verbose: bool = False):
    """Log a message to the specified error log file."""
    if error_log_file is not None:
        with open(error_log_file, "a") as f:
            f.write(msg + "\n")
        if verbose:
            print(msg)
    else:
        print(msg)
