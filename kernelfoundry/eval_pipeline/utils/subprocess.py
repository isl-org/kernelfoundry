from typing import Callable
import subprocess
import functools
import asyncio
import re
import io
import os
import signal
from contextlib import suppress

__all__ = [
    "robust_subprocess_run",
    "force_terminate",
]


class ForwardToStreamReaderProtocol(asyncio.subprocess.SubprocessStreamProtocol):
    def __init__(self, reader, limit, loop):
        """A protocol that forwards data received from the child process's
        stdout and stderr to a given StreamReader.
        Args:
            reader: An asyncio StreamReader instance to forward data to.
            limit: The buffer limit.
            loop: The event loop to use.
        """
        super().__init__(limit=limit, loop=loop)
        self._reader = reader
        self.stdout_buffer = io.BytesIO()
        self.stderr_buffer = io.BytesIO()

    def pipe_data_received(self, fd, data):
        """Called when the child process writes data into its stdout
        or stderr pipe.
        Args:
            fd: The file descriptor the data was received on.
            data: The bytes data received.
        """
        super().pipe_data_received(fd, data)
        if fd == 1 or fd == 2:
            self._reader.feed_data(data)
            if fd == 1:
                self.stdout_buffer.write(data)
            elif fd == 2:
                self.stderr_buffer.write(data)

    def pipe_connection_lost(self, fd, exc):
        """Called when one of the pipes communicating with the child
        process is closed.
        Args:
            fd: The file descriptor the connection was lost on.
            exc: An exception if the connection was lost due to an error or None.
        """
        super().pipe_connection_lost(fd, exc)
        if fd == 1 or fd == 2:
            if exc:
                self._reader.set_exception(exc)
            else:
                self._reader.feed_eof()


def create_ForwardToStreamReaderProtocol(reader, loop, limit=asyncio.streams._DEFAULT_LIMIT):
    """Creates a ForwardToStreamReaderProtocol instance.

    Args:
        reader: An asyncio StreamReader instance to forward data to.
        loop: The event loop to use.
        limit: The buffer limit.

    Returns:
        An instance of ForwardToStreamReaderProtocol.
    """
    return ForwardToStreamReaderProtocol(reader=reader, limit=limit, loop=loop)


def _signal_proc_or_group(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """Send a signal to the process group when available, otherwise to the process."""
    if proc.pid is not None:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), sig)
            return

    with suppress(ProcessLookupError):
        if sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()


async def force_terminate(proc: asyncio.subprocess.Process, wait_after_terminate: float = 2.0):
    """Helper function for forcefully terminating a subprocess.
    Args:
        proc: The subprocess to terminate.
        wait_after_terminate: Time to wait after sending terminate signal before killing the process.
    """
    if proc.returncode is not None:
        return

    # Prefer signaling the full process group so shell-wrapped subprocess trees are terminated.
    _signal_proc_or_group(proc, signal.SIGTERM)

    try:
        await asyncio.wait_for(proc.wait(), timeout=wait_after_terminate)
    except TimeoutError:
        if proc.returncode is not None:
            return
        _signal_proc_or_group(proc, signal.SIGKILL)
        with suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=2.0)


class PytestEndMonitor:
    def __call__(self, line: str) -> tuple[bool, float, str]:
        """Detects the end of pytest output in a line.
        Args:
            line: A line of text from the subprocess output.

        Returns:
            A tuple (terminate: bool, timeout: float, msg: str).
        """
        m = re.match(r"=+\s+(\d+\s+(failed|passed|skipped|deselected|warnings|errors),?\s+)+in\s+\d+.*=+", line.strip())
        if m:
            if "passed" in line and "failed" not in line:
                msg = "All tests passed but the process did not exit cleanly and had to be terminated."
            else:
                msg = ""  # Message will be constructed elsewhere if tests failed
            # allow 10 seconds before terminating
            return True, 10.0, msg
        return False, 0.0, ""


class EndMarkerMonitor:
    def __init__(self, marker: str, grace_period: float = 2.0):
        """Monitors for a literal end-marker string in the subprocess output.

        When the marker is seen, the process is given a short grace period to
        exit cleanly before being force-terminated.

        Args:
            marker: The exact string to look for in each output line.
            grace_period: Seconds to wait after seeing the marker before
                force-terminating the process.
        """
        self.marker = marker
        self.grace_period = grace_period

    def __call__(self, line: str) -> tuple[bool, float, str]:
        if self.marker in line:
            return True, self.grace_period, ""
        return False, 0.0, ""


async def monitor_output_stream(
    reader: asyncio.StreamReader,
    proc: asyncio.subprocess.Process,
    monitors: list[Callable[[str], tuple[bool, float, str]]],
    output_inactivity_timeout: float | None,
):
    """Monitors the output stream of a subprocess and applies monitor functions to each line.

    If any monitor function indicates that the process should be terminated, it will do so.

    Args:
        reader: The StreamReader to monitor.
        proc: The subprocess whose output is being monitored.
        monitors: A list of monitor functions that take a line of text and return a tuple
            (terminate: bool, timeout: float, msg: str).
        output_inactivity_timeout: Time in seconds to wait for output before considering the process inactive.
            If None, inactivity monitoring is disabled.

    Returns:
        A message string if the process was terminated due to a monitor condition, or None if everything
        completed successfully.
    """
    while not reader.at_eof():
        try:
            line = await asyncio.wait_for(reader.readline(), output_inactivity_timeout)
            text = line.decode("utf-8", errors="replace")
            for monitor in monitors:
                terminate, timeout, msg = monitor(text)
                if terminate:
                    if timeout > 0:
                        await asyncio.sleep(timeout)
                    if proc.returncode is None:
                        # if the process is still running terminate it and return the message
                        await force_terminate(proc)
                        return msg
        except TimeoutError:
            await force_terminate(proc)
            return "Process killed due to inactivity"
    return None  # everything ok


async def robust_subprocess_run(
    cmd: list[str] | str,
    timeout: float | None = None,
    output_inactivity_timeout: float | None = None,
    end_marker: str | None = None,
    **kwargs,
) -> tuple[subprocess.CompletedProcess, str | None]:
    """Runs a subprocess command that may not well behave and hang

    Args:
        cmd: List of command arguments to run
        timeout: Overall timeout for the subprocess
        output_inactivity_timeout: If set, terminate the process if no output is received for this many seconds
        end_marker: Controls end-of-process monitoring.
            ``"pytest"`` – terminate after the pytest summary line appears (10 s grace).
            Any other string – terminate when that literal string appears in output (2 s grace).
            ``None`` – no early termination monitoring.
        **kwargs: Additional keyword arguments to pass to subprocess.Popen

    Returns:
        A tuple of (CompletedProcess, termination_message)
    """
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader(loop=loop)

    # Put each subprocess in its own session/process-group so timeouts can
    # reliably terminate the entire command tree, including shell children.
    kwargs.setdefault("start_new_session", True)

    factory_fn = functools.partial(create_ForwardToStreamReaderProtocol, reader, loop)

    if isinstance(cmd, str):
        transport, protocol = await loop.subprocess_shell(
            factory_fn,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
    else:
        transport, protocol = await loop.subprocess_exec(
            factory_fn,
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
    proc = asyncio.subprocess.Process(transport, protocol, loop)

    monitors = []
    if end_marker == "pytest":
        monitors.append(PytestEndMonitor())
    elif end_marker is not None:
        monitors.append(EndMarkerMonitor(end_marker))
    try:
        (out, err), msg = await asyncio.gather(
            asyncio.wait_for(proc.communicate(), timeout=timeout),
            monitor_output_stream(reader, proc, monitors, output_inactivity_timeout=output_inactivity_timeout),
        )
    except TimeoutError:
        await force_terminate(proc)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            # If pipes stay open despite termination attempts, return the data
            # captured so far instead of blocking indefinitely.
            out = protocol.stdout_buffer.getvalue()
            err = protocol.stderr_buffer.getvalue()
        msg = f"Process timed out after {timeout} seconds"
    return (
        subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
        ),
        msg,
    )
