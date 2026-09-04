"""Custom TemporaryDirectory with container-aware cleanup.

When container processes run as root they can leave root-owned files behind,
making normal shutil.rmtree cleanup fail with PermissionError.  This module
provides a drop-in replacement for :class:`tempfile.TemporaryDirectory` that
retries in that case: ``sudo rm -rf`` on POSIX, and clearing read-only bits
before a second rmtree on Windows, where there is no usable ``sudo``.
"""

import logging
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class TemporaryDirectory(tempfile.TemporaryDirectory):
    """A :class:`tempfile.TemporaryDirectory` that can clean up root-owned files.

    After containerized runs (Docker / Podman), files inside the temporary
    directory may be owned by *root*.  Normal cleanup raises
    :class:`PermissionError` in that situation.  This subclass detects that
    and retries the deletion via ``sudo rm -rf``, which must be available and
    permitted for the running user.

    All constructor arguments are forwarded to
    :class:`tempfile.TemporaryDirectory` unchanged.
    """

    def cleanup(self) -> None:
        """Clean up the temporary directory.

        Tries the standard cleanup first.  If that raises a
        :class:`PermissionError`, falls back to a platform-appropriate retry.
        (``_sudo_cleanup`` on POSIX, ``_force_cleanup`` on Windows)
        """
        try:
            super().cleanup()
        except PermissionError:
            if os.name == "posix":
                self._sudo_cleanup()
            else:
                self._force_cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sudo_cleanup(self) -> None:
        path = Path(self.name)
        if not path.exists():
            return

        logger.warning("no permission to clean up %s — retrying with sudo rm -rf", path)
        try:
            subprocess.run(
                ["sudo", "rm", "-rf", str(path)],
                check=True,
                capture_output=True,
                # Name the codec: without it text mode uses the locale codec, which mangles any
                # non-ASCII path or message in the captured stderr below.
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "sudo rm -rf %s failed (exit %d): %s",
                path,
                exc.returncode,
                (exc.stderr or "").strip(),
            )
            raise
        except OSError as exc:
            # sudo not installed at all.
            logger.error("Could not run sudo to clean up %s: %s", path, exc)
            raise

    def _force_cleanup(self) -> None:
        """Retry deletion on platforms with no ``sudo``, clearing read-only files as we go.
        If not succeeding (e.g.  a file still open in another process), this warns and gives up.
        """
        path = Path(self.name)
        if not path.exists():
            return

        logger.warning("no permission to clean up %s: retrying with read-only bits cleared", path)
        # Clear the bits up front rather than from an error callback: rmtree's callback parameter
        # was renamed from `onerror` to `onexc` in 3.12, and this package supports 3.10, so a
        # single spelling would break on one side or the other of that split.
        for entry in path.rglob("*"):
            try:
                os.chmod(entry, stat.S_IWRITE)
            except OSError:
                pass  # Best effort; rmtree below reports what actually remains.
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning(
                "Could not remove the temporary directory %s (%s). It has been left in place; "
                "something else most likely still holds a file open inside it.",
                path,
                exc,
            )

    # Make cleanup() run automatically when used as a plain context manager
    # (tempfile.TemporaryDirectory already does this, but we override __exit__
    # so that *our* cleanup() is called, not the base one).
    def __exit__(self, exc, value, tb):
        self.cleanup()
