"""Custom TemporaryDirectory with container-aware cleanup.

When container processes run as root they can leave root-owned files behind,
making normal shutil.rmtree cleanup fail with PermissionError.  This module
provides a drop-in replacement for :class:`tempfile.TemporaryDirectory` that
falls back to ``sudo rm -rf`` in that case.
"""

import logging
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
        :class:`PermissionError` (e.g. root-owned files left by a container),
        falls back to ``sudo rm -rf``.
        """
        try:
            super().cleanup()
        except PermissionError:
            self._sudo_cleanup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sudo_cleanup(self) -> None:
        path = Path(self.name)
        if not path.exists():
            return

        logger.warning("PermissionError during cleanup of %s — retrying with sudo rm -rf", path)
        try:
            result = subprocess.run(
                ["sudo", "rm", "-rf", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "sudo rm -rf %s failed (exit %d): %s",
                path,
                exc.returncode,
                exc.stderr.strip(),
            )
            raise

    # Make cleanup() run automatically when used as a plain context manager
    # (tempfile.TemporaryDirectory already does this, but we override __exit__
    # so that *our* cleanup() is called, not the base one).
    def __exit__(self, exc, value, tb):
        self.cleanup()
