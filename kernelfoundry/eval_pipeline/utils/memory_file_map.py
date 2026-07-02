import base64
from io import BytesIO, TextIOWrapper
import logging
from pathlib import Path
import tarfile
from collections import defaultdict
import time

IGNORE_PATTERNS = ["__pycache__", "__MACOSX", ".DS_Store", "Thumbs.db", ".pyc", ".pyo", ".git", ".swp"]


class MemoryFileMap:
    """In-memory representation of a file archive"""

    def __init__(self):
        self.file_map = defaultdict(BytesIO)

    def __contains__(self, file_path: str | Path) -> bool:
        """Checks if a file exists in the archive."""
        return str(file_path) in self.file_map

    def clear(self):
        """Clears all files from the archive."""
        self.file_map.clear()

    def open(self, file_path: str | Path, mode: str = "r") -> BytesIO | TextIOWrapper:
        """Opens a file in the archive with the specified mode.

        Args:
            file_path (str | Path): Path of the file within the archive.
            mode (str): Mode to open the file ('r', 'w', 'rb', 'wb').
                - 'r': Read text mode (returns TextIOWrapper)
                - 'w': Write text mode (returns TextIOWrapper)
                - 'rb': Read binary mode (returns BytesIO)
                - 'wb': Write binary mode (returns BytesIO)

        Returns:
            io.BytesIO | io.TextIOWrapper: The file stream in binary or text mode.

        Raises:
            ValueError: If an invalid mode is provided.
            FileNotFoundError: If the file doesn't exist in read mode.
        """
        if mode not in ("r", "w", "rb", "wb"):
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'r', 'w', 'rb', 'wb'.")

        key = str(file_path)
        if mode in ("r", "rb"):
            # Read mode: return existing stream or raise error if not found
            if key not in self.file_map:
                raise FileNotFoundError(f"File not found: {file_path}")
            stream = self.file_map[key]
            stream.close = lambda: None  # Prevent closing the stream
            stream.seek(0)
            if mode == "r":
                return TextIOWrapper(stream, encoding="utf-8")
            return stream
        elif mode in ("w", "wb"):
            # Write mode: create new stream
            stream = BytesIO()
            stream.close = lambda: None  # Prevent closing the stream
            self.file_map[key] = stream
            if mode == "w":
                return TextIOWrapper(stream, encoding="utf-8")
            return stream
        else:
            raise ValueError(f"Unhandled mode: {mode}")

    def list_files(self) -> list[str]:
        """Lists all files in the archive.

        Returns:
            list[str]: List of file paths in the archive.
        """
        return list(self.file_map.keys())

    def remove_root_dir(self) -> bool:
        """Removes a common root directory or directories from all files in the archive.

        Returns:
            bool: True if a common root directory was found and removed, False otherwise.
        """
        all_paths = [Path(fp) for fp in self.file_map.keys()]
        if not all_paths:
            return False

        common_parts = all_paths[0].parts
        for path in all_paths[1:]:
            common_parts = tuple(a for a, b in zip(common_parts, path.parts) if a == b)
            if not common_parts:
                return False

        if len(common_parts) < 1:
            return False  # No common root directory to remove

        common_root = Path(*common_parts)
        new_file_map = defaultdict(BytesIO)

        for file_path, byte_stream in self.file_map.items():
            path_obj = Path(file_path)
            relative_path = path_obj.relative_to(common_root)
            if relative_path == Path("."):
                continue  # Skip the root directory itself
            new_file_map[str(relative_path)] = byte_stream

        self.file_map = new_file_map
        return True

    def __getitem__(self, file_path: str | Path) -> bytes | None:
        """Gets the bytes of a file from the archive.

        Args:
            file_path (str | Path): Path of the file within the archive.

        Returns:
            bytes | None: The file content as bytes, or None if the file doesn't exist.
        """
        key = str(file_path)
        if key not in self.file_map:
            return None
        stream = self.file_map[key]
        stream.close = lambda: None  # Prevent closing the stream
        stream.seek(0)
        return stream.getvalue()

    def __setitem__(self, file_path: str | Path, content: bytes):
        """Sets a file in the archive.

        Args:
            file_path (str | Path): Path of the file within the archive.
            content (bytes): Content of the file.
        """
        self.file_map[str(file_path)] = BytesIO(content)

    def _check_valid_file(self, file_path: str | Path):
        """Check if a file should be included based on ignore patterns."""
        file_str = str(file_path)
        for pattern in IGNORE_PATTERNS:
            if pattern in file_str:
                logging.warning(f"Excluded invalid file {file_str} due to pattern {pattern}.")
                return False
        return True

    def to_tarball(self, mode="w:gz") -> bytes:
        """Creates a tarball from the in-memory files.

        Returns:
            bytes: The tarball as a byte string.
        """
        tar_bytes = BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
            for file_path, byte_stream in self.file_map.items():
                byte_stream.seek(0)
                info = tarfile.TarInfo(name=file_path)
                info.mtime = int(time.time())
                info.size = len(byte_stream.getvalue())
                tar.addfile(tarinfo=info, fileobj=byte_stream)
        tar_bytes.seek(0)
        return tar_bytes.getvalue()

    def from_tarball(self, *, tarball_bytes: bytes | None = None, tarball_path: str | Path | None = None):
        """Loads files from a tarball into the in-memory archive.

        Args:
            tarball_bytes (bytes): The tarball as a byte string.
            tarball_path (str | Path): Path to the tarball file.
        """
        assert (tarball_bytes is None) != (
            tarball_path is None
        ), "Exactly one of tarball_bytes or tarball_path must be provided"

        kwargs = dict(mode="r")
        if tarball_bytes is not None:
            kwargs["fileobj"] = BytesIO(tarball_bytes)
        else:
            kwargs["name"] = tarball_path

        with tarfile.open(**kwargs) as tar:
            for member in tar.getmembers():
                file_obj = tar.extractfile(member)
                if file_obj and self._check_valid_file(member.name):
                    content = file_obj.read()
                    self.file_map[member.name] = BytesIO(content)

    def from_zip(self, *, zip_bytes: bytes | None = None, zip_path: str | Path | None = None):
        """Loads files from a zip archive into the in-memory archive.

        Args:
            zip_bytes (bytes): The zip archive as a byte string.
            zip_path (str | Path): Path to the zip file.
        """
        import zipfile

        assert (zip_bytes is None) != (zip_path is None), "Exactly one of zip_bytes or zip_path must be provided"

        if zip_bytes is not None:
            zip_file_obj = BytesIO(zip_bytes)
        else:
            zip_file_obj = zip_path

        with zipfile.ZipFile(zip_file_obj, "r") as zip_file:
            for file_info in zip_file.infolist():
                if not file_info.is_dir() and self._check_valid_file(file_info.filename):
                    with zip_file.open(file_info) as file_obj:
                        content = file_obj.read()
                        self.file_map[file_info.filename] = BytesIO(content)

    def from_archive(self, *, archive_bytes: bytes | None = None, archive_path: str | Path | None = None):
        """Loads files from an archive (zip or tarball) into the in-memory archive.

        Args:
            archive_bytes (bytes): The archive as a byte string.
            archive_path (str | Path): Path to the archive file.
        """
        assert (archive_bytes is None) != (
            archive_path is None
        ), "Exactly one of archive_bytes or archive_path must be provided"

        if archive_bytes is not None:
            # Try zip first, then tarball because the vscode extension uses zip format
            try:
                self.from_zip(zip_bytes=archive_bytes)
            except Exception:
                self.from_tarball(tarball_bytes=archive_bytes)
        else:
            # Guess from file extension
            archive_path = Path(archive_path)
            ext = archive_path.suffix.lower()

            if ext == ".zip":
                self.from_zip(zip_path=archive_path)
            elif ext in (".tar", ".gz", ".tgz", ".bz2", ".xz"):
                self.from_tarball(tarball_path=archive_path)
            else:
                # Try zip first, then tarball because the vscode extension uses zip format
                try:
                    self.from_zip(zip_path=archive_path)
                except Exception:
                    self.from_tarball(tarball_path=archive_path)

    def from_path(self, path: str | Path, include_extensions: list[str] | None = None):
        """Loads files from a given path to an archive or directory into the in-memory archive.

        Args:
            path (str | Path): Path to the archive file or directory.
            include_extensions (list[str] | None): If provided, only files with these extensions will be included
                when loading from a directory.
        """
        path = Path(path)
        if path.is_dir():
            self.from_disk(input_dir=path, include_extensions=include_extensions)
        else:
            self.from_archive(archive_path=path)

    def to_disk(self, output_dir: str | Path):
        """Extracts the archive to disk.

        Args:
            output_dir (str | Path): Directory where files will be extracted.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tarball_bytes = self.to_tarball("w")
        tar_bytes = BytesIO(tarball_bytes)

        with tarfile.open(fileobj=tar_bytes, mode="r") as tar:
            tar.extractall(path=output_path, filter="data")

    def from_disk(self, input_dir: str | Path, include_extensions: list[str] | None = None):
        """Loads files from disk into the in-memory archive.

        Args:
            input_dir (str | Path): Directory to load files from.
            include_extensions (list[str] | None): If provided, only files with these extensions will be included.
        """
        input_path = Path(input_dir)
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and self._check_valid_file(file_path):
                if include_extensions is not None and file_path.suffix.lower() not in include_extensions:
                    continue
                elif file_path.name == "Dockerfile":  # allow Dockerfile without extension
                    pass
                relative_path = file_path.relative_to(input_path)
                with open(file_path, "rb") as f:
                    content = f.read()
                    self.file_map[str(relative_path)] = BytesIO(content)

    def encode(self) -> dict:
        """Encodes the archive to a serializable dictionary.

        Returns:
            dict: Dictionary mapping file paths to base64-encoded content.
        """
        encoded_map = {}
        for file_path, byte_stream in self.file_map.items():
            byte_stream.seek(0)
            encoded_content = base64.b64encode(byte_stream.getvalue()).decode("utf-8")
            encoded_map[file_path] = encoded_content
        return encoded_map

    def decode(self, encoded_map: dict):
        """Decodes a serializable dictionary into the archive.

        Args:
            encoded_map (dict): Dictionary mapping file paths to base64-encoded content.
        """
        for file_path, encoded_content in encoded_map.items():
            content = base64.b64decode(encoded_content.encode("utf-8"))
            self.file_map[file_path] = BytesIO(content)
