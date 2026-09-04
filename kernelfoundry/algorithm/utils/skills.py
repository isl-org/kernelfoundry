"""Utilities for reading, writing and inspecting Agent Skill files.

A *skill* is a directory that contains, at minimum, a ``SKILL.md`` file made up
of YAML frontmatter followed by a Markdown body. In addition to ``SKILL.md`` a
skill may bundle supporting resources, conventionally grouped under
``scripts/``, ``references/`` and ``assets/`` (any other files/directories are
allowed too).

The format follows the Agent Skills specification:
https://agentskills.io/specification

Design notes
------------
* Resource files (everything besides ``SKILL.md``) are *discovered* and tracked
  as paths relative to the skill root, but their contents are **not** loaded
  eagerly. This mirrors the spec's progressive-disclosure model and avoids
  pulling large reference files into memory. Use :meth:`Skill.list_resources`
  and :meth:`Skill.read_resource` to access them on demand.
* The skill is keyed off its *directory* (not the ``SKILL.md`` file directly),
  because the spec requires ``name`` to match the parent directory name.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml

SKILL_FILE = "SKILL.md"

# Conventional optional directories defined by the specification.
RESOURCE_DIRS = ("scripts", "references", "assets")

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(?P<frontmatter>.*?)\r?\n---[ \t]*\r?\n?(?P<body>.*)\Z",
    re.DOTALL,
)

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillError(Exception):
    """Raised when a skill cannot be parsed, validated or written."""


@dataclass
class SkillFrontmatter:
    """Structured representation of the ``SKILL.md`` YAML frontmatter.

    Only ``name`` and ``description`` are required by the specification. The
    remaining fields are optional and default to empty/None. ``extra`` captures
    any non-standard frontmatter keys so they survive a load/save round-trip.
    """

    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    allowed_tools: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # (de)serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SkillFrontmatter:
        if not isinstance(data, dict):
            raise SkillError("Frontmatter must be a YAML mapping.")

        data = dict(data)  # shallow copy, we'll pop known keys

        try:
            name = data.pop("name")
            description = data.pop("description")
        except KeyError as exc:
            raise SkillError(f"Frontmatter is missing required field: {exc}") from exc

        license_ = data.pop("license", None)
        compatibility = data.pop("compatibility", None)
        metadata = data.pop("metadata", None) or {}
        # The spec uses the hyphenated key ``allowed-tools``.
        allowed_tools = data.pop("allowed-tools", None)

        if not isinstance(metadata, dict):
            raise SkillError("Frontmatter 'metadata' must be a mapping.")

        return cls(
            name=name,
            description=description,
            license=license_,
            compatibility=compatibility,
            metadata=metadata,
            allowed_tools=allowed_tools,
            extra=data,  # whatever is left over
        )

    def to_dict(self) -> Dict[str, Any]:
        """Render the frontmatter back to an ordered mapping for YAML output."""
        out: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.license is not None:
            out["license"] = self.license
        if self.compatibility is not None:
            out["compatibility"] = self.compatibility
        if self.metadata:
            out["metadata"] = self.metadata
        if self.allowed_tools is not None:
            out["allowed-tools"] = self.allowed_tools
        # Preserve any unknown keys last.
        out.update(self.extra)
        return out

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate(self, expected_name: Optional[str] = None) -> None:
        """Validate the frontmatter against the specification constraints.

        Args:
            expected_name: If given (typically the skill directory name), the
                ``name`` field must match it.

        Raises:
            SkillError: if any constraint is violated.
        """
        if not self.name or not (1 <= len(self.name) <= 64):
            raise SkillError("'name' must be between 1 and 64 characters.")
        if not _NAME_RE.match(self.name):
            raise SkillError(
                "'name' may only contain lowercase letters, digits and single "
                "hyphens, and must not start/end with or repeat a hyphen: "
                f"{self.name!r}"
            )
        if expected_name is not None and self.name != expected_name:
            raise SkillError(f"'name' ({self.name!r}) must match the skill directory name " f"({expected_name!r}).")

        if not self.description or not (1 <= len(self.description) <= 1024):
            raise SkillError("'description' must be between 1 and 1024 characters.")

        if self.compatibility is not None and not (1 <= len(self.compatibility) <= 500):
            raise SkillError("'compatibility' must be between 1 and 500 characters.")


@dataclass
class Skill:
    """An Agent Skill loaded from (or destined for) a directory on disk.

    Attributes:
        frontmatter: Parsed ``SKILL.md`` frontmatter.
        body: Markdown body of ``SKILL.md`` (everything after the frontmatter).
        root: Directory the skill was loaded from, if any.
        resources: Resource file paths relative to ``root``, excluding
            ``SKILL.md``. Discovered lazily on load; contents are not read.
    """

    frontmatter: SkillFrontmatter
    body: str = ""
    root: Optional[Path] = None
    resources: List[Path] = field(default_factory=list)

    # ------------------------------------------------------------------
    # convenience accessors for frontmatter / metadata
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.frontmatter.metadata

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return a single value from the frontmatter ``metadata`` mapping."""
        return self.frontmatter.metadata.get(key, default)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> Skill:
        """Parse the raw text of a ``SKILL.md`` file into frontmatter + body."""
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise SkillError("SKILL.md must start with a YAML frontmatter block delimited by " "'---' lines.")
        try:
            raw = yaml.safe_load(match.group("frontmatter")) or {}
        except yaml.YAMLError as exc:
            raise SkillError(f"Invalid YAML frontmatter: {exc}") from exc

        frontmatter = SkillFrontmatter.from_dict(raw)
        body = match.group("body").lstrip("\n")
        return cls(frontmatter=frontmatter, body=body)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        validate: bool = True,
        discover_resources: bool = True,
    ) -> Skill:
        """Load a skill from a directory or directly from a ``SKILL.md`` file.

        Args:
            path: Either the skill directory or the ``SKILL.md`` file inside it.
            validate: Validate frontmatter against the spec after loading.
            discover_resources: Scan the skill directory for bundled resource
                files (everything besides ``SKILL.md``).

        Returns:
            The loaded :class:`Skill`.
        """
        path = Path(path)
        if path.is_dir():
            root = path
            skill_file = path / SKILL_FILE
        elif path.name == SKILL_FILE:
            skill_file = path
            root = path.parent
        else:
            raise SkillError(f"Expected a skill directory or a {SKILL_FILE} file, got: {path}")

        if not skill_file.is_file():
            raise SkillError(f"No {SKILL_FILE} found at: {skill_file}")

        skill = cls.parse(skill_file.read_text(encoding="utf-8"))
        skill.root = root

        if discover_resources:
            skill.resources = skill._discover_resources(root)

        if validate:
            skill.frontmatter.validate(expected_name=root.name)

        return skill

    @staticmethod
    def _discover_resources(root: Path) -> List[Path]:
        """Return sorted resource paths relative to ``root`` (excluding SKILL.md)."""
        resources: List[Path] = []
        for entry in root.rglob("*"):
            if entry.is_dir():
                continue
            rel = entry.relative_to(root)
            if rel.as_posix() == SKILL_FILE:
                continue
            resources.append(rel)
        return sorted(resources, key=lambda p: p.as_posix())

    # ------------------------------------------------------------------
    # resource access (progressive disclosure)
    # ------------------------------------------------------------------
    def list_resources(self, subdir: Optional[str] = None) -> List[Path]:
        """List tracked resource paths, optionally filtered to a subdirectory.

        Args:
            subdir: e.g. ``"scripts"``, ``"references"`` or ``"assets"``.
        """
        if subdir is None:
            return list(self.resources)
        prefix = subdir.strip("/") + "/"
        return [p for p in self.resources if p.as_posix().startswith(prefix)]

    def resource_path(self, relative: Union[str, Path]) -> Path:
        """Return the absolute path to a bundled resource."""
        if self.root is None:
            raise SkillError("Skill has no root directory; cannot resolve resources.")
        return self.root / Path(relative)

    def read_resource(self, relative: Union[str, Path], binary: bool = False):
        """Read a bundled resource file on demand.

        Args:
            relative: Path relative to the skill root (e.g. ``scripts/run.py``).
            binary: Read bytes instead of text.
        """
        target = self.resource_path(relative)
        if not target.is_file():
            raise SkillError(f"Resource not found: {target}")
        if binary:
            return target.read_bytes()
        return target.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def render(self) -> str:
        """Render the skill back to ``SKILL.md`` text (frontmatter + body)."""
        frontmatter = yaml.safe_dump(
            self.frontmatter.to_dict(),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ).strip()
        body = self.body.strip("\n")
        return f"---\n{frontmatter}\n---\n\n{body}\n"

    def save(
        self,
        path: Union[str, Path],
        validate: bool = True,
        copy_resources: bool = True,
    ) -> Path:
        """Write the skill to ``path`` (a directory) and return the skill root.

        Args:
            path: Target skill directory. Created if it does not exist.
            validate: Validate frontmatter before writing.
            copy_resources: Copy tracked resource files from the original
                ``root`` into the target directory. Ignored when the target is
                the same as the source.

        Returns:
            The skill root directory that was written.
        """
        dest = Path(path)
        if validate:
            self.frontmatter.validate(expected_name=dest.name)

        dest.mkdir(parents=True, exist_ok=True)
        (dest / SKILL_FILE).write_text(self.render(), encoding="utf-8")

        if copy_resources and self.root is not None and self.root.resolve() != dest.resolve():
            for rel in self.resources:
                src = self.root / rel
                if not src.is_file():
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)

        self.root = dest
        return dest


def load_skills(directory: Union[str, Path], validate: bool = True) -> List[Skill]:
    """Load every skill found directly under ``directory``.

    A subdirectory is treated as a skill if it contains a ``SKILL.md`` file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SkillError(f"Not a directory: {directory}")

    skills: List[Skill] = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / SKILL_FILE).is_file():
            skills.append(Skill.load(child, validate=validate))
    return skills


def _as_str_set(value: Any) -> set[str]:
    """Normalize a value (scalar or iterable) to a set of string tokens.

    Strings are interpreted as potentially multi-value fields and can be
    separated by commas and/or whitespace.
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return {token for token in re.split(r"[\s,]+", value.strip()) if token}
    if isinstance(value, (list, tuple, set)):
        out: set[str] = set()
        for item in value:
            out.update(_as_str_set(item))
        return out
    return {str(value)}


def filter_skills(
    skills: Iterable[Skill],
    languages_include: Optional[Iterable[str]] = None,
    languages_exclude: Optional[Iterable[str]] = None,
    tags_include: Optional[Iterable[str]] = None,
    tags_exclude: Optional[Iterable[str]] = None,
) -> List[Skill]:
    """Filter skills by their ``languages`` and ``tags`` metadata.

    For each key (``languages``, ``tags``) the include/exclude lists are applied
    independently and a skill must satisfy *all* provided constraints:

    * ``*_include``: keep the skill only if it has at least one of these values
      for the key. An empty/``None`` include list imposes no constraint.
    * ``*_exclude``: drop the skill if it has any of these values for the key.

    Args:
        skills: Skills to filter.
        languages_include: Keep skills whose ``metadata['languages']`` contains
            any of these values.
        languages_exclude: Drop skills whose ``metadata['languages']`` contains
            any of these values.
        tags_include: Keep skills whose ``metadata['tags']`` contains any of
            these values.
        tags_exclude: Drop skills whose ``metadata['tags']`` contains any of
            these values.

    Returns:
        The list of skills that pass every constraint.
    """
    lang_inc = _as_str_set(languages_include) if languages_include else None
    lang_exc = _as_str_set(languages_exclude) if languages_exclude else None
    tag_inc = _as_str_set(tags_include) if tags_include else None
    tag_exc = _as_str_set(tags_exclude) if tags_exclude else None

    def keep(skill: Skill) -> bool:
        languages = _as_str_set(skill.get_metadata("programming-languages"))
        tags = _as_str_set(skill.get_metadata("tags"))

        if lang_inc is not None and languages.isdisjoint(lang_inc):
            return False
        if lang_exc is not None and not languages.isdisjoint(lang_exc):
            return False
        if tag_inc is not None and tags.isdisjoint(tag_inc):
            return False
        if tag_exc is not None and not tags.isdisjoint(tag_exc):
            return False
        return True

    return [skill for skill in skills if keep(skill)]


class SkillLibrary:
    """A collection of skills discovered under one or more root directories.

    Skills are found by recursively scanning for ``SKILL.md`` files, so nested
    layouts such as ``skills/opencl/opencl-gemm/SKILL.md`` are supported.
    """

    def __init__(self, skills: Optional[Iterable[Skill]] = None):
        self.skills: List[Skill] = list(skills) if skills is not None else []

    def __len__(self) -> int:
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills)

    @classmethod
    def load(cls, root: Union[str, Path], validate: bool = True) -> SkillLibrary:
        """Recursively load every skill under ``root`` into a library."""
        root = Path(root)
        if not root.is_dir():
            raise SkillError(f"Not a directory: {root}")

        skills: List[Skill] = []
        for skill_file in sorted(root.rglob(SKILL_FILE)):
            if skill_file.is_file():
                skills.append(Skill.load(skill_file.parent, validate=validate))
        return cls(skills)

    def get(self, name: str) -> Optional[Skill]:
        """Return the skill with the given ``name``, or ``None``."""
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def filter(
        self,
        languages_include: Optional[Iterable[str]] = None,
        languages_exclude: Optional[Iterable[str]] = None,
        tags_include: Optional[Iterable[str]] = None,
        tags_exclude: Optional[Iterable[str]] = None,
    ) -> List[Skill]:
        """Filter the library's skills by ``languages`` and ``tags`` metadata.

        See :func:`filter_skills` for the include/exclude semantics.
        """
        return filter_skills(
            self.skills,
            languages_include=languages_include,
            languages_exclude=languages_exclude,
            tags_include=tags_include,
            tags_exclude=tags_exclude,
        )
