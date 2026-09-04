"""Profiler command wrappers for different profilers (Unitrace, NCU, VTune)"""

import csv
import io
import os
import json
import re
import shutil
import subprocess
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from functools import partial
import uuid

# GPU architectures that are supported - can be extended
VALID_CUDA_ARCHS = [
    "Maxwell",
    "Pascal",
    "Volta",
    "Turing",
    "Ampere",
    "Hopper",
    "Ada",
    "native",
    "A100",
    "A6000",
    "L40S",
    "L4",
]

_VTUNE_HOTSPOTS_COLUMNS_PASS1 = ",".join(
    [
        "Computing Task:Total Time",
        "Computing Task:Average Time",
        "Computing Task:Instance Count",
        "Computing Task:SIMD Width",
        # XVE utilization
        "XVE Array:Active",
        "XVE Array:Stalled",
        "XVE Array:Idle",
        # Occupancy: "XVE Threads Occupancy" also auto-expands Peak XVE Threads Occupancy sub-columns
        "XVE Threads Occupancy",
        # Stall reason breakdown
        "XVE Stall Reasons:Send",
        "XVE Stall Reasons:Barrier",
        "XVE Stall Reasons:SBID",
        "XVE Stall Reasons:Instruction Fetch",
        "XVE Stall Reasons:Dist or Acc",
        # Pipeline activity (% of time each pipe is executing)
        "XVE Pipelines:ALU0 active",
        "XVE Pipelines:ALU1 active",
        "XVE Pipelines:XMX active",
        # Instruction counts (raw; used for relative instruction-mix analysis)
        "XVE Instructions:ALU0 Instructions",
        "XVE Instructions:ALU1 Instructions",
        "XVE Instructions:XMX instructions",
        # Memory
        "GPU Memory Bandwidth",
        "GPU L3:Busy",
        "GPU L3:Stalled",
        "GPU L3:Miss Ratio",
        "GPU L3:Average Bandwidth",
        "TLB Misses",
    ]
)

_VTUNE_HOTSPOTS_COLUMNS_PASS2 = ",".join(
    [
        "Computing Task:Total Time",
        "XVE Threads Occupancy",
    ]
)


class ProfilerUnavailable(Exception):
    """
    Exception if profiler cannot be found
    """


class Profiler(ABC):
    """Abstract base class for profiler helpers"""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the profiler"""
        pass

    def unavailable_reason(self) -> str | None:
        """Why this profiler cannot run on this machine, or ``None`` if nothing rules it out."""
        return None

    @abstractmethod
    def wrap_cmd(self, cmd: str) -> str:
        """Wrap the given command with profiler-specific command
        Args:
            cmd: The original command to run the program without profiling
        Returns:
            A wrapped command that includes the original command and any additional commands needed for profiling.
        """
        pass

    def env_vars(self):
        """Return a dictionary of environment variables to update or add when running the command"""
        return {}

    def end_marker(self) -> str | None:
        """Return the end-of-process marker string for this profiler.

        The marker is passed to ``robust_subprocess_run`` to detect when the
        profiled process has finished and can safely be terminated.

        The default ``"pytest"`` activates the pytest-summary monitor, which
        waits for the pytest result line.  Return a different string to watch
        for that literal in the output, or ``None`` to disable early
        termination entirely.
        """
        return "pytest"

    def prepare(self, host_output_dir: Path) -> None:
        """Prepare the host output directory before running the profiler command.
        Override in subclasses that need to stage files into the host directory
        (e.g. scripts that must be accessible inside a container).
        """
        pass

    @abstractmethod
    def read_output(self) -> dict[str, str]:
        """Read the profiler output into a dictionary that maps the filename to its content"""
        pass


class Unitrace(Profiler):
    """Unitrace profiler helper"""

    def __init__(
        self, output_dir: Path | str, group: str = "ComputeBasic", timeline: bool = True, unitrace_cmd: str = "unitrace"
    ):
        super().__init__(output_dir)
        self.group = group
        self.timeline = timeline
        self.unitrace_cmd = unitrace_cmd

    @property
    def name(self) -> str:
        return "unitrace"

    def wrap_cmd(self, cmd: str) -> str:
        """Wrap the given command with unitrace profiling command"""
        output_prefix = str(self.output_dir / "trace")
        # We assume that VectorEngineProfile and MemoryProfile are available, which is true for LNL, BMG, PTL
        timeline_flag = "--chrome-kernel-logging --chrome-itt-logging" if self.timeline else ""
        session_flag = f"--start-paused --session {uuid.uuid4().hex}"
        return f"{self.unitrace_cmd} {timeline_flag} {session_flag} --group {self.group} --metric-query --output-dir-path {self.output_dir.as_posix()} -o {output_prefix} {cmd}"

    def env_vars(self):
        return {"KERNELFOUNDRY_unitrace_cmd": self.unitrace_cmd}

    def read_output(self) -> dict[str, str]:
        """Read the profiler output into a dictionary that maps the filename to its content"""
        trace_data = {}

        # check for *python*.json first (to avoid e.g. ninja.json) and otherwise for other json with model run loop, or all
        candidates = (
            list(self.output_dir.glob("*python*.json"))
            or [p for p in self.output_dir.glob("*.json") if "model run loop" in p.read_text()]
            or list(self.output_dir.glob("*.json"))
        )
        if len(candidates) > 1:
            logging.warning(f"Multiple timeline files in {self.output_dir}: {[p.name for p in candidates]}")
        if candidates:
            trace_data["timeline"] = json.loads(candidates[0].read_text().replace("\x00", ""))
        else:
            logging.warning(f"No timeline json file found in {self.output_dir}")

        for path in Path(self.output_dir).glob("trace.*"):
            text = path.read_text()
            if text:
                trace_data[path.name] = text
        return trace_data


class OCLUnitrace(Unitrace):
    """Helper to run Unitrace for OpenCL kernels with metric-sampling"""

    def end_marker(self) -> str | None:
        return "UNITRACE_DONE"

    def prepare(self, host_output_dir: Path) -> None:
        """Copy the retry script into the host output directory so it is accessible
        inside the container (where host_output_dir is mounted as self.output_dir).
        """
        import shutil

        src = Path(__file__).parent / "utils" / "retry_unitrace.sh"
        dst = host_output_dir / "retry_unitrace.sh"
        shutil.copy2(src, dst)

    def wrap_cmd(self, cmd: str) -> str:
        """Wrap the given command with unitrace profiling command for OpenCL using retry script"""
        output_prefix = str(self.output_dir / "trace")
        timeline_flag = "--chrome-kernel-logging --chrome-itt-logging" if self.timeline else ""
        session_flag = f"--start-paused --session {uuid.uuid4().hex}"

        # The retry script is copied to self.output_dir by prepare() before this is called,
        # so the path is valid both on the host and inside any container where output_dir is mounted.
        retry_script = self.output_dir / "retry_unitrace.sh"
        return f'bash "{retry_script.as_posix()}" "{self.unitrace_cmd}" "{self.output_dir.as_posix()}" "{output_prefix}" "{self.group}" "{timeline_flag}" "{session_flag}" "{cmd}"'


class NCU(Profiler):
    """Nsight Compute profiler helper"""

    def __init__(self, output_dir: Path | str, ncu_cmd: str = "ncu"):
        super().__init__(output_dir)
        self.ncu_cmd = ncu_cmd

    @property
    def name(self) -> str:
        return "ncu"

    def unavailable_reason(self) -> str | None:
        if shutil.which(self.ncu_cmd) is None:
            return (
                f"{self.ncu_cmd} is not on PATH. It ships with the CUDA toolkit; add the toolkit's "
                "bin directory to PATH, or set eval_config.profile_custom_model: false to run "
                "without profiling."
            )
        return None

    def wrap_cmd(self, cmd: str) -> str:
        """Wrap the given command with NCU profiling command.

        ``ncu`` needs elevated privileges to read performance counters unless the driver has been
        told to allow it for all users. Since sudo is disabled on Windows, the command is issued plainly:
        it succeeds where counter access has been granted, and where it has not, read_output explains what to do.

        See https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
        for how to enable non-root profiling on either platform.
        """
        output_file = self.output_dir / "ncu_report.csv"
        ncu = f"{self.ncu_cmd} --set detailed --csv --log-file {output_file.as_posix()} {cmd}"
        if os.name == "nt":
            return ncu
        # Note that sudo-rs which is used in ubuntu 25.10 does not support --preserve-env
        return f"""sudo -E env "PYTHONPATH=$PYTHONPATH" {ncu}"""

    def read_output(self) -> dict[str, str]:
        """Read the profiler output into a dictionary that maps the filename to its content"""
        trace_data = {}

        output_file = self.output_dir / "ncu_report.csv"
        if not output_file.is_file():
            # No report means ncu never got far enough to write one. Raising a typed error
            missing = self.unavailable_reason()
            if missing:
                raise ProfilerUnavailable(
                    f"{self.ncu_cmd} wrote no report to {output_file}. {missing} "
                    "If the run was containerised, check the image rather than this host."
                )
            hint = (
                "run from an elevated prompt, or allow counter access for all users in the NVIDIA "
                "Control Panel (Desktop > Developer Settings)"
                if os.name == "nt"
                else "see NVIDIA's ERR_NVGPUCTRPERM guidance for enabling non-root profiling"
            )
            raise ProfilerUnavailable(
                f"{self.ncu_cmd} wrote no report to {output_file}, which normally means it was "
                f"denied access to the GPU performance counters: {hint}."
            )
        # Explicit encoding: ncu writes UTF-8, and the locale codec must not get near it.
        text = output_file.read_text(encoding="utf-8", errors="replace")
        trace_data[output_file.name] = text

        return trace_data


class VTune(Profiler):
    """VTune profiler helper for Intel GPUs"""

    def __init__(self, output_dir: Path | str, vtune_cmd: str = "vtune"):
        super().__init__(output_dir)
        self.vtune_cmd = vtune_cmd

    @property
    def name(self) -> str:
        return "vtune"

    def wrap_cmd(self, cmd: str) -> str:
        """Wrap the given command with VTune collection and hotspot report generation.

        Chains two commands with && so all vtune invocations happen inside
        the same environment (e.g. container) where vtune is available:

        1. vtune -collect  (paused; the kernel under test resumes it via ITT)
        2. vtune -report hotspots -> tsv file

        read_output() then simply reads that file without invoking vtune.
        """
        result_dir = self.output_dir / "vtune_result"
        output_dir = self.output_dir.as_posix()
        report_out = self.output_dir / "vtune_hotspots_pass0.tsv"
        collect = f"{self.vtune_cmd} -collect gpu-hotspots --start-paused" f' -result-dir "{result_dir}" -- {cmd}'
        report = (
            f'{self.vtune_cmd} -report hotspots -result-dir "{result_dir}"'
            f' -group-by computing-task -column "{_VTUNE_HOTSPOTS_COLUMNS_PASS1}"'
            f' -format csv -csv-delimiter tab > "{report_out.as_posix()}"'
        )
        return f"{collect} && {report} && echo VTUNE_DONE"

    def end_marker(self) -> str | None:
        return "VTUNE_DONE"

    def env_vars(self):
        return {
            "KERNELFOUNDRY_vtune_cmd": self.vtune_cmd,
            "KERNELFOUNDRY_vtune_result_dir": self.output_dir / "vtune_result",
            "KERNELFOUNDRY_PROFILER": "vtune",
        }

    @staticmethod
    def _parse_tsv_reports(tsv_contents: list[str]) -> dict[str, dict]:
        """Parse a list of VTune hotspot TSV report strings into a counters dict.

        Skips VTune warning lines before the header.  Within a single file,
        rows with the same kernel name (autotune SIMD-width variants) are merged
        by summing total time and keeping the longest-running variant's per-iteration
        metrics.
        """
        counters: dict[str, dict] = {}

        def _num(v: str) -> float:
            try:
                return float(str(v).replace(",", "").rstrip("%").strip())
            except (ValueError, TypeError):
                return 0.0

        for content in tsv_contents:
            # VTune may emit warning lines before the actual header; skip to the
            # row starting with "Computing Task".
            lines = content.splitlines()
            header_idx = next(
                (i for i, ln in enumerate(lines) if ln.startswith("Computing Task")),
                0,
            )
            payload = "\n".join(lines[header_idx:])

            reader = csv.DictReader(io.StringIO(payload), delimiter="\t")
            for row in reader:
                name = (row.get("Computing Task") or "").strip()
                if not name or name.startswith("[") or name.startswith("vtune:") or name.startswith("war:"):
                    continue

                new_time = _num(row.get("Computing Task:Total Time", ""))
                existing = counters.get(name)

                if existing is None:
                    counters[name] = {k: v for k, v in row.items() if v and v.strip()}
                    continue

                # Duplicate kernel name: can occur when the driver compiles SIMD autotune
                # variants (e.g. SIMD8 and SIMD16) reported as separate rows. Sum total
                # time and keep the longest-running variant's per-iteration metrics.
                old_time = _num(existing.get("Computing Task:Total Time", ""))
                if new_time > old_time:
                    existing = {k: v for k, v in row.items() if v and v.strip()} | {
                        k: v for k, v in existing.items() if k not in row or not row[k].strip()
                    }
                existing["Computing Task:Total Time"] = f"{old_time + new_time}"
                counters[name] = existing

        return counters

    def read_output(self) -> dict[str, str]:
        """Read the pre-generated TSV report and return a counters JSON.

        The TSV file was written by the vtune -report command chained in
        wrap_cmd, so no vtune invocation is needed here.
        """
        tsv_contents = []
        pass0 = self.output_dir / "vtune_hotspots_pass0.tsv"
        if pass0.exists():
            tsv_contents.append(pass0.read_text())
        else:
            logging.warning("VTune report file not found: %s", pass0)
        counters = self._parse_tsv_reports(tsv_contents)
        return {"vtune_counters.json": json.dumps(counters)}


_UNITRACE_PASSES = [
    partial(Unitrace, group="ComputeBasic"),
    partial(Unitrace, group="MemoryProfile"),
    partial(Unitrace, group="VectorEngineProfile"),
]

# Mapping of (language, profiler_type) -> profiler class or list of profilers
_PROFILER_MAPPING = {
    ("sycl", "unitrace"): _UNITRACE_PASSES,
    ("sycl", "vtune"): [VTune],
    ("ocl", "unitrace"): [
        partial(OCLUnitrace, group="ComputeBasic"),
        partial(OCLUnitrace, group="MemoryProfile"),
        partial(OCLUnitrace, group="VectorEngineProfile"),
    ],
    ("ocl", "vtune"): [VTune],
    ("cuda", "ncu"): [NCU],
    ("pytorch", "ncu"): [NCU],
    ("pytorch", "unitrace"): _UNITRACE_PASSES,
    ("triton", "ncu"): [NCU],
    ("triton", "unitrace"): _UNITRACE_PASSES,
}

# Default profiler for each language
_DEFAULT_PROFILERS = {
    "sycl": "unitrace",
    "ocl": "vtune",
    "cuda": "ncu",
}


def get_profilers(language: str, arch: str, profiler_type: str | None = None) -> list[type[Profiler]]:
    """Get the appropriate profiler class based on user-specified profiler type or language & GPU architecture."""
    language = language.lower()

    if profiler_type is None:
        if language in ["pytorch", "triton"]:
            gpu_arch_is_cuda = arch in VALID_CUDA_ARCHS
            return [NCU] if gpu_arch_is_cuda else _UNITRACE_PASSES
        profiler_type = _DEFAULT_PROFILERS.get(language)

    key = (language, profiler_type.lower() if profiler_type else None)
    if key in _PROFILER_MAPPING:
        return _PROFILER_MAPPING[key]

    raise ValueError(
        f"Unknown profiler '{profiler_type}' for language '{language}'. "
        f"Available profilers for '{language}': {[p for (l, p) in _PROFILER_MAPPING.keys() if l == language]}"
    )
