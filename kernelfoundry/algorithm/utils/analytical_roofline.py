"""Helpers for the agent-derived analytical roofline target.

The ``analytical-roofline`` skill (see ``skills/analytical-roofline``) asks a one-shot agent
session to derive an ideal, hardware-instruction-count-independent roofline target (FLOPs and
bytes moved) from a task's reference math and benchmark shapes. These helpers build that
agent's prompt and parse/validate its output; see
:meth:`~kernelfoundry.algorithm.agentic_controller.AgenticController._compute_analytical_roofline`
for how the agent session itself is run.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.eval_pipeline.profiler_feedback import (
    get_profiler_feedback_class,
    get_reference_language_for_profiling,
)

ANALYTICAL_ROOFLINE_FILENAME = "analytical_roofline.json"

_BASE_PROMPT = (
    "Analyze the reference implementation and benchmark shapes in this task directory to "
    "derive an ideal, hardware-instruction-count-independent roofline target. Follow the "
    "'analytical-roofline' skill exactly, including its required output file name, "
    "location and JSON schema. Do not modify any other file, and do not call build_and_test."
)


def collate_reference_profiler_feedback(task: Task) -> Optional[str]:
    """Textual summary of the reference's profiler data.
    Handed to the roofline-analysis agent just for reference.

    Returns ``None`` if no reference trace data is available.
    """
    test_result_reference = task.test_result_reference
    if test_result_reference is None or not test_result_reference.trace_results:
        return None
    try:
        ref_language = get_reference_language_for_profiling(task.config)
        gpu_arch = task.config.get("gpu_arch")
        if isinstance(gpu_arch, (list, tuple)):
            gpu_arch = gpu_arch[0]
        elif isinstance(gpu_arch, str):
            gpu_arch = gpu_arch.split(",")[0]
        profiler_feedback_class = get_profiler_feedback_class(
            ref_language, gpu_arch, task.config.get("profiler_reference")
        )
        _, feedback_str = profiler_feedback_class().collate_and_create_feedback(
            {k: v.output_data for k, v in test_result_reference.trace_results.items() if v.output_data},
            worker_info=test_result_reference.worker_info or {},
        )
        return feedback_str or None
    except Exception:
        logging.warning("Failed to collate reference profiler feedback for the roofline-analysis agent.", exc_info=True)
        return None


def build_analytical_roofline_prompt(task: Task) -> str:
    """Build the prompt for the one-shot roofline-analysis agent.

    Appends the reference's measured profiler feedback (if available) as a sanity-check signal
    - see :func:`collate_reference_profiler_feedback`.
    """
    prompt = _BASE_PROMPT
    reference_profiler_feedback = collate_reference_profiler_feedback(task)
    if reference_profiler_feedback:
        prompt += (
            "\n\nFor reference, here is profiler output measured on the actual reference "
            "implementation. Per the skill's guidance, use it only to sanity-check your derived "
            "ideal roofline - never as the source of your flops/bytes numbers:\n"
            f"{reference_profiler_feedback}"
        )
    return prompt


def _parse_and_validate_entries(raw_text: str) -> dict:
    """Parse and validate ``analytical_roofline.json`` content (use-supplied or agent-generated)"""
    entries = json.loads(raw_text)["entries"]
    if not entries:
        raise ValueError("'entries' is empty")
    for benchmark_name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Entry for {benchmark_name!r} is not an object")
        for field_name in ("flops", "bytes_read", "bytes_write"):
            value = entry.get(field_name)
            if value is not None and not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise ValueError(f"{benchmark_name!r}.{field_name} must be a finite, non-negative number")
        if entry.get("bound") not in (None, "compute", "memory"):
            raise ValueError(f"{benchmark_name!r}.bound must be 'compute', 'memory' or omitted")
    return entries


def read_analytical_roofline_output(task_data_dir: Path) -> Optional[dict]:
    """Parse and validate the JSON the roofline-analysis agent should have written.

    Returns ``None`` (triggering a fallback to the instruction-count roofline) if the file
    is missing or fails validation - this number gates an automated stopping criterion, so
    an unparsable or malformed agent output must never be trusted silently.
    """
    output_path = task_data_dir / ANALYTICAL_ROOFLINE_FILENAME
    if not output_path.is_file():
        logging.warning(
            f"Analytical roofline analysis did not produce {output_path.name}; "
            "falling back to the instruction-count roofline."
        )
        return None
    try:
        return _parse_and_validate_entries(output_path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(
            f"Analytical roofline output failed validation ({e}); falling back to the instruction-count roofline."
        )
        return None


def read_user_supplied_analytical_roofline(task: Task) -> Optional[dict]:
    """Check whether the task package already ships a hand-authored ``analytical_roofline.json``."""
    if ANALYTICAL_ROOFLINE_FILENAME not in task.task_data:
        return None
    try:
        return _parse_and_validate_entries(task.task_data[ANALYTICAL_ROOFLINE_FILENAME].decode("utf-8"))
    except Exception as e:
        logging.warning(f"User-supplied {ANALYTICAL_ROOFLINE_FILENAME} failed validation ({e}); ignoring it.")
        return None
