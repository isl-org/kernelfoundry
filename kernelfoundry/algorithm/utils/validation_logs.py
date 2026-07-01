from __future__ import annotations

from typing import Any

from kernelfoundry.eval_pipeline.task import Task


def collect_raw_logs_from_task(task: Task) -> str:
    """Collect raw subprocess logs from build/test results for debugging and fallback output."""

    sections: list[str] = []

    def add_process_result(name: str, result: Any) -> None:
        if result is None:
            return
        parts: list[str] = []
        message = getattr(result, "message", None)
        stdout = getattr(result, "stdout", None)
        stderr = getattr(result, "stderr", None)
        if message:
            parts.append(f"Message:\n{message}")
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if parts:
            sections.append(f"[{name}]\n" + "\n".join(parts))

    add_process_result("build_custom", getattr(getattr(task, "build_result", None), "result", None))
    add_process_result("build_reference", getattr(getattr(task, "build_result_reference", None), "result", None))
    add_process_result(
        "test_custom_correctness", getattr(getattr(task, "test_result", None), "correctness_result", None)
    )
    add_process_result(
        "test_custom_performance", getattr(getattr(task, "test_result", None), "performance_result", None)
    )
    add_process_result(
        "test_reference_correctness",
        getattr(getattr(task, "test_result_reference", None), "correctness_result", None),
    )
    add_process_result(
        "test_reference_performance",
        getattr(getattr(task, "test_result_reference", None), "performance_result", None),
    )

    return "\n\n".join(sections)
