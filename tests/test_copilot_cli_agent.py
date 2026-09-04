"""Tests for CopilotCLIAgent using the copilot-sim simulator.

These tests do NOT access the database and do NOT require GPU hardware.
They use a mock BuildAndTestHandler in place of the real evaluation pipeline.

Run with:
    pytest kernelfoundry/tests/test_copilot_cli_agent.py -v
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

from kernelfoundry.algorithm.agent_base import BuildAndTestHandler, EvaluateFunctionResult
from kernelfoundry.algorithm.copilot_cli_agent import CopilotCLIAgent
from kernelfoundry.algorithm.schemas import EvalResult, Program
from kernelfoundry.eval_pipeline.task import Task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MATMUL_TASK_DIR = Path(__file__).parent.parent / "tasks" / "example_custom"
COPILOT_SIM = Path(__file__).parent / "copilot-sim"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def matmul_task() -> Task:
    """Load the matmul Task from the example_custom task directory."""
    task, _ = Task.create(MATMUL_TASK_DIR)
    return task


# ---------------------------------------------------------------------------
# Mock build-and-test handler
# ---------------------------------------------------------------------------


class MockBuildAndTestHandler(BuildAndTestHandler):
    """A :class:`BuildAndTestHandler` that returns canned results without running
    the real Evaluator or touching the database.

    By default it fails on the first call and succeeds on every subsequent call,
    matching copilot-sim behaviour (broken kernel first, fixed kernel second).
    """

    def __init__(self, always_success: bool | None = None) -> None:
        self.call_count = 0
        self._always_success = always_success
        self.call_log: list[dict] = []

    def call(
        self,
        task: Task,
        folder_path,
        job_id: int,
        task_id: str,
        prompt: str,
        iteration: int,
        branch: int,
        llm_model: str,
        session_log: str,
        previous_program: Program | None = None,
        agent_session_id: str | None = None,
    ) -> EvaluateFunctionResult:
        self.call_count += 1
        if self._always_success is None:
            success = self.call_count > 1  # first call fails, rest pass
        else:
            success = self._always_success

        eval_result = EvalResult(
            compiled=success,
            correctness=success,
            perf_score=5 if success else 1,
            runtime=1.0 if success else -1.0,
            eval_log="mock pass" if success else "mock fail",
        )

        program = Program(
            id=str(uuid.uuid4()),
            code={"kernel.sycl": ["mock code"]},
            language=task.config.get("language", "SYCL"),
            iteration_found=iteration,
            parent_id=previous_program.id if previous_program else None,
            task=task,
            generation=(previous_program.generation + 1) if previous_program else 0,
        )
        program.add_eval_results(eval_result)

        tool_response = {
            "success": success,
            "job_id": job_id,
            "eval_log": eval_result.eval_log,
        }
        self.call_log.append(tool_response)

        return EvaluateFunctionResult(
            tool_response=tool_response,
            eval_result=eval_result,
            program=program,
        )


@pytest.fixture(scope="session")
def copilot_sim_exe(tmp_path_factory) -> str:
    """A form of ``copilot-sim`` this platform can actually launch.

    The sim is a Python script that relies on its ``#!/usr/bin/env python3`` shebang. Windows does
    not honour shebangs, so CreateProcess cannot run it: the simulated session exited 1 having
    written nothing, and the agent then failed on the telemetry file the session never created --
    an obscure FileNotFoundError deep in the agent for what is really "the test harness could not
    start". A one-line .cmd shim forwards to the interpreter running the tests, so one sim serves
    both platforms and neither needs a skip.
    """
    if os.name != "nt":
        return str(COPILOT_SIM)
    shim = tmp_path_factory.mktemp("copilot-shim") / "copilot-sim.cmd"
    shim.write_text(f'@"{sys.executable}" "{COPILOT_SIM}" %*\n', encoding="utf-8")
    return str(shim)


def _make_agent(matmul_task: Task, job_id: int, handler: BuildAndTestHandler, copilot_exe: str) -> CopilotCLIAgent:
    return CopilotCLIAgent(
        task=matmul_task,
        job_id=job_id,
        task_id="test-task",
        config={},
        copilot_exe=copilot_exe,
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not COPILOT_SIM.exists(), reason="copilot-sim not found in repo root")
def test_agent_collects_results_from_copilot_sim(matmul_task, copilot_sim_exe):
    """The agent should collect exactly two (Program, EvalResult) pairs when
    copilot-sim calls build_and_test twice (once with broken kernel, once
    with fixed kernel).
    """
    agent = _make_agent(matmul_task, job_id=1, handler=MockBuildAndTestHandler(), copilot_exe=copilot_sim_exe)

    results = agent.run("Fix the matmul kernel", iteration=0)

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    first_program, first_result = results[0]
    second_program, second_result = results[1]

    assert isinstance(first_program, Program)
    assert isinstance(second_program, Program)
    assert isinstance(first_result, EvalResult)
    assert isinstance(second_result, EvalResult)

    # copilot-sim: first call should fail, second should succeed
    assert not first_result.correctness, "First call should have failed"
    assert second_result.correctness, "Second call should have passed"


@pytest.mark.skipif(not COPILOT_SIM.exists(), reason="copilot-sim not found in repo root")
def test_handler_is_called_for_each_build_and_test(matmul_task, copilot_sim_exe):
    """The handler should be called once per build_and_test invocation."""
    handler = MockBuildAndTestHandler()

    agent = _make_agent(matmul_task, job_id=2, handler=handler, copilot_exe=copilot_sim_exe)
    agent.run("Fix the matmul kernel", iteration=0)

    assert len(handler.call_log) == 2, f"Expected handler to be called 2 times, got {len(handler.call_log)}"
    assert all("success" in r for r in handler.call_log)


@pytest.mark.skipif(not COPILOT_SIM.exists(), reason="copilot-sim not found in repo root")
def test_fork_preserves_task_and_config(matmul_task, copilot_sim_exe):
    """Forking after a run should preserve the task, job_id, and session state."""
    agent = _make_agent(matmul_task, job_id=3, handler=MockBuildAndTestHandler(), copilot_exe=copilot_sim_exe)

    agent.run("Fix the matmul kernel", iteration=0)
    session_state = agent.session_state()

    forked = agent.fork(branch=1)

    assert forked._task is agent._task
    assert forked._job_id == agent._job_id
    assert forked._initial_session_state == session_state


@pytest.mark.skipif(not COPILOT_SIM.exists(), reason="copilot-sim not found in repo root")
def test_session_state_is_json_serializable(matmul_task, copilot_sim_exe):
    """session_state() should return a JSON-serialisable dict."""
    agent = _make_agent(matmul_task, job_id=4, handler=MockBuildAndTestHandler(), copilot_exe=copilot_sim_exe)

    agent.run("Fix the matmul kernel", iteration=0)
    state = agent.session_state()

    # Should not raise
    serialized = json.dumps(state)
    assert isinstance(serialized, str)
    assert isinstance(json.loads(serialized), dict)
