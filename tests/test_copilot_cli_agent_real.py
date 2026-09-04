"""Manual test script for CopilotCLIAgent with the real copilot binary.

Uses a mock BuildAndTestHandler by default so no GPU hardware is required.
Pass --real-eval to use the actual Evaluator (requires hardware).

Usage (from repo root):
    PYTHONPATH=kernelfoundry python kernelfoundry/tests/test_copilot_cli_agent_real.py
    PYTHONPATH=kernelfoundry python kernelfoundry/tests/test_copilot_cli_agent_real.py --real-eval
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MATMUL_TASK_DIR = Path(__file__).parent.parent / "tasks" / "example_custom"

# ---------------------------------------------------------------------------
# Env vars required for the real copilot binary
# ---------------------------------------------------------------------------

ENV_OVERRIDES = {
    "COPILOT_PROVIDER_BEARER_TOKEN": os.environ.get("GNAI_TOKEN", ""),
    "COPILOT_MODEL": "gpt-5.2-codex",
    "COPILOT_PROVIDER_BASE_URL": "https://gnai.intel.com/api/providers/openai/v1/",
    "COPILOT_PROVIDER_WIRE_API": "responses",
}

PROMPT = (
    "Optimize the SYCL matrix multiplication kernel in matrix_mul_kernel.sycl "
    "for better performance on Intel GPU. Use the build_and_test tool to evaluate "
    "your changes."
)


# ---------------------------------------------------------------------------
# Mock build-and-test handler (no GPU required)
# ---------------------------------------------------------------------------


def _make_mock_handler():
    """Return a BuildAndTestHandler that reports success without running the real Evaluator."""
    from kernelfoundry.algorithm.agent_base import BuildAndTestHandler, EvaluateFunctionResult
    from kernelfoundry.algorithm.schemas import EvalResult, Program

    class MockBuildAndTestHandler(BuildAndTestHandler):
        def __init__(self):
            self.call_count = 0

        def call(
            self,
            task,
            folder_path,
            job_id,
            task_id,
            prompt,
            iteration,
            branch,
            llm_model,
            session_log,
            previous_program=None,
            agent_session_id=None,
        ) -> EvaluateFunctionResult:
            self.call_count += 1
            print(f"  [eval #{self.call_count}] folder_path={folder_path}")

            eval_result = EvalResult(
                compiled=True,
                correctness=True,
                perf_score=5,
                runtime=1.0,
                eval_log="mock: all tests passed",
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

            return EvaluateFunctionResult(
                tool_response={
                    "success": True,
                    "job_id": job_id,
                    "eval_log": eval_result.eval_log,
                },
                eval_result=eval_result,
                program=program,
            )

    return MockBuildAndTestHandler()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-eval", action="store_true", help="Use the real Evaluator (requires GPU hardware)")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout in seconds (default: 600)")
    args = parser.parse_args()

    if not os.environ.get("GNAI_TOKEN"):
        print("ERROR: GNAI_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    from kernelfoundry.eval_pipeline.task import Task
    from kernelfoundry.algorithm.copilot_cli_agent import CopilotCLIAgent

    task, _ = Task.create(MATMUL_TASK_DIR)

    # With --real-eval, fall back to the default BuildAndTestHandler (real Evaluator).
    handler = None if args.real_eval else _make_mock_handler()

    agent = CopilotCLIAgent(
        task=task,
        job_id=1,
        task_id="test-task",
        config={},
        copilot_exe="copilot",
        env_overrides=ENV_OVERRIDES,
        handler=handler,
    )

    print(f"Running CopilotCLIAgent (real-eval={args.real_eval}, timeout={args.timeout}s)...")
    print(f"Task dir: {agent._task_data_dir}")
    print()

    results = agent.run(PROMPT, iteration=0, timeout=args.timeout)

    print(f"\nDone — {len(results)} build_and_test call(s)")
    for i, (program, result) in enumerate(results):
        print(f"  [{i+1}] success={result.correctness}  eval_log={result.eval_log[:120]!r}")


if __name__ == "__main__":
    main()
