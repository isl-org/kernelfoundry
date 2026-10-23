"""ClaudeCLIServer: contract, and the auth scrub that keeps runs off metered billing."""

import json
import subprocess
from pathlib import Path

import pytest

from kernelfoundry.algorithm.claude_cli_server import (
    _AUTH_ENV_VARS,
    ClaudeCLIError,
    ClaudeCLIServer,
)


def _cli_reply(text="kernel code", input_tokens=10, output_tokens=20, stop_reason="end_turn", num_turns=1):
    return json.dumps(
        {
            "result": text,
            "is_error": False,
            "stop_reason": stop_reason,
            "num_turns": num_turns,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    return ClaudeCLIServer(model_name="opus")


def test_accepts_the_kwargs_an_ensemble_slot_passes(monkeypatch):
    """configs/inference/server.yaml's model_1/model_2/model_3 slots pass server_type and
    base_url alongside model_name/max_tokens/temperature/num_completions/verbose -- none of
    which ClaudeCLIServer's own __init__ names, so they must land harmlessly via **kwargs."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    server = ClaudeCLIServer(
        server_type="claude_cli",
        model_name="opus",
        base_url=None,
        max_tokens=16000,
        temperature=0.0,
        num_completions=1,
        verbose=False,
    )
    assert server.model == "opus"


class _FakePopen:
    """Stands in for the CLI process.

    The server writes stdout and stderr to files rather than pipes (a pipe kept open by a
    grandchild makes the timeout hang), so the stub writes into the handles it is given.
    """

    def __init__(self, recorder, stdout_text, returncode, stderr_text, cmd, kwargs):
        self.returncode = returncode
        self._out = kwargs.get("stdout")
        self._err = kwargs.get("stderr")
        self._stdout_text = stdout_text
        self._stderr_text = stderr_text
        self.pid = 4242
        self._record = {"cmd": cmd, **kwargs}
        if "--system-prompt-file" in cmd:
            # Read now: the file lives in a temporary directory removed after the call.
            self._record["system_prompt"] = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        recorder.append(self._record)

    def communicate(self, input=None, timeout=None):
        self._record["input"] = input
        self._out.write(self._stdout_text)
        self._err.write(self._stderr_text)
        self._out.flush()
        self._err.flush()
        return None, None

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def _run_stub(recorder, stdout, returncode=0, stderr=""):
    def _popen(cmd, **kwargs):
        return _FakePopen(recorder, stdout, returncode, stderr, cmd, kwargs)

    return _popen


def test_returns_outputs_and_metadata(server, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply()))
    outputs, metadata = server([{"role": "user", "content": "write a kernel"}])
    assert outputs == ["kernel code"]
    assert metadata["input_tokens"] == 10
    assert metadata["output_tokens"] == 20
    assert metadata["truncated_completions"] == []
    assert metadata["model"] == "opus"


def test_api_key_is_stripped_from_the_subprocess(server, monkeypatch):
    for name in _AUTH_ENV_VARS:
        monkeypatch.setenv(name, "should-not-reach-the-cli")
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply()))
    server([{"role": "user", "content": "hi"}])
    child_env = calls[0]["env"]
    # An inherited key silently outranks the subscription login and moves the run onto
    # metered billing, warning only on stderr.
    for name in _AUTH_ENV_VARS:
        assert name not in child_env


def test_system_messages_go_to_the_system_prompt_file(server, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply()))
    server(
        [
            {"role": "system", "content": "you write SYCL"},
            {"role": "user", "content": "write a kernel"},
        ]
    )
    assert "you write SYCL" in calls[0]["system_prompt"]
    assert calls[0]["input"] == "write a kernel"
    # claude.CMD truncates its arguments at a newline, dropping every flag after it.
    assert not any("\n" in arg for arg in calls[0]["cmd"])


def test_num_completions_runs_the_cli_once_per_completion(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    server = ClaudeCLIServer(model_name="opus", num_completions=3)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply()))
    outputs, metadata = server([{"role": "user", "content": "hi"}])
    assert len(calls) == 3
    assert len(outputs) == 3
    assert metadata["output_tokens"] == 60


def test_single_question_ignores_num_completions(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    server = ClaudeCLIServer(model_name="opus", num_completions=3)
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply()))
    outputs, _ = server([{"role": "user", "content": "hi"}], single_question=True)
    assert len(calls) == 1
    assert len(outputs) == 1


def test_failure_reports_stdout_not_just_stderr(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    server = ClaudeCLIServer(model_name="opus", max_retry=1)
    calls = []
    # The CLI writes its error as JSON on stdout and leaves stderr empty.
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _run_stub(calls, '{"stop_reason":"tool_use","result":null}', returncode=1, stderr=""),
    )
    with pytest.raises(ClaudeCLIError, match="tool_use"):
        server([{"role": "user", "content": "hi"}])


def test_truncated_completion_is_reported(server, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply(stop_reason="max_tokens")))
    _, metadata = server([{"role": "user", "content": "hi"}])
    assert metadata["truncated_completions"] == [0]


def test_missing_cli_fails_at_construction(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ClaudeCLIError, match="not found on PATH"):
        ClaudeCLIServer(model_name="opus")


def test_an_agent_transcript_is_not_accepted_as_a_completion(monkeypatch):
    """The CLI can run its agent loop despite the no-tools directive.

    What comes back is then the last turn of a transcript -- "that run is already going in the
    background" -- which reads as ordinary prose and is spliced into the kernel, where it fails
    as `unknown type name 'The'`. Four kernels in real SYCL runs were lost this way.
    """
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    srv = ClaudeCLIServer(model_name="opus", max_retry=1)
    calls = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        _run_stub(calls, _cli_reply(text="Nothing new to request right now.", num_turns=4)),
    )
    with pytest.raises(ClaudeCLIError, match="agent transcript"):
        srv([{"role": "user", "content": "write a kernel"}])


def test_a_single_turn_reply_is_accepted(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    srv = ClaudeCLIServer(model_name="opus")
    calls = []
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, _cli_reply(num_turns=1)))
    outputs, _ = srv([{"role": "user", "content": "write a kernel"}])
    assert outputs == ["kernel code"]


def test_a_reply_without_a_turn_count_is_accepted(monkeypatch):
    """Older CLI versions may not report num_turns; absence must not fail the call."""
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    srv = ClaudeCLIServer(model_name="opus")
    calls = []
    payload = json.loads(_cli_reply())
    payload.pop("num_turns", None)
    monkeypatch.setattr(subprocess, "Popen", _run_stub(calls, json.dumps(payload)))
    outputs, _ = srv([{"role": "user", "content": "write a kernel"}])
    assert outputs == ["kernel code"]
