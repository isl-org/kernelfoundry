"""Reading profiler output must survive a pass that never started.

A metric group can enumerate on a device and still fail to activate. Unitrace then leaves an
empty or truncated timeline behind, and parsing it used to raise -- which threw away the metric
files sitting in the same directory and logged a traceback for a condition the caller already
handles.
"""

import json

import pytest

from kernelfoundry.eval_pipeline.profiler_command import Unitrace


@pytest.fixture
def profiler(tmp_path):
    return Unitrace(tmp_path, group="MemoryProfile")


def test_valid_timeline_is_returned(profiler, tmp_path):
    (tmp_path / "python.json").write_text(json.dumps({"traceEvents": []}))
    assert profiler.read_output()["timeline"] == {"traceEvents": []}


def test_empty_timeline_does_not_raise(profiler, tmp_path, caplog):
    (tmp_path / "python.json").write_text("")
    with caplog.at_level("WARNING"):
        data = profiler.read_output()
    assert "timeline" not in data
    assert "MemoryProfile" in caplog.text


def test_truncated_timeline_does_not_raise(profiler, tmp_path):
    (tmp_path / "python.json").write_text('{"traceEvents": [{"name": "x"')
    assert "timeline" not in profiler.read_output()


def test_metric_files_survive_a_bad_timeline(profiler, tmp_path):
    """The whole point: a broken timeline must not cost us the metrics next to it."""
    (tmp_path / "python.json").write_text("")
    (tmp_path / "trace.metrics.123").write_text("GlobalInstanceId,X\n1,2\n")
    data = profiler.read_output()
    assert data["trace.metrics.123"].startswith("GlobalInstanceId")
