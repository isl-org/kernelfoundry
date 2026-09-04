r"""
A completion cut off by max_tokens must be reported as truncated, not as bad code.
"""

import logging
import types

import pytest

from kernelfoundry.algorithm.inference_server import TRUNCATION_STOP_REASONS, InferenceServer


class TestTruncationReasons:
    def test_anthropic_and_openai_spellings_are_both_covered(self):
        assert "max_tokens" in TRUNCATION_STOP_REASONS, "Anthropic's spelling"
        assert "length" in TRUNCATION_STOP_REASONS, "the OpenAI-compatible spelling"

    @pytest.mark.parametrize("reason", ["end_turn", "stop", "tool_use", "stop_sequence", None])
    def test_ordinary_completions_are_not_truncation(self, reason):
        assert reason not in TRUNCATION_STOP_REASONS


class TestReporting:
    """_report_truncation is the whole user-visible behaviour, so it is tested directly."""

    @staticmethod
    def _server():
        # Built without __init__: constructing one would require a provider client and an API key,
        # and none of that is involved in deciding what to say about a truncated response.
        server = InferenceServer.__new__(InferenceServer)
        server.model = "claude-opus-5"
        server.server_args = {"max_tokens": 5000}
        return server

    def test_nothing_is_logged_when_nothing_was_truncated(self, caplog):
        with caplog.at_level(logging.DEBUG):
            self._server()._report_truncation([], n_outputs=2, max_tokens=5000)
        assert caplog.text == ""

    def test_a_truncated_completion_is_an_error_naming_the_cause(self, caplog):
        with caplog.at_level(logging.ERROR):
            self._server()._report_truncation([0], n_outputs=2, max_tokens=5000)

        text = caplog.text
        assert "1 of 2" in text, "how many were cut off"
        assert "max_tokens" in text, "why"
        assert "5000" in text, "what the ceiling actually was"
        assert "claude-opus-5" in text, "which model"
        assert "compile" in text.lower(), "and what the reader will otherwise blame"

    def test_the_thinking_caveat_is_stated(self, caplog):
        """The reason a previously-adequate ceiling stopped being adequate."""
        with caplog.at_level(logging.ERROR):
            self._server()._report_truncation([0], n_outputs=1, max_tokens=5000)
        assert "think" in caplog.text.lower()


class TestDetection:
    """The shapes each provider returns, reduced to what the detection reads."""

    def test_anthropic_stop_reason_is_detected(self):
        response = types.SimpleNamespace(stop_reason="max_tokens")
        assert getattr(response, "stop_reason", None) in TRUNCATION_STOP_REASONS

    def test_openai_finish_reason_is_detected(self):
        choice = types.SimpleNamespace(finish_reason="length")
        assert getattr(choice, "finish_reason", None) in TRUNCATION_STOP_REASONS

    def test_a_response_without_the_field_is_not_treated_as_truncated(self):
        """Not every proxy or test double populates it, and absent must not mean cut off."""
        response = types.SimpleNamespace()
        assert getattr(response, "stop_reason", None) not in TRUNCATION_STOP_REASONS


def test_the_shipped_ceiling_has_headroom_for_a_kernel_plus_thinking():
    """5000 was the value that truncated a real run; the default must not regress to it."""
    from kernelfoundry.algorithm.inference_server import SERVER_PRESETS

    assert SERVER_PRESETS["anthropic"]["max_tokens"] >= 16000
