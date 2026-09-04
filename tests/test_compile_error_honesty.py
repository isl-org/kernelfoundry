import logging
import pytest

from kernelfoundry.compiler import TorchCompiler
from kernelfoundry.eval_pipeline.task import ProcessResult

# Aliased: pytest tries to collect any module-level name starting with `Test` as a test class,
# and warns that it cannot because of the constructor. The alias keeps the output clean.
from kernelfoundry.eval_pipeline.task import TestResult as VariantResult
from kernelfoundry.eval_pipeline.tasks.test_custom_task import _log_outcome_summary


class TestFailedPhase:
    """The phase classification the error message depends on."""

    def test_success_has_no_failed_phase(self):
        assert TorchCompiler.failed_phase(0) is None

    def test_import_failure_is_the_load_phase(self):
        assert TorchCompiler.failed_phase(TorchCompiler.LOAD_FAILED_RETURNCODE) == "load"

    @pytest.mark.parametrize("returncode", [1, 2, -1, 255])
    def test_anything_else_is_the_compile_phase(self, returncode):
        assert TorchCompiler.failed_phase(returncode) == "compile"

    def test_the_load_code_does_not_collide_with_ordinary_failures(self):
        """It has to be distinguishable from the generic exit 1, which is the whole point."""
        assert TorchCompiler.LOAD_FAILED_RETURNCODE not in (0, 1, 2)


def _result(returncode: int) -> ProcessResult:
    return ProcessResult(returncode=returncode, stdout="", stderr="")


def _summary(caplog, results) -> str:
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _log_outcome_summary(results)
    return caplog.text


def test_a_failed_candidate_is_stated_after_the_reference(caplog):
    """The candidate failed and the reference passed: the summary must not read as success."""
    results = {
        "custom": VariantResult(correctness_result=_result(1)),
        "reference": VariantResult(correctness_result=_result(0)),
    }
    text = _summary(caplog, results)

    assert "RESULT: 1 of 2 variant(s) failed: custom" in text
    assert "FAILED (exit 1)" in text
    # The reference passing must still be visible -- it is true, just not the headline.
    assert "reference" in text


def test_an_all_passing_run_says_so(caplog):
    results = {"custom": VariantResult(correctness_result=_result(0))}
    assert "RESULT: all 1 variant(s) passed" in _summary(caplog, results)


def test_every_variant_failing_is_reported(caplog):
    results = {
        "custom_32_32": VariantResult(correctness_result=_result(1)),
        "custom_64_64": VariantResult(correctness_result=_result(1)),
    }
    text = _summary(caplog, results)
    assert "RESULT: 2 of 2 variant(s) failed" in text
    assert "custom_32_32" in text and "custom_64_64" in text


def test_a_variant_that_never_ran_is_not_counted_as_a_failure(caplog):
    """Absent is not the same as failed, and conflating them would overstate the damage."""
    results = {"custom": VariantResult(correctness_result=None)}
    text = _summary(caplog, results)
    assert "NOT RUN" in text
    assert "failed" not in text.lower().split("result:")[-1]


class TestEvalLogDoesNotMaskTheRealError:
    r"""A missing evaluation log must not replace the reason the run failed.

    The log is only written once evaluation gets far enough to produce output. When a run failed
    earlier than that, `open(artifact_path)` raised FileNotFoundError and that became the reported
    error -- naming a file that was never going to exist, while the actual cause sat unread in
    `exec_result.metadata`. Round 4 hit this on Windows as

        FileNotFoundError: [Errno 2] No such file or directory:
        'runs\test_controller\stdout_level_1_problem_19_ReLU_trial_0_v0.txt'

    where the real failure was "Reference test failed, cannot proceed to testing generated kernel".
    """

    @staticmethod
    def _result(**kwargs):
        from kernelfoundry.algorithm.schemas import EvalResult

        kwargs.setdefault("compiled", False)
        kwargs.setdefault("correctness", False)
        kwargs.setdefault("perf_score", 0)
        return EvalResult(**kwargs)

    def test_missing_log_keeps_the_evaluator_reason(self, tmp_path):
        from kernelfoundry.algorithm.schemas import Program

        reason = "Reference test failed, cannot proceed to testing generated kernel"
        exec_result = self._result(metadata={"error": reason})
        missing = str(tmp_path / "never_written.txt")

        text = Program._read_eval_log(missing, exec_result)

        assert reason in text, "the real cause must survive"
        assert missing in text, "and the reader should still be told which log was absent"

    def test_missing_log_without_a_reason_still_explains_itself(self, tmp_path):
        from kernelfoundry.algorithm.schemas import Program

        text = Program._read_eval_log(str(tmp_path / "never_written.txt"), self._result(metadata={}))

        assert "no log" in text.lower() or "produced no log" in text.lower()

    def test_a_log_that_exists_is_read_verbatim(self, tmp_path):
        from kernelfoundry.algorithm.schemas import Program

        log = tmp_path / "stdout.txt"
        log.write_text("[1/2] nvcc ...\nBuild succeeded.\n", encoding="utf-8")

        assert Program._read_eval_log(str(log), self._result(metadata={})) == log.read_text(encoding="utf-8")

    def test_utf8_log_survives_a_cp1252_default(self, tmp_path):
        """Compilers emit UTF-8; the Windows console default is cp1252 and raises on byte 0x9d."""
        from kernelfoundry.algorithm.schemas import Program

        log = tmp_path / "stdout.txt"
        log.write_bytes("error: unexpected ’quote’ — and an em dash\n".encode("utf-8"))

        text = Program._read_eval_log(str(log), self._result(metadata={}))

        assert "unexpected" in text and "em dash" in text


class TestWorkerInfoToleratesNonMappingMetadata:
    """`metadata` is not exclusively architecture-keyed worker info.

    The evaluator also records failures under it as plain strings which need to be distinguished from the worker info
    """

    @staticmethod
    def _kernel():
        from kernelfoundry.eval_pipeline.database.tables import Kernel

        return Kernel()

    def _populate(self, metadata):
        from kernelfoundry.algorithm.schemas import EvalResult, Program

        program = Program(id="test")
        kernel = self._kernel()
        program.populate_kernel_from_exec_result(
            kernel,
            EvalResult(compiled=False, correctness=False, perf_score=0, metadata=metadata),
        )
        return kernel

    def test_a_string_valued_entry_does_not_raise(self):
        kernel = self._populate({"error": "Reference test failed, cannot proceed"})

        assert kernel.compile_worker_info is None
        assert kernel.eval_worker_info is None

    def test_worker_info_is_still_collected(self):
        kernel = self._populate({"Ampere": {"compile_worker_info": "host-a", "eval_worker_info": "host-b"}})

        assert kernel.compile_worker_info == {"Ampere": "host-a"}
        assert kernel.eval_worker_info == {"Ampere": "host-b"}

    def test_worker_info_survives_alongside_a_string_entry(self):
        """The mixed case is the real one: an error recorded on a run that also reached a worker."""
        kernel = self._populate(
            {
                "error": "Reference test failed, cannot proceed",
                "Ampere": {"compile_worker_info": "host-a"},
            }
        )

        assert kernel.compile_worker_info == {"Ampere": "host-a"}
