"""The compiler's register report must reach the evaluation log.

A build can succeed and still say the kernel spilled. That line used to go to the raw build
dump and be dropped before the log the next prompt is built from, so the model never saw it.
On a setup where the profiler collects nothing, it is the only resource feedback there is.

The lines are logged verbatim: the compiler's own wording is clearer than anything rebuilt from
it, and the whole build output is far too large to log instead -- one task build here is 23 KB
against a median evaluation log of 565 characters.
"""

from types import SimpleNamespace

import pytest

from kernelfoundry.algorithm.evaluator import Evaluator

SPILLED = (
    "warning: kernel _ZTS8Q40MmBig  compiled SIMD16 allocated 256 regs and spilled around 225\n"
    "warning: kernel _ZTS10Q40MmSmall  compiled SIMD16 allocated 256 regs and spilled around 225\n"
)
CLEAN = "warning: kernel _ZTS8Q40MmBig  compiled SIMD16 allocated 128 regs\n"
RETRY = "warning: [RetryManager] Start recompilation of the kernel\n" "in kernel: 'typeinfo name for Q40MmBig'\n"
NOISE = (
    "icpx: note: some unrelated remark\n" "warning: 'parallel_for' is deprecated: use a launch configuration object\n"
)


@pytest.fixture
def evaluator():
    ev = Evaluator.__new__(Evaluator)  # no config or logger needed for this
    ev._logged = []
    ev.log = ev._logged.append
    return ev


def _build(stdout="", stderr=""):
    return SimpleNamespace(result=SimpleNamespace(stdout=stdout, stderr=stderr))


def logged(ev):
    return "\n".join(ev._logged)


def test_spill_is_reported(evaluator):
    evaluator.log_compiler_resource_report(_build(stderr=SPILLED))
    text = logged(evaluator)
    assert "spilled around 225" in text
    assert "Q40MmBig" in text and "Q40MmSmall" in text


def test_registers_are_reported_without_spill(evaluator):
    evaluator.log_compiler_resource_report(_build(stderr=CLEAN))
    assert "allocated 128 regs" in logged(evaluator)


def test_a_retry_is_reported_with_its_kernel(evaluator):
    evaluator.log_compiler_resource_report(_build(stderr=RETRY))
    text = logged(evaluator)
    assert "RetryManager" in text
    assert "Q40MmBig" in text  # the kernel name is on the following line


def test_repeated_lines_are_not_repeated_in_the_log(evaluator):
    evaluator.log_compiler_resource_report(_build(stderr=SPILLED + SPILLED))
    assert logged(evaluator).count("_ZTS8Q40MmBig") == 1


def test_unrelated_output_is_left_out(evaluator):
    evaluator.log_compiler_resource_report(_build(stderr=NOISE))
    assert evaluator._logged == []


def test_only_the_resource_lines_survive(evaluator):
    evaluator.log_compiler_resource_report(_build(stdout=NOISE, stderr=NOISE + SPILLED))
    text = logged(evaluator)
    assert "deprecated" not in text and "unrelated remark" not in text
    assert "spilled around 225" in text


def test_nothing_is_logged_without_a_build(evaluator):
    evaluator.log_compiler_resource_report(None)
    evaluator.log_compiler_resource_report(SimpleNamespace(result=None))
    assert evaluator._logged == []


def test_stdout_is_searched_as_well_as_stderr(evaluator):
    evaluator.log_compiler_resource_report(_build(stdout=SPILLED))
    assert "spilled around 225" in logged(evaluator)
