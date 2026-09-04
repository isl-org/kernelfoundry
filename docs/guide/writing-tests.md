# Writing tests

The tests are the specification. They decide whether a generated kernel is accepted as correct, and
they produce the runtime that speedup is measured from, so the quality of a KernelFoundry run depends
directly on them.

A coding agent normally writes them, reusing your existing tests where they exist; see
[Optimize a kernel with a coding agent](agentic-workflow.md). This page is for writing them yourself,
and for judging whether what an agent produced measures the right thing, which is worth a look either
way because everything downstream rests on it.

The framework is built on `pytest`. KernelFoundry supplies fixtures and helpers that remove
most of the boilerplate; see [Public API](../api/public.rst) for the full list.

**Naming:** tests must start with `test_`, per pytest. Every test is treated as a correctness
test unless it carries the `@pytest.mark.performance` decorator.

## Correctness tests

At least one is required. Tests live in a class deriving from `TestBase`.

The pattern that works for most tasks is to run the kernel and the reference on the same random
inputs and compare:

```python
from kernelfoundry.testing import assert_allclose

def test_correctness(self, data, kernel):
    x, y = data
    result = kernel(x, y)
    expected_result = reference(x, y)
    assert_allclose(result, expected_result)
```

How it works:

- A fixture creates random input data, for example two torch tensors `x`, `y`.
- The kernel and the reference both run on that data.
- `assert_allclose` compares the results with tolerances suited to GPU floating point.

For most tasks you can use this as-is and only change the `reference` function, the `kernel`
fixture (which imports the compiled kernel module), and the `data` fixture.

:::{tip}
Make the input data representative. A kernel tuned against a single small shape will often
regress on the shapes you actually run in production. Parametrize with `@pytest.mark.parametrize`
to cover several shapes.
:::

## Benchmark test

You must define **exactly one** benchmark, marked with `@pytest.mark.performance`. Additional
performance-marked functions will run but are ignored when computing runtime and speedup.

The interface between your benchmark and the framework is the `profile_store` fixture, which
records the measured runtimes. The simplest correct benchmark uses the provided helper:

```python
@pytest.mark.performance
def test_benchmark_my_kernel(self, kernel, device, data, measure_runtime_torch):
    measure_runtime_torch(kernel, device, args=data)
```

`measure_runtime_torch` handles moving inputs to the device, warmup trials, and storing results
through `profile_store`. For non-torch workloads use `measure_runtime`.

### Writing a custom benchmark

If you need to measure something the helpers don't cover, your function must:

- use the **`profile_store`** fixture to store results as a list of runtimes; and
- run the timed trials inside the **`profiler_session`** context. Otherwise profiling silently
  produces nothing, because the profiler has no entry point to attach to.

That second requirement is the usual cause of a run that reports timings but no profiler
feedback.

## Debugging your tests

With the package installed, you can iterate locally before submitting anything.

Check that your tests pass against the reference. This validates the task definition itself:

```bash
pytest --ref -s task.py
```

Check that the kernel builds. This requires a working kernel already in the evolve block, and
runs the `build` function of your `TestBase` class to compile it to a shared object:

```bash
python task.py
```

Then run the tests against the compiled kernel:

```bash
pytest -s task.py
```

If the reference fails, fix the task before anything else: every speedup is measured against
it, so a broken reference makes every subsequent number meaningless.
