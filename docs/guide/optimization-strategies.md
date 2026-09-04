# Optimization strategies

A menu of **optional** strategies for getting more out of a run. The example task is
deliberately simple, a few iterations with one model, so these are the levers to reach for when
a baseline run plateaus.

Each strategy gives you two ways to apply it: a line to paste into a coding agent's prompt, or
the equivalent config yourself. If you are working through an agent, it can set these up for
you and write them into your task's `USER_INSTRUCTIONS` block or `config.yaml`. See
[Optimize a kernel with a coding agent](agentic-workflow.md).

:::{tip}
You can combine several. Paste more than one *Ask the agent* line, or set the config keys
together.
:::

## Widen the search

**When:** the run keeps making small refinements to one lineage and stops improving. Evolutionary
search samples a parent kernel and explores several branches per iteration instead.

**Ask the agent:**

```
Use evolutionary search: explore about 3 branches per iteration over roughly 15 iterations.
```

**Or set it yourself:**

```yaml
branches_per_iteration: 3
max_iters: 15
```

Cost scales with `branches_per_iteration × max_iters` LLM calls and benchmark runs, so this is
the main knob on how much compute and API spend a job consumes.

## Start from the best kernel so far

**When:** you already produced a good kernel for this `task_name` and `gpu_arch` in an earlier
job, and want to continue rather than start cold.

**Ask the agent:**

```
Continue from the best existing kernel for this task instead of starting cold.
```

**Or set it yourself:**

```yaml
start_from_best: true
```

To start from a specific file instead, point `kernels_iter_0_path` at it.

## Use an ensemble of models

**When:** you want the strongest and most varied generation. Different models fail differently,
so an ensemble explores more of the design space than repeated calls to one model.

**Ask the agent:**

```
Generate with an ensemble of a few strong models instead of a single model.
```

**Or set it yourself:** list several servers under `inference`. See
[`kernelfoundry/configs/inference/ensemble.yaml`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/configs/inference/ensemble.yaml)
for a worked example, and [Config parameters](config-parameters.md) for the fields.

## Let the model write templated kernels

**When:** the best tiling, vector width or work-group size is not obvious. With
`allow_templated`, the model can express a kernel with several parameter options and
KernelFoundry benchmarks every combination, tuning to your inputs and hardware rather than
guessing.

**Ask the agent:**

```
Write a templated kernel with tunable parameters and let KernelFoundry benchmark the options.
```

**Or set it yourself:**

```yaml
prompt:
  allow_templated: true   # off by default
```

:::{warning}
This works for `SYCL` and `CUDA` only. KernelFoundry has to show the model a worked example of
a templated kernel, and examples ship for those two languages alone. Enabling it for `OCL` or
`triton` aborts the run with `AssertionError: Must provide example if allow_templated=True`.
:::

## Feed profiler evidence back more aggressively

**When:** the model is guessing at bottlenecks. Profiling the reference as well as the candidate
gives it a baseline to compare against.

**Or set it yourself:**

```yaml
eval_config:
  profile_original_model: true
```

Profiler feedback needs a profiler installed and non-root profiling enabled; see the README.

## Summarize long evaluation logs

**When:** builds fail with very long compiler output and the model appears not to notice the
real error. A second LLM condenses the log before it is fed back.

**Or set it yourself:**

```yaml
use_feedback_llm: true
```

## Benchmark across several architectures

**When:** the kernel must be fast on more than one target, and you want to avoid tuning into a
shape that only wins on one.

**Or set it yourself:**

```yaml
gpu_arch: lnl,ptl,bmg
```

## Write better user instructions

The cheapest strategy, and often the most effective. The `USER_INSTRUCTIONS` block is passed
straight into the prompt, so it is where domain knowledge belongs: which optimizations you
already know pay off, constraints the kernel must respect, data layouts that cannot change,
and what the tests actually measure. See
[Anatomy of a task package](task-package.md).
