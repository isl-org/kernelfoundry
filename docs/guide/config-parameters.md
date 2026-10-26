# Config parameters

Each task's `config.yaml` sets how KernelFoundry runs it. This page covers the parameters worth
knowing, grouped by section.

The **Required** and **Common top-level** parameters below are all most users need. The later
sections (inference, prompt, evolutionary database) are reference for tuning.

KernelFoundry uses [Hydra](https://hydra.cc/) for hierarchical configuration, so parameters are
grouped into topical sections. Anything you set in a task's `config.yaml` is merged over the
defaults in
[`kernelfoundry/configs/run.yaml`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/configs/run.yaml),
which is the authoritative list.

:::{important}
A task `config.yaml` may only set keys that already exist in `kernelfoundry/configs/run.yaml`. Anything else
aborts the run with `ConfigKeyError: Key '<name>' is not in struct`. This is deliberate: it
catches typos rather than silently ignoring them.
:::

Every parameter can also be overridden on the command line:

```bash
python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom \
    job_name=my_job gpu_arch=lnl language=SYCL max_iters=10
```

## Required parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `task_name` | string | Name for the operation the kernel implements, e.g. `relu`. Groups jobs in the results database. |
| `job_name` | string | Name for one execution of the algorithm. Appears in the UI. |
| `task` | string | The task package to run: a directory, tar archive or zip file. |
| `task_origin` | string | Where the task comes from. Use `custom` for your own task package; `KernelBench` or `robust_kbench` pull a task from the bundled dataset instead of from `task`. Recorded with the results either way. |

`task_name` and `job_name` have no defaults at all; a run that omits either aborts immediately.

## Common top-level parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `language` | string | Kernel language, which determines the compiler and profiler used. One of `SYCL`, `CUDA`, `OCL`, `triton`, `CM` (Intel explicit-SIMD "C for Metal"), or `vulkan`. Inferred from the detected GPU when unset: CUDA on NVIDIA, SYCL on Intel. | inferred |
| `gpu_arch` | string | Architecture to benchmark on. Intel: `lnl` (Lunar Lake), `ptl` (Panther Lake), `mtl` (Meteor Lake), `arl` (Arrow Lake), `rpl` (Raptor Lake), `dg2` (Flex 170 / Arc A770), `bmg` (Battlemage, alias for `b580`), `b580`, `b70`, `b60`, `pvc` (Ponte Vecchio). NVIDIA (requires `language: CUDA`): `Ampere`/`ampere`, `A100`, `A6000`, `Maxwell`, `Pascal`, `Volta`, `Turing`, `Hopper`, `Ada`, `L40S`, `L4`, `native`. Every value gets hardware-specific prompt context except `native`, which has no recorded specs; a KernelBench speedup figure is only recorded for `Ampere`, `ptl`, `bmg`/`b580` and `dg2`, not the rest. Accepts a comma-separated list to benchmark on several. Inferred when unset: `Ampere` on NVIDIA, `dg2` on Intel. | inferred |
| `max_iters` | int | Maximum optimization iterations. `0` evaluates without generating anything. | `3` |
| `branches_per_iteration` | int | Kernels generated per iteration (LLM calls). **Above 1 this enables evolutionary search**: a parent kernel is sampled and multiple branches explored. At 1 the algorithm simply improves the previous kernel. | `2` |
| `stop_once_correct` | bool | Stop as soon as a correct kernel is found. Useful for pure translation tasks. | `false` |
| `build_timeout` | int | Compilation timeout, seconds. Raise if you see build timeouts. | `200` |
| `test_timeout` | int | Timeout for execution, correctness testing, benchmarking and profiling, seconds. Raise if tests load models or data. | `300` |
| `test_reference` | bool | Whether to test and benchmark the reference. Needed to compute speedup; set false only if the reference exists for the prompt but is not executable. | `true` |
| `has_build_step` | bool | Whether the task has a `build` function to compile the kernel. | `true` |
| `has_reference_build_step` | bool | Whether the reference needs building. False when the reference is e.g. PyTorch. | `true` |
| `use_feedback_llm` | bool | Use a second LLM to summarize evaluation logs such as compile errors before feeding them back. | `false` |
| `start_from_best` | bool | Start from the best kernel recorded for this `task_name` rather than cold. Useful to continue a promising run. | `false` |
| `validate` | bool | Evaluate the kernel currently in the evolve block. Combine with `max_iters: 0` to benchmark without generating. | `false` |
| `profiler_kernel` | string or null | Profiler to run on the generated kernel. `default` picks the language's recommended profiler (SYCL/CM → unitrace, OCL → vtune, CUDA → ncu); `null` disables profiling for the kernel entirely; or name one explicitly (e.g. `unitrace`, `vtune`, `ncu`) to force it regardless of language default. `vulkan` has no profiler implemented yet, so `default` yields no profiling for it. | `default` |
| `profiler_reference` | string or null | Same options as `profiler_kernel`, applied to the reference implementation instead. | `null` |

There is no `evolve_mode` parameter; evolutionary search is derived from
`branches_per_iteration > 1`.

(supported-gpu-targets)=
### Supported GPU targets

Set `gpu_arch` to a value below, using the capitalization shown. Aliases in the same row
select the same GPU target. `bmg` is a backward-compatible alias for `b580`.

| Intel target | GPU |
| --- | --- |
| `dg2` | Intel Flex 170 / Arc A770 |
| `b580`, `bmg` | Intel Arc B580 (Battlemage) |
| `b70` | Intel Arc Pro B70 |
| `b60` | Intel Arc Pro B60 |
| `lnl` | Intel Lunar Lake |
| `ptl` | Intel Panther Lake |
| `mtl` | Intel Meteor Lake |
| `arl` | Intel Arrow Lake |
| `rpl` | Intel Raptor Lake |
| `pvc` | Intel Data Center GPU Max (Ponte Vecchio) |

| NVIDIA target | GPU |
| --- | --- |
| `Ampere`, `ampere`, `A6000` | NVIDIA RTX A6000 |
| `A100` | NVIDIA A100 |
| `Maxwell` | NVIDIA Tesla M40 |
| `Pascal` | NVIDIA Tesla P100 |
| `Volta` | NVIDIA Tesla V100 |
| `Turing` | NVIDIA T4 |
| `Hopper` | NVIDIA H100 |
| `Ada`, `L40S` | NVIDIA L40S |
| `L4` | NVIDIA L4 |

## Hierarchical structure

Sections nest under a top-level key:

```yaml
prompt:
  reference_language: SYCL

inference:
  provider: openai
  model: gpt-5
  temperature: 0.3
```

That config tells the prompt the reference is SYCL and generates with one OpenAI model at temperature 0.3.

## Task hyperparameters (`hyperparameters`)

Values your own task needs, passed through untouched. KernelFoundry does not interpret them;
it only delivers them to your build function and your tests, so you can vary a tile size or a
problem shape from `config.yaml` instead of editing `task.py`.

```yaml
hyperparameters:
  buildtime:
    tile_size: 32
  runtime:
    batch_size: 8
```

`buildtime` is expanded as keyword arguments into your task's build function, alongside the
`gpu_arch` it always receives. The example above calls `build(gpu_arch=..., tile_size=32)`, so
every key must be a parameter that function accepts. An unexpected one raises `TypeError`.

`runtime` is JSON-encoded and handed to pytest as `--runtime_params`. Read it in a test with:

```python
import json

def test_something(self, request):
    params = json.loads(request.config.getoption("--runtime_params") or "{}")
    batch_size = params.get("batch_size", 1)
```

For a value needed at collection time, in a `@pytest.mark.skipif` condition for example, the
argument is not yet parsed, so use `kernelfoundry.conftest.get_runtime_params_from_argv()`
instead, which reads it straight from `sys.argv`.

Both default to `null`, and both are recorded with the run so a result can be traced back to
the values that produced it.

## LLM inference (`inference`)

Configures the model(s) used for generation. `inference` is always a small ensemble under the
hood (`model_1`, `model_2`, `model_3`, weighted by the matching entries of `weights`), but
`model_2`/`model_3` are disabled (`provider: null`) by default, so most configs only ever
touch the top-level `provider`/`model` shortcut, which is just `model_1`:

```yaml
inference:
  provider: openai
  model: gpt-5
```

The open-source `provider: openai` and `provider: anthropic` read `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` from the environment respectively. Any OpenAI-compatible endpoint can be
targeted by setting `base_url`.

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `provider` | string | `openai`, `anthropic`, or `claude_cli` (generates via a Claude subscription instead of an API key — see [LLM setup](llm-setup.md); internal deployments may register more). | — |
| `model` | string | Model identifier. `default` picks the first known model for that provider. | `default` |
| `max_tokens` | int | Maximum tokens to generate. | `16000` |
| `temperature` | float | Sampling temperature; `0.0` is deterministic. | `0.0` |
| `num_completions` | int | Completions per call. Requires `temperature > 0` above 1. | `1` |
| `verbose` | bool | Log inference details. | `false` |

For an ensemble, enable `model_2` and/or `model_3` with the same fields, and give `weights` a
matching entry per slot (selection probability, normalized automatically).

```yaml
inference:
  model_1:
    provider: openai
    model: gpt-5.4
  model_2:
    provider: anthropic
    model: claude-sonnet-5
  weights: [0.3, 0.7]
```

See
[`kernelfoundry/configs/inference/server.yaml`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/configs/inference/server.yaml)
for the full schema. Any model your provider offers works; `model` is passed straight through,
so there is no list to keep in step with provider catalogues. The `DEFAULT_MODELS` table at the
top of
[`inference_server.py`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/algorithm/inference_server.py)
is only the fallback consulted when `model` is left as `default`; it is not a validation list.

Power users can still bypass all of this with an explicit `_target_` (a raw
`hydra.utils.instantiate`-style config), which is used as-is.

## Prompt (`prompt`)

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `reference_language` | string | Language the reference is written in. | `Pytorch` |
| `num_optimization_tips` | int | High-level optimization strategies sampled into the prompt. | `2` |
| `include_inspirations` | bool | Include prior generated kernels as inspiration. Only meaningful when evolutionary search is active. | `true` |
| `include_best_program` | bool | Include the best kernel so far for reference. Only meaningful when evolutionary search is active. | `true` |
| `include_hardware_specs` | bool | Include target hardware specifications. | `true` |
| `allow_templated` | bool | Let the model write templated kernels with multiple parameter options; all options get benchmarked. Only SYCL and CUDA ship the worked example this requires; enabling it for `OCL` or `triton` aborts the run. See [Optimization strategies](optimization-strategies.md). | `false` |

## Evolutionary database (`database`)

Active when `branches_per_iteration > 1`. Tunes the MAP-Elites quality-diversity search.

```yaml
branches_per_iteration: 3

database:
  config:
    exploration_ratio: 0.3
    num_top_programs: 2
```

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `num_top_programs` | int | Top-performing programs included in the prompt. | `1` |
| `num_diverse_programs` | int | Diverse programs included in the prompt. | `0` |
| `num_inspirations` | int | Inspiration examples drawn from the archive. | `2` |
| `population_size` | int | Total population. | `1000` |
| `archive_size` | int | Size of the elite archive. | `100` |
| `num_islands` | int | Evolutionary islands, for maintaining diversity. | `4` |
| `programs_per_island` | int | Programs per island before switching. | `10` |
| `elite_selection_ratio` | float | Fraction selected as elites. | `0.1` |
| `exploration_ratio` | float | Exploration versus exploitation balance. | `0.2` |

Defaults for every field are in
[`kernelfoundry/configs/database/evolve_db_optimization_aware.yaml`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/configs/database/evolve_db_optimization_aware.yaml).

For the strategies these parameters serve, see
[Optimization strategies](optimization-strategies.md).

## Skills (`skills`)

A skill is a directory holding a `SKILL.md` file — YAML frontmatter plus a Markdown body,
following the [Agent Skills specification](https://agentskills.io/specification) — and optionally
bundled `scripts/`, `references/` and `assets/`. Skills are filtered by the
`metadata.programming-languages` field against the run's `language`, so a SYCL run sees only SYCL
skills.

```yaml
skills:
  directories:
    - ./skills
```

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `directories` | list | Directories scanned recursively for `SKILL.md` files. Relative paths resolve against the working directory. | `["./skills"]` |

Both workflows load the `skills: default` group via `kernelfoundry/configs/base.yaml`, so
`./skills` is scanned by default whether you run the classic search or the agentic one. Override
`skills.directories` in a task's `config.yaml` to point elsewhere, or set it to `[]` to load no
skills at all.

The two workflows use a skill differently, because one has tools and the other does not. The
agentic workflow hands the agent the skill directory and lets it open what it needs, which is the
progressive disclosure the specification describes. The classic search has no tools, so its prompt
inlines each `SKILL.md` body and leaves the bundled resources on disk; a skill that would push the
prompt past the budget is left out whole rather than truncated, and says so in the log.
