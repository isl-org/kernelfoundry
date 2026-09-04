# Understanding results

Every kernel KernelFoundry generates, including the failures, is recorded in the results
database along with its prompt, the model's full answer, the measured runtimes and the
evaluation log. That record is the main tool for working out *why* a run went the way it did.

Browse it in the web UI:

```bash
kernelfoundry-gui          # needs the ui extra: pip install 'kernelfoundry[ui]'
python start_gui.py        # equivalent, from a clone
```

It serves `http://localhost:8885`. Find your run by the `task_name` and `job_name` you set in
`config.yaml`.

## The generation graph

The central view is a graph of one job, where every circle is a generated kernel:

- **Colour** is how far the candidate got:

  | Colour | Meaning |
  | --- | --- |
  | Green | Correct **and** faster than the reference |
  | Blue | Correct, but no faster than the reference |
  | Orange | Compiled, but failed a correctness test |
  | Yellow | Compiled, but the tests timed out |
  | Red | Did not compile |
  | Pink | The build itself timed out |
  | White, black outline | The best kernel of the job |

  The green/blue split is the one to read first: blue means the model is producing valid
  kernels and simply not beating the baseline, which is a very different problem from red.

- **Size** is speed: bigger is faster.
- **Arrows** point from parent to child. The kernel at the base of an arrow was included in the
  prompt as inspiration for the one it points to.
- **Labels** read `i<N>-b<M>`: iteration *N*, branch *M*. So `i0-b0` through `i0-b3` are the
  four branches generated in the first iteration.

Reading the shape of the graph tells you a lot quickly. A wall of red means the model cannot
compile against your build setup. Green but small circles mean it is producing correct kernels
that aren't faster. A single lineage with no branching means `branches_per_iteration` is 1.

## Per-kernel fields

Clicking a circle opens everything recorded for that candidate.

**Kernel info**: task and job name, the iteration it came from, the model that produced it, and
the hardware it was benchmarked on.

**Runtime stats**: computed from the benchmark test in your `task.py`, not from the profiler
(see [Writing tests](writing-tests.md)). Alongside the usual statistics it shows the number of
trials, which varies because fast kernels are benchmarked more times, and the speedup over the
reference measured the same way.

**Prompt**: what was actually sent to the model. This is the field to check when a run behaves
oddly: it shows whether your reference and user instructions really made it into the prompt.
The standard structure is intro, user instructions, reference implementation, best kernel so far,
parent kernel, language-specific instructions, optimization strategies.

**Input code**: the reference from your task.

**Answer**: the model's complete response, including its reasoning about previous attempts and
what it planned to change. Useful for understanding whether it correctly diagnosed the last
failure.

**Output code**: the kernel extracted from the answer. Extraction expects a fenced code block,
so a malformed answer shows up as an empty or truncated output here.

**Evaluation log**: build errors if it did not compile, correctness test output, and the
profiling summary. This log is fed back into later prompts when this kernel is sampled as a
parent, which is how profiler evidence reaches the next generation.

## Where the data lives

Results are stored in SQLite at `runs/kernels.sqlite3`, relative to where you started the run.
Change the location with `paths.kernels_db_path`.

:::{note}
When the MCP server runs in local execution mode from an editor, the database is written
relative to your home directory, because that is where the editor launches the server from.
:::

## What to do with what you see

| Symptom in the graph | Likely cause | Where to look |
| --- | --- | --- |
| All red | Build environment or missing includes | Evaluation log, build errors |
| Compiles, never correct | Reference mismatch, or tolerances too tight | Correctness output, `assert_allclose` usage |
| Correct but no speedup | Reference is already well optimized, or the benchmark shape is too small to show a difference | Runtime stats, benchmark shapes |
| No profiler data | Timed trials not inside `profiler_session` | [Writing tests](writing-tests.md) |
| One lineage, no branches | `branches_per_iteration` is 1 | [Config parameters](config-parameters.md) |
