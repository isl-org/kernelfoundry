# Optimizing a kernel with KernelFoundry — instructions for coding agents

This file tells a coding agent (GitHub Copilot, Claude Code, Cursor, …) how to drive
KernelFoundry end to end: take a kernel the user wants to speed up, build a submittable
kernel package, validate it, optimize it, and integrate the result back into their code —
preserving correctness and generality throughout.

It is intentionally kernel-agnostic: no specific operation, language or hardware is assumed.

> Scope note: this describes using KernelFoundry *as a tool* on a user's kernel. It is not a
> guide to contributing to the KernelFoundry repository itself.

## How to work with the user

The agentic path is conversational. The user may open with one sentence describing their
kernel. Make the opening effort near zero while still producing a high-quality package.

- **Autonomous by default.** Auto-discover everything you can before asking: the language,
  the reference behavior, benchmark shapes (from existing tests and call sites), and
  build/test commands (from build files). Read the codebase directly.
- **Propose, don't interrogate.** Ask only when an input is blocking or high-consequence —
  for example, there is genuinely no way to build or validate, or the benchmark regime would
  materially change what gets tuned. **Batch** such questions into one short set; never drip
  them one at a time.
- **State your defaults.** For everything you infer or choose — config values, benchmark
  shapes, build and test commands, assumptions — pick a sensible default and say so, writing
  it into the package's `USER_INSTRUCTIONS` block so it is visible and easy to correct.
- **You author `USER_INSTRUCTIONS`.** The user does not hand-write it. Distill it from the
  conversation plus your own discovery. The user edits it only to correct or augment.
- **Two checkpoints require explicit confirmation.** Everything between them is autonomous:
  1. **Before submitting a job** — it consumes compute and time. Summarize the package and
     your assumptions, then ask for go/no-go.
  2. **Before integrating** the optimized kernel into the user's source — it edits their
     code. Confirm first.
- **Report concisely.** After each phase: *Actions · Evidence · Validation ·
  Risks/assumptions · Next step.*

## Inputs

Gather these from the conversation and your own discovery — this is not a form to make the
user fill in.

| Input | How you get it |
|---|---|
| `REFERENCE` — the kernel or function to optimize | Required; from the user (file + symbol, or a URL) |
| `LANGUAGE` — SYCL / OpenCL / CUDA | Infer from the source |
| `UPSTREAM` — the source of truth for behavior and benchmark evidence | Infer; ask only if external and unclear |
| `CORRECTNESS_TEST`, `BENCHMARK_TEST` | Discover existing tests; otherwise create reasonable defaults |
| `BENCHMARK_PROBLEM_SIZE(S)` | Extract from existing tests and call sites, then add coverage; state your picks |
| `LOCAL_ENVIRONMENT` — shell init (oneAPI `setvars.sh`, venv activation) | Discover; ask only if blocking |
| `LOCAL_FULL_BUILD`, `LOCAL_FULL_TEST` — to validate reintegration | Discover; ask only if blocking |
| `OUTPUT_PATH` — the single task directory | Default sensibly |
| `GPU_ARCH` — target architecture | Ask if unknown; see the `gpu_arch` table in the README |

## Hard constraints

1. **One task directory.** Create exactly one, do all work inside it, and never make
   duplicate copies.
2. **Immutable reference.** Build the reference to match the original as faithfully as
   possible; make only the minimal changes needed to make it standalone (compatibility stubs,
   includes, macros, build glue, a pybind wrapper so pytest can call it). Once validated, do
   not modify it again — every speedup is measured against it, so if the reference drifts, the
   numbers become meaningless.
3. **Benchmark shapes.** Extract real shapes from the user's code, then add a small set of
   representative coverage shapes. Annotate every shape with `origin: upstream|coverage` and a
   short `rationale`.
4. **Preserve flexibility.** Do not specialize away macro-driven or runtime-configurable
   behavior. The optimized kernel must stay as general as the original across data types,
   sizes and runtime configuration.
5. **Multi-variant strategy.** Consider a small portfolio of variants (2–4) with runtime
   dispatch selecting per input pattern. Report the complexity cost against the gain — do not
   add dispatch that does not pay for itself.
6. **Config and pass gate.** Set `max_iters` and `branches_per_iteration` appropriately. Do
   **not** invent config keys: a task `config.yaml` may only set keys that exist in
   `configs/run.yaml`, and anything else fails the run with
   `ConfigKeyError: Key '<name>' is not in struct`. In particular there is no `evolve_mode`
   key — evolutionary search turns on by itself when `branches_per_iteration > 1`. Do **not**
   report success unless `build_and_test` passes end to end: `success=true`, with valid
   runtime stats and a real speedup — never `N/A` or empty.

## Workflow

- **A · Setup.** Create the one task directory. Import the kernel plus any compatibility
  scaffolding it needs to stand alone. Freeze the reference and validate its baseline
  correctness. Write `config.yaml` with explicit `max_iters`, `branches_per_iteration` and
  `gpu_arch`.
- **B · Evidence.** Locate and summarize the relevant existing tests, benchmarks and call
  sites. Build the benchmark set with an explicit `origin` and `rationale` per shape.
- **C · Optimize.** Edit **only** the `EVOLVE` region; the reference stays frozen. Keep macro
  and runtime compatibility. Evaluate candidate strategies, including multi-variant dispatch.
  Ground every technique decision in profiler evidence rather than assumption, and consult the
  vendor optimization guide for the target architecture.
- **D · Local validation.** Run correctness against the reference. Run benchmarks and report
  per-shape and aggregate speedup. Require a full `build_and_test` pass.
- **E · Submission.** **[CHECKPOINT]** Summarize the package and your assumptions, and get
  go/no-go. On go, call the MCP tool `build_and_test(folder_path)`.
- **F · Review.** When the optimized kernel comes back, verify every hard constraint above and
  produce a pass/fail compliance report with evidence. Patch and re-validate if needed.
- **G · Reintegration.** **[CHECKPOINT]** Confirm before editing the user's source. Merge the
  optimized kernel into the original implementation, preserving semantics and generality. Run
  `LOCAL_FULL_BUILD` and `LOCAL_FULL_TEST`. Record in-situ performance against the original.

## Package structure

A KernelFoundry task package contains:

- `config.yaml` — task name, language, `gpu_arch`, iteration settings.
- `task.py` — derives `TestBase`, with correctness tests and one `@pytest.mark.performance`
  benchmark.
- The kernel file (`.sycl` / `.cpp` / `.cl`) containing an `[EVOLVE_START]` / `[EVOLVE_END]`
  region — the only part that gets rewritten.
- Optionally a reference, marked `[REFERENCE_START]` / `[REFERENCE_END]`.
- A `[USER_INSTRUCTIONS_START]` / `[USER_INSTRUCTIONS_END]` block — **you** author this: the
  distilled per-task guidance (constraints, what is tested, strategies to try) drawn from the
  conversation and your discovery.
- `conftest.py`, unchanged: `from kernelfoundry.conftest import *`.

Because block markers live in `.py` or `.cpp` files, they must be inside comments.

Copy the closest starting point from [`tasks/example_custom/`](tasks/example_custom/) and
follow the full specification in [`tasks/README.md`](tasks/README.md). Limit your changes to
the kernel, the reference and `task.py`; put any additional code in auxiliary files inside the
package.

## Validating locally before submitting

If the machine has a GPU (check with `xpu-smi` or `nvidia-smi`) and KernelFoundry installed, a
task is valid when the **reference** passes its tests:

```bash
pytest --ref task.py
```

If there is no local GPU or no local install, it is fine to let validation happen at
submission time — `build_and_test` reports an error if the package is invalid.

## Definition of done

One task directory, no duplicates · frozen, valid reference · benchmarks covering real and
coverage shapes, annotated · macro and runtime compatibility preserved · dispatch strategy
evaluated and justified · submission checkpoint honored · post-run compliance report produced ·
reintegration tested with recorded speedup.
