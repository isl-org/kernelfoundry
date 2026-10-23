---
name: kernelfoundry-optimize
description: >-
  Use when the user wants to optimize, speed up, port, or benchmark a GPU kernel (SYCL, CUDA,
  OpenCL, or Triton) with KernelFoundry. Creates a KernelFoundry task package from the user's
  kernel, drives the build_and_test/submit_task MCP tools to validate and iteratively improve it,
  and reintegrates the result into the user's source.
---

# Optimizing a kernel with KernelFoundry: instructions for coding agents

This file explains how to drive KernelFoundry end to end: take a kernel the user wants to speed 
up, build a submittable task package, validate it, optimize it, and integrate the result back 
into their code, preserving correctness and generality throughout.

It is intentionally kernel-agnostic: no specific operation, language or hardware is assumed.

## How to work with the user

The agentic path is conversational. The user may open with one sentence describing their
kernel. Make the opening effort near zero while still producing a high-quality package.

- **Autonomous by default.** Auto-discover everything you can before asking: the language,
  the reference behavior, benchmark shapes (from existing tests and call sites), and
  build/test commands (from build files). Read the codebase directly.
- **Propose, don't interrogate.** Ask only when an input is blocking or high-consequence: for
  example, there is no way to build or validate, or the benchmark regime would change what
  gets tuned. **Batch** such questions into one short set; never drip
  them one at a time.
- **State your defaults.** For config values, benchmark shapes, build and test commands and
  assumptions, pick a sensible default and say so by reporting to the user.
- **You author `USER_INSTRUCTIONS` (optional).** The user does not hand-write it. Distill it from the
  conversation plus your own discovery. The user edits it only to correct or augment.
- **Two checkpoints require explicit confirmation.** Everything between them is autonomous:
  1. **Before first submission**: Summarize the task package and
     your assumptions, then ask for go/no-go.
  2. **Before integrating** the optimized kernel into the user's source, which edits their
     code. Confirm first.
- **Report concisely.** After each phase, cover: actions taken, evidence, what you validated,
  risks and assumptions, and the next step.

## Inputs

Gather these from the conversation and your own discovery. This is not a form to make the user
fill in.

| Input | How you get it |
|---|---|
| `REFERENCE`: the kernel or function to optimize | From the user (file + symbol, or a URL); invent one if none exists (see below) |
| `LANGUAGE`: SYCL / OCL / CUDA / triton/ CM / vulkan | Infer from the source or user input |
| `UPSTREAM`: the source of truth for behavior and benchmark evidence | Infer; ask only if external and unclear |
| `CORRECTNESS_TEST`, `BENCHMARK_TEST` | Discover existing tests; otherwise create reasonable defaults |
| `BENCHMARK_PROBLEM_SIZE(S)` | Extract from existing tests and call sites, then add coverage; state your picks |
| `LOCAL_ENVIRONMENT` (optional): shell init (oneAPI `setvars.sh`, venv activation) | Discover; ask only if blocking |
| `LOCAL_FULL_BUILD`, `LOCAL_FULL_TEST` (optional): to validate reintegration | Discover; ask only if blocking |
| `OUTPUT_PATH`: the single task directory | Default sensibly |
| `GPU_ARCH`: target architecture | Discover the target GPU; ask if unknown. Use the [supported GPU targets](https://isl-org.github.io/kernelfoundry/docs/guide/config-parameters.html#supported-gpu-targets). |

When the user has no existing kernel to start from (e.g. writing one from scratch), invent a
reasonable reference yourself — e.g. the equivalent `torch` op — rather than skipping it.
Skipping the reference entirely (`test_reference: false` in `config.yaml`) is possible but not
recommended: it leaves no correctness baseline and no speedup figure to measure against.

## Hard constraints

1. **One task directory.** Create exactly one, do all work inside it, and never make
   duplicate copies.
2. **Immutable reference.** Build the reference to match the original as faithfully as
   possible; make only the minimal changes needed to make it standalone (compatibility stubs,
   includes, macros, build glue, a pybind wrapper so pytest can call it). Once validated, do
   not modify it again. Every speedup is measured against it, so if the reference drifts, the
   numbers become meaningless.
3. **Benchmark shapes.** Extract real shapes from the user's code, then add a small set of
   representative coverage shapes. Annotate every shape with `origin: upstream|coverage` and a
   short `rationale`.
4. **Preserve flexibility.** Do not specialize away macro-driven or runtime-configurable
   behavior. The optimized kernel must stay as general as the original across data types,
   sizes and runtime configuration.
5. **Multi-variant strategy.** Consider a small portfolio of variants (2–4) with runtime
   dispatch selecting per input pattern. Report the complexity cost against the gain, and do not
   add dispatch that does not pay for itself.
6. **Config and pass gate.** Read the public [config reference](https://isl-org.github.io/kernelfoundry/docs/guide/config-parameters.html)
   before writing `config.yaml`. Use it for accepted keys, required values, defaults and search settings. Do **not**
   invent config keys: a task `config.yaml` may only set keys that exist in
   `configs/run.yaml`, and anything else fails the run with
   `ConfigKeyError: Key '<name>' is not in struct`. Do **not**
   report success unless `build_and_test` passes end to end: `success=true`, with valid
   runtime stats and a real speedup, never `N/A` or empty.

## Workflow

Two phases: build the task package (A-D), then optimize the kernel inside it (E-H).

### Phase 1: Task package

- **A. Setup.** Create the one task directory. Import the kernel plus any compatibility
  scaffolding it needs to stand alone. Freeze the reference and validate its baseline
  correctness. Write `config.yaml` with explicit `max_iters`, `branches_per_iteration` and
  `gpu_arch`.
- **B. Evidence.** Locate and summarize the relevant existing tests, benchmarks and call
  sites. Build the benchmark set with an explicit `origin` and `rationale` per shape.
- **C. Local validation (optional).** Run python task.py; pytest --ref task.py locally as a quick 
  check whether the reference is okay. Only if KernelFoundry is fully installed locally; might be 
  unavailable if the MCP is configured to submit to a server. 
- **D. First Submission.** **[CHECKPOINT]** Summarize the package and your assumptions, and get
  go/no-go. On go, call the MCP tool `build_and_test(folder_path)` to get the initial performance.

### Phase 2: Kernel optimization

- **E. Optimize.** Edit **only** the `EVOLVE` region; the reference stays frozen. Keep macro
  and runtime compatibility. Evaluate candidate strategies, including multi-variant dispatch.
  Ground every technique decision in profiler evidence rather than assumption, and consult the
  vendor optimization guide for the target architecture. Run `build_and_test(folder_path)`
  whenever you want to benchmark / profile a new version.
- **F Submit evolutionary job (optional).** You can also use the `submit_task(folder_path)` tool
  which allows to submit a job that will run many iterations of optimization autonomously. For
  this you must set the config parameters (`max_iters`, `branches_per_iteration`, etc.)
- **G. Review.** When the optimized kernel comes back, verify every hard constraint above and
  produce a pass/fail compliance report with evidence. Patch and re-validate if needed.
- **H. Reintegration.** **[CHECKPOINT]** Confirm before editing the user's source. Merge the
  optimized kernel into the original implementation, preserving semantics and generality. Run
  `LOCAL_FULL_BUILD` and `LOCAL_FULL_TEST`. Record in-situ performance against the original.

## Package structure

A KernelFoundry task package contains:

- `config.yaml`: task name, language, `gpu_arch`, iteration settings.
- `task.py`: derives `TestBase`, with correctness tests and one `@pytest.mark.performance`
  benchmark. `build()` must not require a GPU: it cross-compiles ahead-of-time for the given
  `gpu_arch`, since build and test are dispatched to different, potentially GPU-less workers.
  If the language instead compiles JIT at test time (e.g. OCL, CM), leave `build()` returning
  `[]` and set `has_build_step: false` in `config.yaml`.
- The kernel file (`.sycl` / `.cpp` / `.cl`) containing an `[EVOLVE_START]` / `[EVOLVE_END]`
  region, the only part that gets rewritten.
- Optionally a reference, marked `[REFERENCE_START]` / `[REFERENCE_END]`.
- A `[USER_INSTRUCTIONS_START]` / `[USER_INSTRUCTIONS_END]` block (optional). The
  distilled per-task guidance (constraints, what is tested, strategies to try) drawn from the
  conversation and your discovery.
- `conftest.py`, unchanged: `from kernelfoundry.conftest import *`.

Because block markers live in `.py` or `.cpp` files, they must be inside comments.

Copy the closest starting point for the target language from
[`tasks/templates/`](https://github.com/isl-org/kernelfoundry/tree/main/tasks/templates/) — e.g.
`cm_gemm` for CM, `ocl_gemm` for OCL, `vulkan_vector_add` for Vulkan — matching algorithmic
structure and test style over exact operation. If the templates are unavailable or no closer
match exists, use the basic SYCL template at
[`tasks/example_custom/`](https://github.com/isl-org/kernelfoundry/tree/main/tasks/example_custom/)
(also a reasonable default for CUDA or Triton). Then follow the guide for the details this
page summarizes:

| For | Read |
|---|---|
| The full package specification | [Task package guide](https://isl-org.github.io/kernelfoundry/docs/guide/task-package.html) |
| Correctness and benchmark tests | [Writing tests](https://isl-org.github.io/kernelfoundry/docs/guide/writing-tests.html) |
| Every config key, and which are required | [Public config reference](https://isl-org.github.io/kernelfoundry/docs/guide/config-parameters.html) |
| Optimization techniques worth trying | [Optimization strategies](https://isl-org.github.io/kernelfoundry/docs/guide/optimization-strategies.html) |
| Reading a finished run | [Understanding results](https://isl-org.github.io/kernelfoundry/docs/guide/understanding-results.html) |

Limit your changes to the kernel, the reference and `task.py`; put any additional code in
auxiliary files inside the package.

## Validating locally before submitting

If the machine has a GPU (check with `xpu-smi` or `nvidia-smi`) and KernelFoundry installed, a
task is valid when the **reference** passes its tests:

```bash
pytest --ref task.py
```

If there is no local GPU or no local install, it is fine to let validation happen at
submission time; `build_and_test` reports an error if the package is invalid.

## Definition of done

- One task directory, no duplicates.
- A frozen, valid reference.
- Benchmarks covering real and coverage shapes, annotated.
- Macro and runtime compatibility preserved.
- Dispatch strategy evaluated and justified.
- Submission checkpoint honored.
- Post-run compliance report produced.
- Reintegration tested, with the speedup recorded.
