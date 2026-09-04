# Anatomy of a task package

A **task package** is the folder KernelFoundry evaluates. It defines one optimization task: the
kernel to be rewritten, something to compare it against, and the tests that decide whether a
candidate is correct and fast.

On the recommended path you do not write one. A coding agent builds it from the kernel you point it
at, and this page is what it builds; see
[Optimize a kernel with a coding agent](agentic-workflow.md). Read on if you are authoring a package
by hand, or checking what an agent produced.

If you are writing one yourself, start from
[`tasks/example_custom/`](https://github.com/isl-org/kernelfoundry/tree/main/tasks/example_custom)
in the repository and adapt it. That is faster and less error-prone than starting from scratch.

```{image} ../assets/task-package-light.svg
:alt: A task package folder contains config.yaml, task.py, a kernel file and conftest.py. The kernel file is shown expanded, with an EVOLVE_START and EVOLVE_END pair bracketing the only region KernelFoundry rewrites. task.py carries the optional REFERENCE block, which a candidate is compared against, and the optional USER_INSTRUCTIONS block, which steers the model. All three markers live inside comments.
:class: only-light
:width: 700px
```

```{image} ../assets/task-package-dark.svg
:alt: A task package folder contains config.yaml, task.py, a kernel file and conftest.py. The kernel file is shown expanded, with an EVOLVE_START and EVOLVE_END pair bracketing the only region KernelFoundry rewrites. task.py carries the optional REFERENCE block, which a candidate is compared against, and the optional USER_INSTRUCTIONS block, which steers the model. All three markers live inside comments.
:class: only-dark
:width: 700px
```

## File structure

A package broadly consists of:

| File | Purpose |
| --- | --- |
| `config.yaml` | Task name, language, target GPU architecture, iteration settings. See [Config parameters](config-parameters.md). |
| `task.py` | Defines the task: a reference implementation (often a PyTorch operation) and the tests a kernel must pass. See [Writing tests](writing-tests.md). |
| `kernel.cpp` / `kernel.sycl` / `kernel.cl` | The kernel file the model rewrites. It can be almost empty at the start, as long as it carries the evolve markers below. |
| `reference.cpp` *(optional)* | A baseline implementation to compare against, when the reference is not expressed in `task.py`. |
| `conftest.py` | Must be included **unchanged**. Its only job is to wire your task into the KernelFoundry pytest fixtures. |

## Block structure

Three markers tell KernelFoundry which parts of your code mean what. Because they sit inside
`.py` or `.cpp` files, they are always written **inside comments**.

`[EVOLVE_START]` / `[EVOLVE_END]`
: The only region KernelFoundry will rewrite. Everything outside it is left untouched.
  This block is **required**. It may be empty; leave it empty to generate a kernel from
  scratch with no given function header, includes or bindings.

`[REFERENCE_START]` / `[REFERENCE_END]`
: The implementation the kernel is compared against, for example
  `torch.matmul(a, b)`, where the task is to write a custom kernel that beats it. Usually
  also used to check correctness, by comparing output tensors. **Optional**: you can instead
  guide the model purely through user instructions and test the kernel some other way.

`[USER_INSTRUCTIONS_START]` / `[USER_INSTRUCTIONS_END]`
: Free-form guidance passed to the model: optimization strategies worth trying, what will be
  tested, constraints to respect. **Optional**, but the cheapest way to steer a run. See
  [Optimization strategies](optimization-strategies.md).

Notes:

- By default all three blocks are included in the prompt, so the model sees the reference, the
  code to evolve, and your instructions together.
- The blocks can live anywhere; KernelFoundry searches every file you provide. In practice it
  is clearest to keep a dedicated kernel file holding the evolve block.

## Steps for creating a task

1. Define the **evolve block** in your kernel file.
2. Define a **reference**. Usually required, since it is what correctness and speedup are
   measured against.
3. *Optional:* add **user instructions**.
4. Edit **`config.yaml`**: at minimum `task_name`, `job_name`, `language` and `gpu_arch`.
   See [Config parameters](config-parameters.md).
5. Write **tests**: at least one correctness test and exactly one benchmark. See
   [Writing tests](writing-tests.md).

Then validate locally before spending compute on a full run:

```bash
pytest --ref -s task.py
```

That runs your tests against the *reference* rather than the kernel. If the reference does not
pass its own tests, the task definition is wrong and no amount of kernel generation will fix
it. See [Quickstart](quickstart.md) for a complete worked example.
