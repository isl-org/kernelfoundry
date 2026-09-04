# Quickstart

This walks through the shipped example task end to end. It makes no LLM API calls, so it needs
no API key. The point is to prove your GPU, compiler and test setup work before you spend
anything on generation.

```{note}
The shipped example is a SYCL kernel targeting an Intel GPU, and execution will fail on an NVIDIA GPU. 
Try to generate a kernel for KernelBench instead as a quickstart on an NVIDIA GPU (see README). 
```

## What you need

- Python 3.10+ (3.12 recommended)
- A GPU and its toolchain: Intel oneAPI for SYCL, or CUDA for NVIDIA. See the
  [README](https://github.com/isl-org/kernelfoundry#installation).
- A clone of the repository, for the example task and configs:

```bash
git clone https://github.com/isl-org/kernelfoundry.git
cd kernelfoundry
pip install -e .
```

On Intel GPUs, source the oneAPI environment in every new shell:

```bash
source /opt/intel/oneapi/setvars.sh
```

```{note}
**Windows users:** do not source the full oneAPI environment. Instead, add only the oneAPI,
MSVC, and Windows SDK toolchain directories to `PATH`, set `INCLUDE`, `LIB`, `LIBPATH`, and
`CMPLR_ROOT`, and pass `gpu_arch` explicitly. See the installation notes in the README for the
known torch/MSVC and separate-venv caveats.
```

## 1. Look at the task

The example lives in
[`tasks/example_custom/`](https://github.com/isl-org/kernelfoundry/tree/main/tasks/example_custom),
a SYCL matrix multiplication:

```
tasks/example_custom/
├── config.yaml               # task_name: matmul, gpu_arch: lnl, language: SYCL
├── task.py                   # reference, fixtures, tests
├── matrix_mul_kernel.sycl    # the kernel, with the evolve block
└── conftest.py               # wires in the KernelFoundry fixtures (do not edit)
```

`task.py` defines a `TestMatrixMultiplication` class deriving from `TestBase` with:

- a **reference**, marked `[REFERENCE_START]` / `[REFERENCE_END]`
- two **correctness tests**, against the reference and against an identity matrix
- one **benchmark**, marked `@pytest.mark.performance` and parametrized over sizes 512, 1024
  and 2048

`matrix_mul_kernel.sycl` holds the `[EVOLVE_START]` / `[EVOLVE_END]` block, the only region
KernelFoundry would rewrite.

## 2. Validate the reference

Before anything else, check the task definition itself by running the tests against the
*reference* rather than the kernel:

```bash
cd tasks/example_custom
pytest --ref -s task.py
```

The correctness tests should pass and the benchmark should report runtimes for each size. This
is the single most useful check you can run: if the reference does not pass its own tests, every
speedup measured later is meaningless, because speedup is measured against it.

If this fails, the problem is your environment or the task, not the kernel.

## 3. Build the kernel

The evolve block already contains a working, unoptimized kernel, so it will compile as-is:

```bash
python task.py
```

This runs the `build` method of the test class, compiling the SYCL source into a shared object.
Build errors here are compiler or toolchain problems. Check that `setvars.sh` was sourced and
that `gpu_arch` in `config.yaml` matches your hardware.

## 4. Test and benchmark the kernel

```bash
pytest -s task.py
```

Now the tests run against the compiled kernel. You get correctness results plus a runtime for
each benchmark size, measured the same way the reference was, so the ratio between this run
and step 2 is the speedup the baseline kernel already achieves.

## 5. Run it through the pipeline without generating

To exercise the full evaluation path, the same one an optimization job uses, without calling
an LLM, evaluate the kernel currently in the evolve block:

```bash
cd ../..
python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom \
    job_name=my_first_run gpu_arch=lnl language=SYCL validate=true max_iters=0
```

`validate=true max_iters=0` means "evaluate what is there, generate nothing". Results land in
`runs/kernels.sqlite3`, viewable in the UI:

```bash
kernelfoundry-gui       # http://localhost:8885 (python start_gui.py from a clone)
```

See [Understanding results](understanding-results.md) for how to read what you find there.

## What you have proven

Your toolchain compiles kernels, your GPU runs them, the tests measure them, and results reach
the database and UI. Everything above this point is shared with a real optimization run.

## Next steps

- **Optimize this kernel for real.** Same command without the validate flags, plus an API key.
  This is the first point at which you need one:

  ```bash
  export OPENAI_API_KEY=...
  python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom \
      job_name=matmul_opt gpu_arch=lnl language=SYCL
  ```

- **Package your own kernel**: [Anatomy of a task package](task-package.md)
- **Write tests for it**: [Writing tests](writing-tests.md)
- **Tune the run**: [Config parameters](config-parameters.md) and
  [Optimization strategies](optimization-strategies.md)
- **Let a coding agent do it**: [Optimize a kernel with a coding agent](agentic-workflow.md)
  exposes this same loop as a single tool your editor can call, and hands it the packaging
  work you just did by hand.
