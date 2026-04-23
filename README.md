# KernelFoundry
<img src="assets/kernelfoundry.png" alt="KernelFoundry Logo" width="200" align="right">

KernelFoundry is a Python package for defining and evaluating GPU kernel tasks.

This repository provides the **task-side test harness** used in kernel generation workflows:

- A base task interface (``kernelfoundry.custom_test.CustomTest``) for implementing task-specific build and pytest test logic.
- Build helpers for compiling candidate kernels into PyTorch extensions (via Torch or `icpx`).
- Pytest fixtures for correctness/performance runs and collecting runtime data.
- Validation helpers (`assert_allclose`, cosine similarity, and related utilities).
- Runtime and machine-info helpers for benchmarking and metadata capture.

In other words, this package is what you use to **author tasks and evaluate generated kernels**. The external orchestration layers (for example, UI/services that generate candidates) are separate from this repo.

## Typical workflow

1. Create a task test class by deriving from `kernelfoundry.custom_test.CustomTest`.
2. Implement build logic (optional) and correctness/performance tests.
3. Compile candidate kernel code with `compile_torch_extension(...)`.
4. Run pytest to validate correctness and collect benchmark timings.

KernelFoundry supports SYCL, OpenCL and CUDA kernels.

### Installation

Install the package and basic dependencies with

```
pip install .
```

The package requires Python 3.10+ (we commonly use Python 3.12). Install into a virtual environment if desired.

The `kernelfoundry` package has minimal core dependencies. Depending on your task, you may also need framework/toolchain dependencies (for example PyTorch and GPU toolchains) to compile and test kernels locally.

To run PyTorch-based tasks, install torch appropriate for your hardware:

```
# on Intel hardware:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/xpu
# on NVIDIA hardware:
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
```

### Documentation

API documentation is available and can be built using Sphinx. To build the documentation:

```
pip install .[docs]
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`. See the [docs/README.md](docs/README.md) for more information.