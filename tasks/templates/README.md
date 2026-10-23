# Language templates

A complete, runnable example task per language supported by KernelFoundry, each written
against its native API with no extra abstraction layer in between. Copy whichever one
matches your target language as the starting point for a new task.

| Language | Directory | Host API |
| --- | --- | --- |
| SYCL | [`../example_custom/`](../example_custom/) | PyTorch SYCL extension (built via `compile_torch_extension`) |
| OpenCL (OCL) | [`ocl_gemm/`](ocl_gemm/) | `pyopencl`, called directly — no compiled extension |
| CM (C for Metal) | [`cm_gemm/`](cm_gemm/) | `clops`, JIT-compiled via the Intel NEO driver — no AOT build step |
| Vulkan | [`vulkan_vector_add/`](vulkan_vector_add/) | Raw Vulkan compute (`vulkan` Python bindings + `glslangValidator`) |

CUDA and Triton don't have a dedicated template here: their tasks look like ordinary
PyTorch extensions / Triton kernels and `example_custom/` already illustrates the pattern
closely enough.

See `tasks/README.md` and the
[task package guide](https://isl-org.github.io/kernelfoundry/docs/guide/task-package.html)
for what every task needs (`config.yaml`, `conftest.py`, `task.py`, the
`[EVOLVE_START]`/`[REFERENCE_START]` markers).
