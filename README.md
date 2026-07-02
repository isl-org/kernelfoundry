# KernelFoundry
<img src="https://github.com/isl-org/kernelfoundry/raw/main/assets/kernelfoundry.png" alt="KernelFoundry Logo" width="200" align="right">

KernelFoundry is an evolutionary framework for hardware-aware GPU kernel optimization. It combines an automated kernel generation algorithm with a comprehensive test harness and evaluation pipeline, enabling rapid exploration of the GPU kernel design space.

The package provides:

- **Kernel generation algorithm**: An evolutionary approach using MAP-Elites quality-diversity search, meta-prompt evolution, and template-based parameter optimization to efficiently generate optimized GPU kernels.
- **Evaluation pipeline**: A distributed framework for benchmarking kernels, collecting profiler feedback, and storing results in a database.
- **Web UI**: Interactive visualization of generated kernels, performance metrics, and optimization progress.
- **Task-side test harness**: Base interfaces and utilities for implementing custom tasks and evaluating kernels locally.


## Installation

Prerequesites:
- The package requires Python 3.10+ (we commonly use Python 3.12). Install into a virtual environment if desired.
- oneAPI: For compiling kernels on Intel GPUs, you need Intel OneAPI. See section below for installation instructions.

#### Install KernelFoundry Python package

A) For test harness only (evaluating existing kernels):
```
pip install kernelfoundry
# If tasks are based on pytorch, install torch for your hardware:
# - Intel: pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/xpu
```

B) For kernel generation algorithm, UI, and full pipeline:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install pip
uv pip install pip

# Install torch (Intel XPU wheels)
python -m pip install torch==2.9 torchvision==0.24 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu
# - NVIDIA: pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129

# Install all dependencies
uv pip install -e .[all]
```

#### Install Intel oneAPI toolkit

For running kernel generation on Intel GPUs, you need OneAPI. Installation on Linux:
```bash
# Download and add Intel GPG key
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# Add Intel OneAPI repository
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list

# Update and install Intel OneAPI toolkit and VTune
sudo apt update && sudo apt install -y \
    intel-oneapi-base-toolkit-2025.2 \
    intel-oneapi-vtune=2026.1.0-13

# Source Intel environment
source /opt/intel/oneapi/setvars.sh
```

#### Install profiler

For profiling SYCL kernels on Intel GPUs, we use unitrace. Install with:
```bash
git clone https://github.com/intel/pti-gpu.git
pushd pti-gpu/tools/unitrace
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF ..
make
popd
# allow non root profiling
if [ -f /proc/sys/dev/i915/perf_stream_paranoid ]; then
    sudo sh -c 'echo 0 > /proc/sys/dev/i915/perf_stream_paranoid'
fi
if [ -f /proc/sys/dev/xe/observation_paranoid ]; then
    sudo sh -c 'echo 0 > /proc/sys/dev/xe/observation_paranoid'
fi
```

Alternatively, VTune is also supported as a profiler, which already comes with oneAPI. Make sure to enable non-root proofiling with:
```bash
if [ -f /proc/sys/kernel/kptr_restrict ]; then
    sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
fi
```
For OpenCL kernel optimization, VTune is the default profiler. For SYCL, set `profiler_kernel: vtune` and `profiler_reference: vtune`.

## Running KernelFoundry

### 1. Running the kernel generation algorithm

Generate optimized kernels by running the evolutionary algorithm on a task. The algorithm supports KernelBench tasks, robust_kbench, and custom tasks.

Optimize KernelBench task:
```bash
python -m kernelfoundry.algorithm run task=19_ReLU task_origin=KernelBench job_name=my_experiment gpu_arch=lnl language=SYCL
```

Optimize custom task (task=path/to/task/folder):
```bash
python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom job_name=my_custom_experiment gpu_arch=lnl language=SYCL
```

Common parameters:
- `task`: Path to task folder or KernelBench task ID (e.g., `19_ReLU` for KernelBench task 19)
- `task_origin`: Either `KernelBench`, `robust_kbench`, or `custom`
- `gpu_arch`: GPU architecture (e.g., `lnl` for Intel Arc, `Ampere` for NVIDIA)
- `language`: Kernel language (`SYCL` or `CUDA`; defaults based on GPU type)
- `job_name`: your name for the optimization job (for tracking results; will appear in UI)

See [main config](configs/run.yaml) for all available parameters.

Generated kernels are stored in a SQLite database (configurable via `paths.kernels_db_path` in the config).

### 2. Using the KernelFoundry test harness

Implement and evaluate your own kernel optimization tasks using the test harness:

1. Create a task test class by deriving from `TestBase` (`from kernelfoundry import TestBase`).
2. Implement build logic (optional) and correctness/performance tests (see [task format](tasks/README.md)).
3. Compile candidate kernel code with `compile_torch_extension(...)`.
4. Run pytest to validate correctness and collect benchmark timings.

The test harness provides:

- A base task interface (`from kernelfoundry import TestBase`) for implementing task-specific build and pytest test logic.
- Build helpers for compiling candidate kernels into PyTorch extensions (via Torch or `icpx`).
- Pytest fixtures for correctness/performance runs and collecting runtime data.
- Validation helpers (`assert_allclose`, cosine similarity, and related utilities).
- Runtime and machine-info helpers for benchmarking and metadata capture.
- Support for SYCL, OpenCL, and CUDA kernels.

You can run the KernelFoundry algorithm with option `validate` to *test* a kernel without generating a new kernel. First, [create a task](tasks/README.md) and place your kernel in the EVOLVE-block (see [example](tasks/example_custom/matrix_mul_kernel.sycl)). Then run:
```bash
python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom job_name=my_custom_experiment gpu_arch=lnl language=SYCL validate=true max_iters=0
```

## Visualization and monitoring

Monitor kernel generation progress and view results through the web-based UI:

```bash
python start_gui.py
```

The UI will be available at `http://localhost:8885` and provides:

- Job logs and execution monitoring
- Kernel performance metrics and comparisons
- Optimization progress visualization
- Roofline analysis and performance profiling
- Detailed kernel implementations and metadata


### MCP server

The KernelFoundry test harness can also be used as a tool by by installing the KF MCP server. See [here](kernelfoundry/mcp_server/README.md) for installation instructions and details.


### Documentation

API documentation is available and can be built using Sphinx. To build the documentation:

```
pip install .[docs]
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`. See the [docs/README.md](docs/README.md) for more information.

## References

If you use KernelFoundry in your research, please cite:

```bibtex
@inproceedings{wiedemann2026kernelfoundry,
  title={KernelFoundry: Hardware-aware evolutionary GPU kernel optimization},
  author={Wiedemann, Nina and Leboutet, Quentin and Paulitsch, Michael and Wofk, Diana and Ummenhofer, Benjamin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML 2026)},
  year={2026}
}
```