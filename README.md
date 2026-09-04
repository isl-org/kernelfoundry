# KernelFoundry
<img src="./assets/kernelfoundry.png" alt="KernelFoundry Logo" width="200" align="right">

[![Website](https://img.shields.io/badge/Website-KernelFoundry-0A7E8C?style=flat-square)](https://isl-org.github.io/kernelfoundry/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.12440-B31B1B?style=flat-square)](https://arxiv.org/abs/2603.12440)
[![Docs](https://img.shields.io/badge/Docs-Documentation-1F6FEB?style=flat-square)](https://isl-org.github.io/kernelfoundry/docs/index.html)
[![Lint](https://github.com/isl-org/kernelfoundry/actions/workflows/black.yml/badge.svg)](https://github.com/isl-org/kernelfoundry/actions/workflows/black.yml)
[![PyPI](https://img.shields.io/pypi/v/kernelfoundry)](https://pypi.org/project/kernelfoundry/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/kernelfoundry/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/isl-org/kernelfoundry/blob/main/LICENSE)

KernelFoundry is a toolkit for hardware-aware GPU kernel optimization, targeting SYCL, OpenCL and CUDA with a focus on Intel GPUs.

The simplest way to use KernelFoundry is its MCP server and custom agent. From your usual coding IDE, install the KF agent and tools and point the agent at a slow GPU kernel. The agent isolates the kernel, builds a reference implementation to beat, finds your existing tests or writes new ones, and derives benchmark sizes from how the code is actually called — asking you only about what it can't work out on its own.

From there, KernelFoundry generates, compiles, benchmarks and profiles candidate kernels on real hardware, keeping the fastest ones that still pass those tests. On KernelBench it reaches 2.3x geometric-mean speedup for SYCL ([paper](https://arxiv.org/abs/2603.12440)).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/assets/architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/architecture-light.svg">
    <img src="./docs/assets/architecture-light.svg" alt="KernelFoundry optimization workflow diagram" width="860">
  </picture>
  <br>
  <sub>
    A coding agent turns the input into a task package of reference, kernel-wrapper and tests, which you can also author yourself. The task package enters an optimization loop, driven either by (A) the evaluation tool in a modify-and-test cycle or (B) an evolutionary algorithm searching for the best kernel autonomously. The loop generates, compiles, benchmarks on a real GPU and profiles candidates, feeding test results, runtime stats and profiler feedback back into generation, and emits an optimized kernel to integrate back into your code.
  </sub>
</p>


## Core concepts

- **Task**: one kernel you want to speed up, defined by a reference implementation plus tests.
- **Job**: one run of the optimization on a task. A task can have many jobs.
- **Task package**: the folder KernelFoundry evaluates: kernel file, reference, tests, and
  `config.yaml`. The agent builds it for you, or you write it yourself. See [Anatomy of a task package](https://isl-org.github.io/kernelfoundry/docs/guide/task-package.html) for the full specification.
- **EVOLVE block**: the part of the kernel code that is modified in the optimization loop, marked
  `[EVOLVE_START]` / `[EVOLVE_END]`. Everything outside it (e.g. tests) is left untouched at that point.

## Quick start with Dev Container

The fastest way to set up KernelFoundry is by cloning the [example repository](https://github.com/isl-org/kernelfoundry.examples), which comes with a dev container configuration with dependencies preinstalled for Intel GPU development. Click on the badge to clone the example; VS Code then offers to reopen it in the container.

<p align="center">
  <a href="https://vscode.dev/redirect?url=vscode%3A%2F%2Fvscode.git%2Fclone%3Furl%3Dhttps%3A%2F%2Fgithub.com%2Fisl-org%2Fkernelfoundry.examples">
    <img src="https://img.shields.io/static/v1?label=Dev%20Containers&message=Clone%20the%20example&color=blue&logo=visualstudiocode" alt="Clone the example" />
  </a>
</p>

Requires a Linux host system with Intel GPU drivers, Docker, and VS Code with the Dev Containers extension.


## Getting started

Running KernelFoundry requires a GPU and a working toolchain for your GPU vendor, plus an
optional profiler. Expand this section for the full toolchain setup.

<details>
<summary><h3 style="display:inline">GPU toolchain, compiler and profiler setup (click to expand) </h3></summary>

### Toolchain requirements

Both paths need Python 3.10+ (3.12 recommended), a GPU, and a working toolchain for your GPU
vendor. A profiler is optional and improves the optimizer's feedback. Kernels are compiled at
runtime, so the toolchain has to be a working one, not merely installed:

- **Linux**: `icpx` (Intel) and `nvcc` (NVIDIA) both need a host C++ compiler they can find and
  drive — a system `g++`.
- **Windows (NVIDIA/CUDA)**: `nvcc` needs **MSVC** (Visual Studio 2022 or the Build Tools,
  "Desktop development with C++") and invokes `cl.exe`, so run KernelFoundry from a **Developer
  Command Prompt** or a shell where `vcvars64.bat` has been sourced — otherwise `cl.exe` isn't on
  `PATH`.
  - Match CUDA and MSVC versions: each CUDA release only supports host compilers up to a stated
    version, and a newer MSVC than that is rejected outright — pair a recent CUDA with a recent
    MSVC rather than mixing a new toolset into an old CUDA.
  - Match CUDA and torch: install the CUDA toolkit matching the torch wheel's major version — the
    `cu129` wheel wants CUDA 12.9, `cu130` wants CUDA 13.0.
  - Verified end to end: **CUDA 13.0 with MSVC 14.44 and `torch 2.13.0+cu130`**.

### Install the Intel oneAPI toolkit

Required to compile SYCL and OpenCL kernels on Intel GPUs, on either path above, including local
execution behind the MCP server.

```bash
# Download and add the Intel GPG key
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

# Add the Intel oneAPI repository
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list

# Install the oneAPI toolkit and VTune
sudo apt update && sudo apt install -y \
    intel-oneapi-base-toolkit-2025.2 \
    intel-oneapi-vtune=2026.1.0-13

# Source the Intel environment (needed in every new shell)
source /opt/intel/oneapi/setvars.sh
```

On Windows, do not source the full oneAPI environment. Instead, put only the compiler/ocloc/MSVC/Windows SDK toolchain directories on `PATH` and set `INCLUDE`, `LIB`, `LIBPATH`, and `CMPLR_ROOT`. Also pass `gpu_arch` explicitly, because Intel GPU detection on Windows can pick up an NVIDIA card and forward the wrong architecture to the compiler. In addition, torch/MSVC incompatibilities can trigger `C2872: 'std': ambiguous symbol`, and CUDA and Intel torch wheels are mutually exclusive, so separate virtual environments are required for each accelerator family.

On NVIDIA GPUs, install the CUDA toolkit instead; `ncu` ships with it and is the default profiler there.

### Install a profiler

Optional but highly recommended. Profiler feedback gives the optimizer hardware-level signals instead of timings alone.
For SYCL on Intel GPUs the default is [unitrace](https://github.com/intel/pti-gpu):

```bash
git clone https://github.com/intel/pti-gpu.git
pushd pti-gpu/tools/unitrace
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF ..
make
popd

# Allow non-root profiling
if [ -f /proc/sys/dev/i915/perf_stream_paranoid ]; then
    sudo sh -c 'echo 0 > /proc/sys/dev/i915/perf_stream_paranoid'
fi
if [ -f /proc/sys/dev/xe/observation_paranoid ]; then
    sudo sh -c 'echo 0 > /proc/sys/dev/xe/observation_paranoid'
fi
```

For OpenCL we recommend to use VTune, which already ships with oneAPI. Enable non-root profiling with:

```bash
if [ -f /proc/sys/kernel/kptr_restrict ]; then
    sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'
fi
```

VTune is the default profiler for OpenCL. To use it for SYCL, set `profiler_kernel: vtune`
and `profiler_reference: vtune` in the config file.

**Profiling on NVIDIA GPUs:**

On NVIDIA GPUs the default profiler is `ncu`, which ships with the CUDA toolkit. Every profiler needs
permission to read GPU performance counters, and how you grant it differs by platform:

- **Linux**: run as root, or follow NVIDIA's [ERR_NVGPUCTRPERM guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
  to enable non-root profiling.
- **Windows**: run from an elevated prompt, or allow counter access for all users in the NVIDIA
  Control Panel under *Desktop > Developer Settings*.

Profiling is on by default. To **run without a profiler**, set `eval_config.profile_custom_model: false` (for the reference, profiling is disabled by default - `profile_original_model: false`).

</details>
  
### Install KernelFoundry with MCP server

When the GPU and toolkit are installed, create a virtual environment and install Python 3.12 and KernelFoundry.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
source $HOME/.local/bin/env

# Create and activate a virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install pip

# install kernelfoundry mcp
pip install 'kernelfoundry[mcp,algo]'
```
Then, use our helper script to create the MCP server config:

```
python -m kernelfoundry.mcp_server create_config   # prompts for a few answers, writes an MCP client config
```

This script will output the MCP config file, which you can copy-paste when adding the MCP server to your coding agent (GitHub Copilot, Claude Code, Cursor). Note: `create_config` is interactive: it asks a few questions and writes the client config from the answers. It has no non-interactive mode, so it cannot be scripted or run by an agent as-is.

### Using KernelFoundry via the MCP server

Point the agent to [`agent-workflow.md`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/mcp_server/agent-workflow.md) to instruct it how to use KernelFoundry.

A typical prompt is the following (fill in dependent on your use case):
```
Optimize <function> in <file path or upstream URL> for GPU with KernelFoundry (see <path/to/agent-workflow.md> for instruction how to use KernelFoundry).
# Optional (the agent will infer or ask if omitted):
# Language: <SYCL | OpenCL | CUDA | triton>
# Benchmark shapes: <e.g. from upstream tests, or specific sizes>
# Build: <command>   Test: <command> 
# Constraints: <e.g. preserve macro-driven runtime flexibility>
```

The agent creates the task package (i.e. code for testing the kernel in isolation), validates it, runs the optimization loop and reports the speedup. Full walkthrough:
[Optimize a kernel with a coding agent](https://isl-org.github.io/kernelfoundry/docs/guide/agentic-workflow.html).

<details>
<summary><h4 style="display:inline">Details on MCP server setup and how it works (click to expand) </h4></summary>

The KernelFoundry [MCP](https://modelcontextprotocol.io) server exposes two tools:
* **`build_and_test(folder_path)`**, which builds and benchmarks the task package in that
folder and reports whether it passed, the evaluation log, the measured runtimes and the speedup
over your reference.
* **`submit_task(folder_path)`**, which starts the full evolutionary algorithm that iteratively optimizes the kernels. NOTE: this option requires an LLM-API to generate the kernels, see [Configuring the LLM API](#configuring-the-llm-api).

The agent will
1) follow the instructions to create the task package
2) validate the task package by submitting it to the `build_and_test` tool (baseline result)
3) edit the kernel
4) submit the new version to the `build_and_test` tool and reads the failures and timings,
rewrites the EVOLVE block, and calls the tool again. It is the cycle in the diagram above, with
your agent in the *Generate* box.
5) Optional: submit to the `submit_task` tool to make use of the evolutionary algorithm (this tool can take up to a few hours dependent on the number of iterations)

Further references:
- **[Optimize a kernel with a coding agent](https://isl-org.github.io/kernelfoundry/docs/guide/agentic-workflow.html)**,
  the full walkthrough with a worked example and how to integrate the result.
- **[MCP server reference](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/mcp_server/README.md)**
  for client setup.
- **[`agent-workflow.md`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/mcp_server/agent-workflow.md)**, the workflow
  contract the agent need to follow for creating the kernelfoundry task package and optimizing the kernel.
</details>

## Running the algorithm via the CLI 

### Install requirements

Create uv environment:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate a virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install pip
```

**Pip install KernelFoundry**: (Note: pip installing gives you the algorithm CLI, MCP server and the results UI for running KernelFoundry on custom tasks. Clone instead if you want the KernelBench task set, which is data and does not ship in the package.)
```
pip install 'kernelfoundry[all]'
``` 
Options: [all], [algo] (to execute the algorithm from CLI), [mcp] (only mcp server dependencies, cannot execute locally unless algo is installed), and [ui] (gui for viewing KF results).

Or **install from source**:
```
# Install all dependencies from clone
git clone https://github.com/isl-org/kernelfoundry.git
cd kernelfoundry
uv pip install -e .[all]
```

#### Install torch:
Torch is NOT a requirement for running KernelFoundry; it is only required if the task package is using torch tensors, e.g. to move data to the GPU. However, it is needed for KernelBench examples.

```
# Install torch. Intel GPUs (XPU wheels):
python -m pip install torch==2.9 torchvision==0.24 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu
# NVIDIA GPUs instead:
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

### Configuring the LLM API

The algorithm runs an evolutionary search autonomously, rather than step by step with an agent. 
The generation step calls an LLM API. Set the key for your provider:

```bash
export OPENAI_API_KEY=...      # server_type: openai
export ANTHROPIC_API_KEY=...   # server_type: anthropic
```

Choose the model and provider in [`kernelfoundry/configs/inference/server.yaml`](https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/configs/inference/server.yaml). Any model your provider offers works; `model_name` is passed straight through. 

### Run the optimization algorithm

Two examples are provided within the repo. Running them requires cloning the repo instead of pip installing it, so that the task files are found. Both of them require a torch installation in the unit tests.

```
# Optimize a KernelBench task 
python -m kernelfoundry.algorithm run task=19_ReLU task_origin=KernelBench \
    job_name=my_experiment gpu_arch=lnl language=SYCL

# Optimize your own task
python -m kernelfoundry.algorithm run task=tasks/example_custom task_origin=custom \
    job_name=my_custom_experiment gpu_arch=lnl language=SYCL
```

### Configuring a job

You can configure the optimization algorithm in the config.yaml file of your job. 
Common parameters are:

| Parameter | Meaning |
|---|---|
| `task` | Path to a task folder, or a KernelBench task ID (e.g. `19_ReLU`) |
| `task_origin` | `KernelBench`, `robust_kbench`, or `custom` |
| `gpu_arch` | Target architecture. See [Target hardware](#target-hardware) |
| `language` | `SYCL`, `OCL`, `CUDA` or `triton` (defaults by GPU vendor) |
| `job_name` | Your label for the run; appears in the UI |
| `max_iters` | Optimization iterations (default 3) |
| `branches_per_iteration` | Candidates generated per iteration (default 2) |

See [Config Parameters](https://isl-org.github.io/kernelfoundry/docs/guide/config-parameters.html) for a full list.

### Target hardware

| | Intel GPU | NVIDIA GPU |
|---|---|---|
| Languages | SYCL, OpenCL | CUDA |
| `gpu_arch` values | `lnl` (Lunar Lake), `ptl` (Panther Lake), `bmg` (Battlemage), `dg2` (Flex 170 / Arc A770) | `Ampere` is the only one with a hardware profile and baselines. `Maxwell`, `Pascal`, `Volta`, `Turing`, `Hopper`, `Ada` and `native` run, but without hardware-specific prompt context or a KernelBench speedup figure |
| Default profiler | unitrace for SYCL, VTune for OpenCL | ncu (ships with the CUDA toolkit) |

`gpu_arch` and `language` must agree: a CUDA architecture requires `language: CUDA`. Both accept
a comma-separated list to benchmark on several targets.


## Results and monitoring

Every candidate is recorded, including the ones that fail: the prompt, the model's answer, the
measured runtimes and the evaluation log. Nothing measured gets thrown away.

Generated kernels, jobs and metrics are stored in a SQLite database, by default
`runs/kernels.sqlite3` relative to where you started the run. Change it with
`paths.kernels_db_path`.

> When the MCP server runs in local execution mode under VS Code, the database is written
> relative to your home directory, because that is where the editor launches the server from.

Browse results in the web UI:

```bash
python -m kernelfoundry.gui
```

It serves `http://localhost:8885` and shows:

- Job logs and execution monitoring
- Kernel performance metrics and comparisons
- Optimization progress visualization
- Roofline analysis and performance profiling
- Generated kernel source and metadata

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `KeyError: 'OPENAI_API_KEY'` | The algorithm needs an LLM key. Export `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` with `server_type: anthropic`. |
| `Could not find 'run'` from hydra | Your `configs/` copy is being used instead of the shipped one. Drop `--config-dir`, or point it at `kernelfoundry/configs`. |
| `Unknown architecture gpu_arch: …` | Use a value from the [`gpu_arch` table](#target-hardware). |
| `gpu_arch must be specified` | Pass `gpu_arch=…` on the command line or set it in the task's `config.yaml`. |
| `Server type … not available` | `server_type` must be `openai` or `anthropic`. |
| MCP: `Missing required config keys` | A `server_url` is set, so `user` and `token` are required too. Provide them via environment variables or `~/.config/kernelfoundry/config.yml`. |
| Profiler permission errors | Enable non-root profiling. See [Install a profiler](#getting-started). |
| Invalid keys when merging a task config | Your task's `config.yaml` contains keys absent from `kernelfoundry/configs/run.yaml`. Usually a typo. |

## Documentation

Full documentation, including the user guide and API reference, is at
**[isl-org.github.io/kernelfoundry/docs](https://isl-org.github.io/kernelfoundry/docs/index.html)**.

This README covers deciding, installing and getting a first run. The guide covers everything
after that:

| Guide | What it answers |
|---|---|
| [Quickstart](https://isl-org.github.io/kernelfoundry/docs/guide/quickstart.html) | Run the shipped example end to end, no LLM API key needed |
| [Anatomy of a task package](https://isl-org.github.io/kernelfoundry/docs/guide/task-package.html) | What goes in the folder you submit |
| [Writing tests](https://isl-org.github.io/kernelfoundry/docs/guide/writing-tests.html) | Correctness tests and the benchmark that speedup is measured from |
| [Config parameters](https://isl-org.github.io/kernelfoundry/docs/guide/config-parameters.html) | Every parameter worth setting, with defaults |
| [Understanding results](https://isl-org.github.io/kernelfoundry/docs/guide/understanding-results.html) | Reading a finished run and diagnosing what went wrong |
| [Optimization strategies](https://isl-org.github.io/kernelfoundry/docs/guide/optimization-strategies.html) | Levers to pull when a run plateaus |
| [Public API](https://isl-org.github.io/kernelfoundry/docs/api/public.html) | The classes, helpers and fixtures a task author uses |

To build the docs locally:

```bash
pip install .[docs]
cd docs
make html          # output in docs/_build/html/
```

## Contributing

Bug reports and pull requests are welcome via
[GitHub issues](https://github.com/isl-org/kernelfoundry/issues). Code is formatted with
[black](https://github.com/psf/black) (line length 120) and checked in CI.

## License

Apache License 2.0. See
[LICENSE](https://github.com/isl-org/kernelfoundry/blob/main/LICENSE).

## Citation

If you use KernelFoundry in your research, please cite:

```bibtex
@inproceedings{wiedemann2026kernelfoundry,
  title={KernelFoundry: Hardware-aware evolutionary GPU kernel optimization},
  author={Wiedemann, Nina and Leboutet, Quentin and Paulitsch, Michael and Wofk, Diana and Ummenhofer, Benjamin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML 2026)},
  year={2026}
}
```
