# KernelFoundry MCP Server

A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes two tools: `build_and_test`,
which builds and benchmarks a task package once, and `submit_task`, which runs the full
multi-iteration optimization loop. The coding agent assembles the task package from the kernel
you point it at, so this is a setup guide rather than something you feed by hand.

Evaluation runs locally, on your own GPU. That is the only mode this guide covers, and it is the
default; submitting to a remote server is opt-in and stays switched off unless a `server_url` is
configured.

This is the reference for setting the server up. See also:

- [`agent-workflow.md`](agent-workflow.md), the workflow an agent should follow once the server is
  running. Kept alongside the server so an installed package has it without a network fetch.
- [Anatomy of a task package](../../docs/guide/task-package.md), the specification the
  `build_and_test` input must satisfy.
- [Understanding results](../../docs/guide/understanding-results.md) for how to read what the tool
  returns.

## Install the server

1. (Skip if kernelfoundry is already installed) Create a virtual environment and install the kernelfoundry python package with mcp option (see [main readme](../../README.md#install-kernelfoundry-python-package)):
   * Install directly with pip from your package source or repository:

     ```bash
     python -m pip install --verbose 'kernelfoundry[mcp] @ git+https://github.com/isl-org/kernelfoundry.git'
     ```
   * Build and install from source:
     1. Clone the repository.

        ```bash
        git clone git@github.com:isl-org/kernelfoundry.git
        ```

     2. Install this repo as a package. From the root of the repo run:

        ```bash
        python -m pip install ".[mcp]"
        ```
        NOTE: this only installs the dependencies of the MCP server itself. If it should test kernels in local execution mode (see below), run:
        ```bash
        python -m pip install ".[mcp,algo]"
        ```

     3. If the "[mcp]" extra does not work in your shell, install dependencies manually with:

        ```bash
        python -m pip install fastmcp httpx pyyaml
        ```
2. Test if the mcp server starts
   ```bash
   python -m kernelfoundry.mcp_server
   ```
   Stop the process, Ctrl-c
3. Remember the location of your virtual env python binary
   ```bash
   python -c "import sys; print(sys.executable)"
   ```

## Add server to VSCode
1. Ctrl-Shift-P or Cmd-Shift-P. Select `MCP: Add Server`
2. Choose `Command (stdio)`
3. Command to run: `path/to/your/python -m kernelfoundry.mcp_server`
4. Enter server id: `kernelfoundry-mcp` or some name you like
5. Global, remote, or workspace depends on your project or workspace

The mcp.json should now open in your VSCode window.

Benchmarking runs on your machine, in a subprocess, so KernelFoundry must be installed with a
GPU available. Install it with the `algo` extra so the build and test dependencies are present
(see [instructions](../../README.md#installation)).

### Test the MCP server

If everything worked correctly, you should be able to start the MCP server in VSCode. To test
it, open a chat window to your agent (e.g. Copilot or Claude Code). You should see the
`build_and_test` and `submit_task` tools from `kernelfoundry-mcp`. Try this prompt:

> Can you use the `build_and_test` tool on the folder `tasks/example_custom/`?

The tool returns JSON from which the agent can read the test outcome, runtime stats and
profiler data. The agent can then refine the kernel code and call `build_and_test` again to
benchmark the new version, repeating until it stops improving.

Results are also stored in the database. Note that the database is written relative to your
home directory, because that is where VSCode launches the MCP server from.

## Use server with other clients
To use the MCP server with other tools create the more standardized `.mcp.json` which looks slightly different than the VSCode `mcp.json` created by VSCode.
```json
{
    "mcpServers": {
        "kernelfoundry-mcp": {
            "type": "stdio",
            "command": "/path/to/your/python",
            "args": [
                "-m",
                "kernelfoundry.mcp_server"
            ]
        }
    }
}
```
Note that you can generate this file interactively by running
```bash
python -m kernelfoundry.mcp_server create_config
```


## Running

```bash
python -m kernelfoundry.mcp_server
```

## Tools

### `build_and_test(folder_path: str)`

Builds and benchmarks the task package in `folder_path` once, on the local GPU. Optimization
parameters in `config.yaml` are ignored.

Returns a dictionary with:

| Field | Type | Meaning |
|---|---|---|
| `success` | bool | Whether the job completed successfully |
| `job_id` | int | ID of the job. Use it to find the run in the web UI |
| `eval_log` | str | Evaluation log, including build errors and test results |
| `runtime_stats` | dict | Runtime statistics from kernel execution |
| `speedup` | float \| str | Runtime improvement, or `"N/A"` if unavailable |
| `error` | str | Error message, present only if the job failed |

### `submit_task(folder_path: str)`

Runs the full multi-iteration optimization loop on the task package in `folder_path` (locally,
or on the server if `server_url` is configured), using the `max_iters` and
`branches_per_iteration` set in its `config.yaml`. Writes the best kernel found back into the
task folder, between the `[EVOLVE_START]` / `[EVOLVE_END]` markers.

Returns a dictionary with the same fields as `build_and_test`, plus `best_kernel_id` (the
database ID of the best kernel).
