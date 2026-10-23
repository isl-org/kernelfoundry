# KernelFoundry MCP Server

A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes tools to build and optimize
kernels (`build_and_test`, `submit_task`) and to search the shared database of previously
generated kernels (`search_kernels_in_db`, `get_kernel_from_db`). The coding agent assembles the task package
from the kernel you point it at, so this is a setup guide rather than something you feed by hand.

Evaluation runs locally, on your own GPU, unless you configure a `server_url`. Local is the
default and is what most of this guide assumes; submitting to a server is opt-in and covered
under [Submitting to a server](#submitting-to-a-server-instead-of-running-locally).

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

## Register the server with your agent

This is a stdio server, so **any MCP client can run it** — no editor required. Every client needs
the same two facts: the interpreter from step 3 above, and the module `kernelfoundry.mcp_server`.
What differs between them is only the file format, which is why the sections below spell each one
out.

```bash
python -m kernelfoundry.mcp_server create_config
```

writes a config interactively, in the widely used `mcpServers` shape. VS Code and Codex CLI use
different formats; copy those from below.

Benchmarking runs on your machine, in a subprocess, so KernelFoundry must be installed with a
GPU available. Install it with the `algo` extra so the build and test dependencies are present
(see [instructions](../../README.md#installation)).

Registering the server gets the agent the `build_and_test`/`submit_task` tools, but it still needs
to be told the workflow contract in [`agent-workflow.md`](agent-workflow.md). For Claude Code and
Copilot CLI, a second helper installs it where that client auto-discovers it as a custom agent, so
it's picked up with no mention in your prompt:

```bash
python -m kernelfoundry.mcp_server install_agent --client claude    # or --client copilot
```

Defaults to `--scope user` (`~/.claude/agents/` or `~/.copilot/agents/`, available in every
project); pass `--scope project` to install into the current directory instead
(`.claude/agents/` or `.github/agents/`). Re-running is a no-op once the file matches; pass
`--force` to overwrite a copy you've since hand-edited. Other clients don't have this mechanism —
point the agent at `agent-workflow.md` directly, or reference it from whichever instructions file
they already read (see [Optimize a kernel with a coding agent](../../docs/guide/agentic-workflow.md)).

### Claude Code

```bash
claude mcp add kernelfoundry-mcp -- /path/to/your/python -m kernelfoundry.mcp_server
```

Everything after `--` is the command, passed through untouched; without it, Claude Code would
read `-m` as one of its own flags. Add `--scope user` to register the server for every project
rather than just this one, and `-e KEY=value` for each environment variable. Committing a
`.mcp.json` at the repository root works too, and shares the server with everyone on the project.

### Codex CLI

```bash
codex mcp add kernelfoundry-mcp -- /path/to/your/python -m kernelfoundry.mcp_server
```

Codex stores servers as TOML in `~/.codex/config.toml`, not JSON, so the equivalent by hand is:

```toml
[mcp_servers.kernelfoundry-mcp]
command = "/path/to/your/python"
args = ["-m", "kernelfoundry.mcp_server"]

[mcp_servers.kernelfoundry-mcp.env]
KERNELFOUNDRY_SERVER_URL = "https://your-server"
```

### Cursor

`.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` for every project:

```json
{
  "mcpServers": {
    "kernelfoundry-mcp": {
      "type": "stdio",
      "command": "/path/to/your/python",
      "args": ["-m", "kernelfoundry.mcp_server"]
    }
  }
}
```

### VS Code

1. `Ctrl-Shift-P` or `Cmd-Shift-P`. Select `MCP: Add Server`
2. Choose `Command (stdio)`
3. Command to run: `/path/to/your/python -m kernelfoundry.mcp_server`
4. Enter server id: `kernelfoundry-mcp` or some name you like
5. Global, remote, or workspace depends on your project or workspace

VS Code then opens the file it wrote.

Editing it by hand, note that **VS Code keys this block `servers`**, not `mcpServers` — a config
copied from another client silently registers nothing. It lives in `.vscode/mcp.json` for one
workspace, or the profile-wide file from `MCP: Open User Configuration`.

```json
{
  "servers": {
    "kernelfoundry-mcp": {
      "type": "stdio",
      "command": "/path/to/your/python",
      "args": ["-m", "kernelfoundry.mcp_server"]
    }
  }
}
```

### Anything else

Cline and most other clients take the `mcpServers` block shown under [Cursor](#cursor). Not all
of them use that key, though — Zed wants the same fields under `context_servers` — so if the
server never appears, check the shape yours expects before assuming it failed to start. The
server itself is a plain stdio process and cares about none of this.

### Submitting to a server instead of running locally

Add the credentials to whichever `env` block your client uses:

| Variable | Meaning |
|---|---|
| `KERNELFOUNDRY_SERVER_URL` | The server to submit to. Unset means run locally on your own GPU. |
| `KERNELFOUNDRY_USER` | Your username. Required once `SERVER_URL` is set. |
| `KERNELFOUNDRY_TOKEN` | Your bearer token. Required once `SERVER_URL` is set. |

These can also live in `~/.config/kernelfoundry/config.yml` as `server_url`, `user` and `token`,
which keeps the token out of every client config you own. The environment wins over the file.
Setting `SERVER_URL` without the other two fails with `Missing required config keys`.

### Check it works

Open a chat with your agent and confirm `build_and_test` and `submit_task` appear under
`kernelfoundry-mcp`. Then:

> Can you use the `build_and_test` tool on the folder `tasks/example_custom/`?

The tool returns JSON from which the agent can read the test outcome, runtime stats and
profiler data. The agent can then refine the kernel code and call `build_and_test` again to
benchmark the new version, repeating until it stops improving.

Results are also stored in the database. The client launches the server, so the database is
written relative to *its* working directory — usually your home directory rather than the task
folder.

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
| `best_kernel_id` | int \| None | Database ID of the best kernel, when one was recorded |
| `graph_url` | str \| None | Link into the graph UI for this task and job. Falls back to the local UI address when no `server_url` is set |
| `error` | str \| None | Error message, present only if the job failed |

### `submit_task(folder_path: str)`

Runs the full multi-iteration optimization loop on the task package in `folder_path` (locally,
or on the server if `server_url` is configured), using the `max_iters` and
`branches_per_iteration` set in its `config.yaml`. Writes the best kernel found back into the
task folder, between the `[EVOLVE_START]` / `[EVOLVE_END]` markers.

Returns the same fields as `build_and_test`, with `best_kernel_id` naming the best kernel the
run settled on.

### `search_kernels_in_db(...)`

Searches the shared kernel database for previously generated kernels, e.g. to see how other
kernels solved a similar problem or check whether an optimization construct has already been
tried. Always restricted to `status == "correct"` kernels with more than 0.5x `speedup` over
the reference (a ratio, e.g. 1.5 = 1.5x faster). Results are diversified across `task_name`
(each matching task's best kernel first, then each task's second-best, and so on) rather than
just sorted by speedup, since one job produces many near-identical iterations of the same task.
Supports case-insensitive exact filters `task_name`, `gpu_arch`, and `output_language`; substring
filters `task_name_contains`, `output_code_contains`, and `gpu_arch_contains`; and
`output_code_terms`, which requires every supplied term to occur in the source. Numeric filters
include `max_runtime` (milliseconds) and `min_speedup`/`max_speedup`. Paginated via `limit`
(default 20, max 100) and zero-based `offset`.

Returns a dictionary with:

| Field | Type | Meaning |
|---|---|---|
| `success` | bool | Whether the search completed successfully |
| `total_count` | int | Total number of matching rows across all pages |
| `results` | list[dict] | One compact entry per match (metadata plus a truncated `output_code_preview`) |
| `error` | str \| None | Error message, if the search failed |

### `get_kernel_from_db(kernel_id: int, include_profiler_data: bool = False)`

Fetches the full record for one kernel found via `search_kernels_in_db`, including the untruncated
`output_code` and `eval_log`.

Returns a dictionary with:

| Field | Type | Meaning |
|---|---|---|
| `success` | bool | Whether the lookup completed successfully |
| `kernel` | dict \| None | Full kernel fields, or `None` if no kernel with that ID exists |
| `error` | str \| None | Error message, if the lookup failed |
