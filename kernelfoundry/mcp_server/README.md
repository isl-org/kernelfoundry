# KernelFoundry MCP Server

A [FastMCP](https://github.com/jlowin/fastmcp) server that exposes a single tool,
`build_and_test`, which builds and benchmarks a kernel package — either locally or by
submitting it to the KernelFoundry `/api/validate_job` endpoint.

This is the reference for setting the server up. For the workflow an agent should follow once
it is running, see [`AGENTS.md`](../../AGENTS.md) in the repository root.

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

The mcp.json should now open in your VSCode window. Now, there are two execution modes:
1) **Server**: If KernelFoundry runs on a server with celery and rabbitmq, the MCP server will just zip the task files and send them to the server, where it will be benchmarked on the GPU workers. To use this mode, you need to add the following to the mcp.json dictionary:
    ```json
        "kernelfoundry-mcp": {
            "env": {
                "KERNELFOUNDRY_TOKEN": "YOUR_BEARER_TOKEN_HERE",
                "KERNELFOUNDRY_SERVER_URL": "https://your-kernelfoundry-server.example.com",
                "KERNELFOUNDRY_USER": "YOUR_USER_HERE"
            },
    ```
2) **Locally**: If KernelFoundry is installed locally and a GPU is available, the MCP server will start the benchmarking in a subprocess. All requirements must be installed. 

The tool will run **locally** unless a server configuration is provided. For local execution, make sure you have all requirements installed (see [instructions](../../README.md#installation)).

### Test the MCP server

If everything worked correctly, you should be able to start the MCP server in VSCode. To test
it, open a chat window to your agent (e.g. Copilot or Claude Code) — you should see the
`build_and_test` tool from `kernelfoundry-mcp` — and try this prompt:

> Can you use the `build_and_test` tool on the folder `tasks/example_custom/`?

The tool returns JSON from which the agent can read the test outcome, runtime stats and
profiler data. The agent can then refine the kernel code and call `build_and_test` again to
benchmark the new version, repeating until it stops improving.

Results are also stored in the database. Note that in local execution mode the database is
written relative to your home directory, because that is where VSCode launches the MCP server
from.

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
            ],
            "env": {
                "KERNELFOUNDRY_TOKEN": "YOUR_BEARER_TOKEN_HERE",
                "KERNELFOUNDRY_SERVER_URL": "https://your-kernelfoundry-server.example.com",
                "KERNELFOUNDRY_USER": "YOUR_USER_HERE"
            }
        }
    }
}
```
Note that you can generate this file interactively by running
```bash
python -m kernelfoundry.mcp_server create_config
```


### Server configuration

If no environment variables are provided the server reads its configuration from:

```
~/.config/kernelfoundry/config.yml
```

Required keys:

```yaml
server_url: https://your-kernelfoundry-server.example.com
user: your_user
token: your_bearer_token
```

## Running

```bash
python -m kernelfoundry.mcp_server
```

## Tool

### `build_and_test(folder_path: str)`

Builds and benchmarks the kernel package in `folder_path`. If a server URL is configured, the
folder is zipped and submitted to the KernelFoundry server; otherwise it is evaluated locally.

Returns a dictionary with:

| Field | Type | Meaning |
|---|---|---|
| `success` | bool | Whether the job completed successfully |
| `job_id` | int | ID of the submitted job — use it to find the run in the web UI |
| `eval_log` | str | Evaluation log, including build errors and test results |
| `runtime_stats` | dict | Runtime statistics from kernel execution |
| `speedup` | float \| str | Runtime improvement, or `"N/A"` if unavailable |
| `error` | str | Error message, present only if the job failed |
