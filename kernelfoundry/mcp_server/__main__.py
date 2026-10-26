from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVER_NAME = "kernelfoundry-mcp"

# Where the `install_agent` subcommand copies agent-workflow.md
AGENT_INSTALL_TARGETS = {
    ("claude", "user"): Path(".claude/agents/kernelfoundry-optimize.md"),
    ("claude", "project"): Path(".claude/agents/kernelfoundry-optimize.md"),
    ("copilot", "user"): Path(".copilot/agents/kernelfoundry-optimize.agent.md"),
    ("copilot", "project"): Path(".github/agents/kernelfoundry-optimize.agent.md"),
}


def agent_workflow_path() -> Path:
    """The packaged, frontmatter-carrying workflow instructions, installed alongside this module."""
    return Path(__file__).parent / "agent-workflow.md"


def resolve_install_target(client: str, scope: str, home: Path, cwd: Path) -> Path:
    """Where `install_agent` writes, for a given client/scope. Pure function so tests can pass a
    throwaway `home`/`cwd` instead of touching the real filesystem."""
    try:
        relative = AGENT_INSTALL_TARGETS[(client, scope)]
    except KeyError:
        valid = sorted({c for c, _ in AGENT_INSTALL_TARGETS})
        raise SystemExit(f"Unknown client {client!r}. Choose one of: {', '.join(valid)}.") from None
    base = home if scope == "user" else cwd
    return base / relative


# Every client launches the same stdio process; only the file format differs. Clients that share
# a format share an entry here.
#
#   mcpServers  Claude Code (.mcp.json), Cursor, Cline, Windsurf and most others
#   servers     VS Code (.vscode/mcp.json) -- same JSON, different top-level key, and a config
#               copied from another client registers nothing at all
#   toml        Codex CLI (~/.codex/config.toml)
CLIENT_FORMATS = {
    "generic": "mcpServers",
    "claude": "mcpServers",
    "cursor": "mcpServers",
    "vscode": "servers",
    "codex": "toml",
}


def build_server_entry(env: dict[str, str]) -> dict[str, object]:
    """The stdio invocation, as the JSON-shaped clients express it."""
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "kernelfoundry.mcp_server"],
        "env": env,
    }


def _toml_config(env: dict[str, str]) -> str:
    """Render the Codex CLI form.

    Hand-rendered rather than written with a library: the structure is fixed and tomllib is
    read-only, so a dependency would buy nothing. ``json.dumps`` quotes each string, which is
    valid for TOML basic strings.
    """
    entry = build_server_entry(env)
    lines = [
        f"[mcp_servers.{SERVER_NAME}]",
        f"command = {json.dumps(entry['command'])}",
        f"args = [{', '.join(json.dumps(a) for a in entry['args'])}]",
    ]
    if env:
        lines += ["", f"[mcp_servers.{SERVER_NAME}.env]"]
        lines += [f"{key} = {json.dumps(value)}" for key, value in env.items()]
    return "\n".join(lines)


def render_config(client: str, env: dict[str, str]) -> str:
    """Return the config text for ``client``, ready to paste into its config file."""
    try:
        shape = CLIENT_FORMATS[client]
    except KeyError:
        raise SystemExit(f"Unknown client {client!r}. Choose one of: {', '.join(sorted(CLIENT_FORMATS))}.") from None
    if shape == "toml":
        return _toml_config(env)
    return json.dumps({shape: {SERVER_NAME: build_server_entry(env)}}, indent=2)


def collect_settings(
    server_url: str | None = None,
    user: str | None = None,
    token: str | None = None,
    interactive: bool = True,
) -> dict[str, str]:
    """Resolve the connection settings into the ``env`` block the client will set.

    Anything passed in is used as-is; the rest is prompted for when ``interactive``. An empty
    server URL means local execution, which needs no credentials.
    """
    import getpass

    if server_url is None:
        server_url = (
            input("KernelFoundry server URL [optional; leave empty for local execution]: ").strip()
            if interactive
            else ""
        )

    # Set explicitly so local mode can override a server_url coming from config.yml.
    env: dict[str, str] = {"KERNELFOUNDRY_SERVER_URL": server_url}
    if not server_url:
        return env

    if user is None and interactive:
        default_user = getpass.getuser()
        user = input(f"User [{default_user}]: ").strip() or default_user
    if token is None and interactive:
        token = getpass.getpass("Bearer token [required]: ").strip()

    if not user:
        raise SystemExit("User is required when server URL is provided.")
    if not token:
        raise SystemExit("Token is required when server URL is provided.")

    env["KERNELFOUNDRY_USER"] = user
    env["KERNELFOUNDRY_TOKEN"] = token
    return env


def _create_config(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kernelfoundry.mcp_server create_config",
        description="Print an MCP client config for this interpreter. Prompts for anything not given.",
    )
    parser.add_argument(
        "--client",
        default="generic",
        choices=sorted(CLIENT_FORMATS),
        help="Which client's config format to emit (default: generic, the mcpServers shape).",
    )
    parser.add_argument("--server-url", help="Server to submit to. Empty or omitted means run locally.")
    parser.add_argument("--user", help="Username. Required with --server-url.")
    parser.add_argument("--token", help="Bearer token. Required with --server-url.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt, so an agent or script can run this. Missing settings mean local execution.",
    )
    opts = parser.parse_args(argv)

    env = collect_settings(
        server_url=opts.server_url,
        user=opts.user,
        token=opts.token,
        interactive=not opts.non_interactive,
    )
    print(render_config(opts.client, env))
    return 0


def _install_agent(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kernelfoundry.mcp_server install_agent",
        description=(
            "Install agent-workflow.md where the client auto-discovers it as a custom agent, so "
            "it's picked up without the user naming it in every prompt."
        ),
    )
    parser.add_argument(
        "--client",
        required=True,
        choices=sorted({c for c, _ in AGENT_INSTALL_TARGETS}),
        help="Which host's custom-agent convention to install into.",
    )
    parser.add_argument(
        "--scope",
        default="user",
        choices=["user", "project"],
        help=(
            "'user' installs under your home directory, available to every project. 'project' "
            "installs under the current directory -- the project you're optimizing a kernel in, "
            "not kernelfoundry's own repo. Default: user."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists with different content.",
    )
    opts = parser.parse_args(argv)

    content = agent_workflow_path().read_bytes()
    target = resolve_install_target(opts.client, opts.scope, home=Path.home(), cwd=Path.cwd())

    if target.exists():
        if target.read_bytes() == content:
            print(f"{target} is already up to date.")
            return 0
        if not opts.force:
            raise SystemExit(f"{target} already exists with different content. Pass --force to overwrite.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(f"Installed {target}. Open a new chat session for {opts.client} to pick it up.")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "create_config":
        raise SystemExit(_create_config(argv[1:]))
    if argv and argv[0] == "install_agent":
        raise SystemExit(_install_agent(argv[1:]))

    parser = argparse.ArgumentParser(description="KernelFoundry MCP stdio server.")
    parser.add_argument(
        "positional",
        nargs="*",
        default=[],
        metavar="ARG",
        help=(
            "Pass 'create_config' to print an MCP client config, or 'install_agent' to install "
            "agent-workflow.md as a custom agent; add --help after either for its options."
        ),
    )
    opts = parser.parse_args()
    pos: list[str] = opts.positional
    if len(pos) == 2 and pos[0] == "_internal":
        from kernelfoundry.mcp_server.server import main as server_main

        server_main(pos[1])
    elif pos and pos[0] == "_internal":
        parser.error("'_internal' requires exactly one path argument")
    elif pos:
        parser.error(
            "unexpected arguments: %r (use no arguments to run the server, 'create_config' or "
            "'install_agent' alone, or '_internal <path>')" % (" ".join(pos),)
        )
    else:
        from kernelfoundry.mcp_server.server import main as server_main

        server_main()
