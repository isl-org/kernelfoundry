from __future__ import annotations

import argparse
import sys


def run_mcp_config_wizard() -> dict[str, object]:
    """Prompt for KernelFoundry connection settings and return an MCP client config dict."""
    import getpass

    url_in = input("KernelFoundry server URL [optional; leave empty for local execution]: ").strip()
    server_url = url_in

    env: dict[str, str] = {
        # Set explicitly so local mode can override values coming from config.yml.
        "KERNELFOUNDRY_SERVER_URL": server_url,
    }

    if server_url:
        default_user = getpass.getuser()
        user_in = input(f"User [{default_user}]: ").strip()
        user = user_in or default_user

        token = getpass.getpass("Bearer token [required]: ").strip()

        if not user:
            raise SystemExit("User is required when server URL is provided.")
        if not token:
            raise SystemExit("Token is required when server URL is provided.")

        env["KERNELFOUNDRY_USER"] = user
        env["KERNELFOUNDRY_TOKEN"] = token

    return {
        "mcpServers": {
            "kernelfoundry-mcp": {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "kernelfoundry.mcp_server"],
                "env": env,
            }
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KernelFoundry MCP stdio server.")
    parser.add_argument(
        "positional",
        nargs="*",
        default=[],
        metavar="ARG",
        help="Pass 'create_config' as the only argument to run the config wizard and print JSON.",
    )
    opts = parser.parse_args()
    pos: list[str] = opts.positional
    if pos == ["create_config"]:
        import json

        print(json.dumps(run_mcp_config_wizard(), indent=2))
    elif len(pos) == 2 and pos[0] == "_internal":
        from kernelfoundry.mcp_server.server import main as server_main

        server_main(pos[1])
    elif pos and pos[0] == "_internal":
        parser.error("'_internal' requires exactly one path argument")
    elif pos:
        parser.error(
            "unexpected arguments: %r (use no arguments to run the server, 'create_config' alone, or '_internal <path>')"
            % (" ".join(pos),)
        )
    else:
        from kernelfoundry.mcp_server.server import main as server_main

        server_main()
