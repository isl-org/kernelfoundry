"""Guard the MCP client config generator.

Each client reads a different file format, and getting one wrong fails silently: VS Code keyed
with ``mcpServers`` instead of ``servers`` registers no server at all and reports no error, so the
user sees an agent that simply has no KernelFoundry tools. These tests parse what we emit rather
than comparing strings, so they fail on a malformed config instead of a reformatted one.

They need no GPU, no LLM key and no network.
"""

import json
import tomllib

import pytest

from kernelfoundry.mcp_server.__main__ import (
    CLIENT_FORMATS,
    SERVER_NAME,
    collect_settings,
    render_config,
)

SERVER_SETTINGS = {"server_url": "https://kf.example", "user": "jdoe", "token": "t0k3n"}


@pytest.fixture
def local_env():
    return collect_settings(server_url="", interactive=False)


@pytest.fixture
def server_env():
    return collect_settings(**SERVER_SETTINGS, interactive=False)


@pytest.mark.parametrize("client", [c for c, shape in CLIENT_FORMATS.items() if shape != "toml"])
def test_json_clients_use_the_documented_top_level_key(client, local_env):
    """``mcpServers`` for most clients, ``servers`` for VS Code. Confusing them registers nothing."""
    config = json.loads(render_config(client, local_env))
    expected_key = CLIENT_FORMATS[client]
    assert list(config) == [expected_key], f"{client} must key its block {expected_key!r}"

    entry = config[expected_key][SERVER_NAME]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["-m", "kernelfoundry.mcp_server"]


def test_codex_emits_parseable_toml(server_env):
    """Codex CLI reads TOML, not JSON, and we hand-render it -- so prove it actually parses."""
    entry = tomllib.loads(render_config("codex", server_env))["mcp_servers"][SERVER_NAME]
    assert entry["args"] == ["-m", "kernelfoundry.mcp_server"]
    assert entry["env"]["KERNELFOUNDRY_SERVER_URL"] == SERVER_SETTINGS["server_url"]


def test_toml_escaping_survives_a_token_with_quotes():
    """Tokens are opaque. A quote or backslash in one must not produce invalid TOML."""
    awkward = 'tok"with\\backslash'
    env = collect_settings(server_url="https://kf.example", user="jdoe", token=awkward, interactive=False)
    entry = tomllib.loads(render_config("codex", env))["mcp_servers"][SERVER_NAME]
    assert entry["env"]["KERNELFOUNDRY_TOKEN"] == awkward


def test_local_execution_needs_no_credentials(local_env):
    """An empty server URL is the default path: run on your own GPU, with nothing to authenticate."""
    assert local_env == {"KERNELFOUNDRY_SERVER_URL": ""}


@pytest.mark.parametrize("missing", ["user", "token"])
def test_server_url_without_credentials_is_rejected(missing):
    """The server rejects these at startup; failing here names the missing key instead."""
    settings = dict(SERVER_SETTINGS, **{missing: ""})
    with pytest.raises(SystemExit, match=missing.capitalize()):
        collect_settings(**settings, interactive=False)


def test_unknown_client_lists_the_supported_ones():
    with pytest.raises(SystemExit, match="Choose one of"):
        render_config("emacs", {})
