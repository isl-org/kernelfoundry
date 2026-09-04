"""CopilotCLIAgent: uses the GitHub Copilot CLI to autonomously work on a task."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import ssl
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from omegaconf import OmegaConf

import kernelfoundry
from kernelfoundry.algorithm.agent_base import AgentBase, BuildAndTestHandler
from kernelfoundry.algorithm.schemas import EvalResult, Program
from kernelfoundry.algorithm.utils.skills import Skill
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.eval_pipeline.utils.container import Image
from kernelfoundry.eval_pipeline.utils.tmp_dir import TemporaryDirectory
from kernelfoundry.algorithm.utils.token_usage import parse_copilot_otel_usage, zero_token_usage

logger = logging.getLogger(__name__)


class CopilotCLIBuildAndTestHandler(BuildAndTestHandler):
    """Build-and-test handler used by the Copilot CLI agent.

    This handler keeps the shared evaluation logic from
    :class:`~kernelfoundry.algorithm.agent_base.BuildAndTestHandler` and adds
    Copilot-specific end-of-session logging.
    """

    def session_end(self, session_log: str, session_id: str, token_usage: dict | None = None) -> None:
        log_extra = {
            "data": {
                "log": session_log,
                "token_usage": token_usage,
            },
            "agent_session_id": session_id,
        }

        logging.raw("session_log", extra=log_extra)


class CopilotCLIAgent(AgentBase):
    """Agent that uses the GitHub Copilot CLI to autonomously work on a task.

    The agent launches the Copilot CLI as a subprocess, provides it with a
    KernelFoundry MCP server in ``_internal`` mode (file-based communication),
    and collects the ``(Program, EvalResult)`` pairs produced by each
    ``build_and_test`` tool call.

    The workspace is created in a temporary directory and cleaned up after
    :meth:`run` returns.

    Args:
        task: The task to work on.
        job_id: Job ID associated with this run.
        task_id: Task ID associated with this run.
        container_image: Optional resolved :class:`~kernelfoundry.eval_pipeline.utils.container.Image`.
            When provided, the Copilot CLI (and the MCP server it spawns) run inside a
            container created from this image.
        copilot_exe: Path or name of the Copilot CLI binary.  Use the
            ``copilot-sim`` script for local testing without GPU hardware.
        env_overrides: Environment variable overrides passed to the Copilot
            subprocess in addition to the current process environment.
        handler: A :class:`~kernelfoundry.algorithm.agent_base.BuildAndTestHandler` instance
            whose :meth:`~kernelfoundry.algorithm.agent_base.BuildAndTestHandler.call` method is
            invoked after every ``build_and_test`` tool call. If omitted,
            :class:`CopilotCLIBuildAndTestHandler` is used so the session log is
            emitted at the end of each :meth:`run` invocation.
        initial_session_state: Base64-encoded COPILOT_HOME contents for
            restoring a prior session (populated automatically by
            :meth:`fork`).
        branch: Branch number for this agent instance.
        parent_session_uuid: The session UUID of the parent agent instance, if any.
        parent_program: The program to be used as the parent for this agent instance, if any.
        skills: Optional list of :class:`~kernelfoundry.algorithm.utils.skills.Skill` instances
            that will be made available to the Copilot CLI agent.
        extra_mcp_servers: Optional additional MCP server definitions merged with
            the default ``kernelfoundry`` MCP server.
    """

    # Path inside the container where the host CA bundle is mounted. Used as the
    # TLS trust anchor (e.g. via NODE_EXTRA_CA_CERTS) so the Copilot CLI can
    # verify endpoints signed by an internal/corporate CA.
    _CONTAINER_CA_BUNDLE = "/etc/kernelfoundry/host-ca-certificates.crt"

    def __init__(
        self,
        task: Task,
        job_id: int,
        task_id: str,
        config: dict,
        container_image: Image | None = None,
        copilot_exe: str = "copilot",
        env_overrides: dict[str, str] | None = None,
        handler: BuildAndTestHandler | None = None,
        initial_session_state: dict | None = None,
        branch: int = 0,
        parent_session_uuid: str | None = None,
        parent_program: Program | None = None,
        skills: list[Skill] | None = None,
        extra_mcp_servers: dict[str, dict] | None = None,
    ):
        super().__init__(
            task=task,
            job_id=job_id,
            task_id=task_id,
            config=config,
            container_image=container_image,
            initial_session_state=initial_session_state,
            build_test_handler=handler,
            branch=branch,
            parent_session_uuid=parent_session_uuid,
            parent_program=parent_program,
            skills=skills,
        )
        if handler is None:
            self._handler = CopilotCLIBuildAndTestHandler()
        self._copilot_exe = copilot_exe
        self._env_overrides = dict(env_overrides or {})
        self._extra_mcp_servers: dict[str, dict] = dict(extra_mcp_servers or {})
        self._current_prompt: str = ""
        self._mcp_pythonpath_mounts: list[tuple[str, str, str]] = []

        self._use_container = self._container_image is not None
        self._host_ca_bundle = self._find_host_ca_bundle() if self._use_container else None
        self._container_python = self._resolve_container_python() if self._use_container else None

        self._tmpdir = TemporaryDirectory(prefix="copilot_agent_", delete=False)
        print(f"Temporary directory for CopilotCLIAgent: {self._tmpdir.name}")
        self._tmpdir_path = Path(self._tmpdir.__enter__())

        self._task_data_dir = self._tmpdir_path / "task_data"
        self._task.task_data.to_disk(output_dir=self._task_data_dir)
        # rm files copilot does not need config.yml, environment/

        self._copilot_home = self._tmpdir_path / "copilot_home"
        self._copilot_home.mkdir()
        if self._initial_session_state:
            self._restore_session_state(self._copilot_home, self._initial_session_state)

        self.token_usage = zero_token_usage()
        self._install_skills(self._copilot_home)

        self._comm_dir = self._tmpdir_path / "mcp_comm"
        self._comm_dir.mkdir()

        mcp_config = self._create_mcp_config(self._comm_dir)
        (self._copilot_home / "mcp-config.json").write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")

    def __del__(self):
        try:
            pass
            # self._tmpdir.__exit__(None, None, None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # AgentBase abstract methods
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        iteration: int,
        timeout: float = 3600.0,
    ) -> list[tuple[Program, EvalResult]]:
        """Run the Copilot CLI agent on the task.

        The Copilot subprocess runs in a background thread while this thread
        drives the MCP file-based communication directly.

        Args:
            prompt: The input prompt piped into the Copilot CLI via stdin.
            timeout: Maximum number of seconds to wait for the agent to finish.
                Defaults to 3600 (one hour).  A :exc:`TimeoutError` is raised
                if the deadline is exceeded.

        Returns:
            A list of ``(Program, EvalResult)`` tuples — one entry per
            ``build_and_test`` tool call made by the agent during the session.
        """
        results: list[tuple[Program, EvalResult]] = []
        self._current_prompt = prompt

        copilot_share_file = self._comm_dir / "copilot-session.md"

        copilot_cmd = [
            self._copilot_exe,
            "--yolo",
            "--resume",
            "--no-ask-user",
            "--no-auto-update",
            "--output-format=text",  # json
            f"--share={str(copilot_share_file)}",
        ]
        if self._use_container:
            cmd = self._wrap_in_container(copilot_cmd)
            env = os.environ.copy()  # host env only needs to locate the container runtime
        else:
            cmd = copilot_cmd
            env = self._build_env()
        stdbuf_path = shutil.which("stdbuf")
        if stdbuf_path:
            cmd = [stdbuf_path, "-oL", "-eL", *cmd]
            logger.debug("Using stdbuf for line-buffered Copilot output: %s", stdbuf_path)
        else:
            logger.warning("stdbuf not found; Copilot subprocess output may be buffered")
        copilot_stdout_log = self._tmpdir_path / "copilot_stdout.log"
        logger.debug("Launching Copilot CLI: %s", cmd)

        deadline = time.monotonic() + timeout
        copilot_done = threading.Event()

        def _run_copilot():
            remaining = deadline - time.monotonic()
            try:
                with copilot_stdout_log.open("w", encoding="utf-8", buffering=1) as stdout_file:
                    proc = subprocess.run(
                        cmd,
                        input=prompt.encode(),
                        env=env,
                        cwd=str(self._task_data_dir),
                        timeout=remaining,
                        stdout=stdout_file,
                        stderr=subprocess.STDOUT,
                    )
                if copilot_stdout_log.exists():
                    logger.info(
                        "Copilot stdout log finalized at %s bytes",
                        copilot_stdout_log.stat().st_size,
                    )
                if proc.returncode != 0:
                    stdout_contents = (
                        copilot_stdout_log.read_text(encoding="utf-8", errors="replace")
                        if copilot_stdout_log.exists()
                        else "<copilot_stdout.log not found>"
                    )
                    logger.warning(
                        "Copilot process exited with non-zero code %d. stdout log (%s):\n%s",
                        proc.returncode,
                        copilot_stdout_log,
                        stdout_contents,
                    )
            except subprocess.TimeoutExpired:
                logger.error("Copilot process timed out")
            finally:
                copilot_done.set()

        copilot_thread = threading.Thread(target=_run_copilot, daemon=True, name="copilot-process")
        copilot_thread.start()

        self._handle_mcp_communication(results, copilot_done, deadline, iteration, copilot_stdout_log)

        copilot_thread.join()

        copilot_share_file_contents = ""
        if copilot_share_file.exists():
            copilot_share_file_contents = copilot_share_file.read_text(encoding="utf-8", errors="replace")
            # print("Copilot session share file contents:\n", copilot_share_file_contents)

        session_log = (
            self._tmpdir_path.joinpath("copilot_stdout.log").read_text(encoding="utf-8", errors="replace")
            if self._tmpdir_path.joinpath("copilot_stdout.log").exists()
            else ""
        )
        otel_usage = parse_copilot_otel_usage(self._tmpdir_path / "copilot-otel.jsonl")
        self.token_usage = otel_usage
        logging.info("Copilot token usage for session %s: %s", self._session_uuid, otel_usage)

        session_end = getattr(self._handler, "session_end", None)
        if callable(session_end):
            session_end(session_log, self._session_uuid, token_usage=otel_usage)
        return results

    def fork(self, branch: int, parent_program: Program | None = None) -> CopilotCLIAgent:
        """Create a new agent instance that continues from the current session state.

        Must not be called while this agent is running.
        """
        return CopilotCLIAgent(
            task=self._task,
            job_id=self._job_id,
            task_id=self._task_id,
            config=self._config,
            container_image=self._container_image,
            copilot_exe=self._copilot_exe,
            env_overrides=self._env_overrides.copy(),
            handler=self._handler,
            initial_session_state=self.session_state(),
            branch=branch,
            parent_session_uuid=self._session_uuid,
            parent_program=parent_program,
            skills=self._skills,
            extra_mcp_servers=self._extra_mcp_servers,
        )

    def session_state(self) -> dict:
        """Return the serialized COPILOT_HOME as a JSON-serialisable dict.

        The values are base64-encoded file contents keyed by relative path.
        Can be called at any time after construction, including after :meth:`run`.
        """
        self._ensure_copilot_home_ownership()
        state: dict[str, str] = {}
        for f in self._copilot_home.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(self._copilot_home))
                state[rel] = base64.b64encode(f.read_bytes()).decode()
        return state

    def _ensure_copilot_home_ownership(self) -> None:
        """Ensure every entry under ``COPILOT_HOME`` is owned by the current user.

        Files written by a containerised Copilot CLI may be owned by ``root``.
        If any entry (including ``COPILOT_HOME`` itself) is not owned by the
        current user, ownership of the whole tree is reset recursively via
        ``sudo chown``. Not executed in non-posix OS since this condition cannot arise there.
        """
        if os.name != "posix":
            logger.debug("Skipping COPILOT_HOME ownership check: not a POSIX platform (os.name=%r)", os.name)
            return

        uid = os.getuid()
        needs_chown = False
        for path in (self._copilot_home, *self._copilot_home.rglob("*")):
            try:
                if path.stat().st_uid != uid:
                    needs_chown = True
                    break
            except OSError as exc:
                logger.warning("Could not stat %s while checking ownership: %s", path, exc)
                needs_chown = True
                break

        if not needs_chown:
            return

        gid = os.getgid()
        logger.info("Resetting ownership of %s to %d:%d", self._copilot_home, uid, gid)
        try:
            subprocess.run(
                ["sudo", "chown", "-R", f"{uid}:{gid}", str(self._copilot_home)],
                check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.error("Failed to reset ownership of %s: %s", self._copilot_home, exc)
            raise

    # ------------------------------------------------------------------
    # MCP internal-communication handler
    # ------------------------------------------------------------------

    def _handle_mcp_communication(
        self,
        results: list[tuple[Program, EvalResult]],
        copilot_done: threading.Event,
        deadline: float,
        iteration: int,
        copilot_stdout_log: Path,
    ) -> None:
        """Service file-based ``build_and_test`` requests from the MCP server.

        Runs in the same thread as :meth:`run`.  Loops until Copilot has exited
        *and* there are no more pending requests to process, or until *deadline*
        (a :func:`time.monotonic` timestamp) is reached — whichever comes first.

        Raises:
            TimeoutError: if the deadline is reached before Copilot finishes.
        """
        input_done_path = self._comm_dir / "input.json.done"
        input_path = self._comm_dir / "input.json"
        output_path = self._comm_dir / "output.json"
        output_done_path = self._comm_dir / "output.json.done"

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("CopilotCLIAgent timed out")

            if input_done_path.exists():
                response: dict = {"success": False, "job_id": self._job_id, "eval_log": "Unknown error"}
                try:
                    payload = json.loads(input_path.read_text(encoding="utf-8"))
                    folder_path = str(Path(payload["folder_path"]).resolve())
                    session_log = ""
                    log_exists = copilot_stdout_log.exists()
                    log_size = copilot_stdout_log.stat().st_size if log_exists else 0
                    if copilot_stdout_log.exists():
                        session_log = copilot_stdout_log.read_text(encoding="utf-8", errors="replace")
                    input_done_path.unlink()

                    evaluate_result = self._handler.call(
                        task=self._task,
                        folder_path=folder_path,
                        job_id=self._job_id,
                        task_id=self._task_id,
                        prompt=self._current_prompt,
                        iteration=iteration,
                        branch=self._branch,
                        llm_model="copilot-cli",
                        session_log=session_log,
                        previous_program=self._parent_program,
                        agent_session_id=self._session_uuid,
                    )
                    self.set_parent_program(evaluate_result.program)
                    response = evaluate_result.tool_response
                    if evaluate_result.program is not None and evaluate_result.eval_result is not None:
                        results.append((evaluate_result.program, evaluate_result.eval_result))
                    logger.debug("build_and_test completed: success=%s", response.get("success"))
                except Exception:
                    err_msg = traceback.format_exc()
                    logger.error("Error in MCP handler: %s", err_msg)
                    response = {
                        "success": False,
                        "job_id": self._job_id,
                        "eval_log": f"Internal agent error:\n{err_msg}",
                    }
                    input_done_path.unlink(missing_ok=True)
                finally:
                    output_path.write_text(json.dumps(response), encoding="utf-8")
                    output_done_path.write_text("", encoding="utf-8")
            elif copilot_done.is_set():
                break
            else:
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_in_container(self, copilot_cmd: list[str]) -> list[str]:
        """Wrap the Copilot CLI invocation in a container ``run`` command.

        The agent temp dir is bind-mounted at the *same* absolute path inside the
        container so that ``COPILOT_HOME``, the MCP ``comm_dir`` and any kernel
        ``folder_path`` reported through the MCP bridge are identical on host and
        container. The kernelfoundry package directory is mounted read-only at
        ``/kernelfoundry`` and exposed via ``PYTHONPATH`` so the in-container MCP
        server always runs the current code (its dependencies are expected to be
        installed in the image).
        """
        volumes = [(str(self._tmpdir_path), str(self._tmpdir_path))]
        volumes.extend(self._mcp_pythonpath_mounts)
        if self._host_ca_bundle is not None:
            volumes.append((str(self._host_ca_bundle), self._CONTAINER_CA_BUNDLE, "ro"))
        run_cmd = self._container_image.runtime.get_run_cmd(
            self._container_image,
            workdir=str(self._task_data_dir),
            volumes=volumes,
            env_vars=self._container_env_vars(),
            gpus=None,  # the Copilot CLI itself needs no GPU; evaluation runs on the host
        )
        # Enable stdin (-i) so the prompt can be piped into the Copilot CLI.
        run_cmd.insert(2, "-i")
        return run_cmd + copilot_cmd

    def _container_env_vars(self) -> dict[str, str]:
        """Environment variables passed into the Copilot container via ``-e``."""
        env = {
            "COPILOT_HOME": str(self._copilot_home),
            "COPILOT_OFFLINE": "true",
            "COPILOT_OTEL_FILE_EXPORTER_PATH": str(self._tmpdir_path / "copilot-otel.jsonl"),
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
            "PYTHONPATH": "/kernelfoundry",
        }
        # Make the mounted host CA bundle the trust anchor for the Copilot CLI
        # (Node) and any other tooling, so TLS to an internally-signed LLM
        # provider endpoint verifies inside the container.
        if self._host_ca_bundle is not None:
            env["NODE_EXTRA_CA_CERTS"] = self._CONTAINER_CA_BUNDLE
            env["SSL_CERT_FILE"] = self._CONTAINER_CA_BUNDLE
            env["REQUESTS_CA_BUNDLE"] = self._CONTAINER_CA_BUNDLE
            env["CURL_CA_BUNDLE"] = self._CONTAINER_CA_BUNDLE
        env.update(self._env_overrides)
        return env

    @staticmethod
    def _find_host_ca_bundle() -> Path | None:
        """Locate the host CA bundle (which includes any internal/corporate CAs).

        Returns the first existing candidate, or ``None`` if none is found. The
        bundle is mounted into the container and used as the TLS trust anchor so
        the in-container Copilot CLI can verify internally-signed endpoints.
        """
        candidates: list[str] = []
        for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            value = os.environ.get(env_name)
            if value:
                candidates.append(value)
        cafile = ssl.get_default_verify_paths().cafile
        if cafile:
            candidates.append(cafile)
        candidates += [
            "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
            "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/Fedora
        ]
        for candidate in candidates:
            path = Path(candidate)
            if path.is_file():
                return path.resolve()
        logger.warning("No host CA bundle found; container TLS may fail for internally-signed endpoints")
        return None

    def _create_mcp_config(self, comm_dir: Path) -> dict:
        """Build the MCP server config written to ``COPILOT_HOME/mcp-config.json``.

        The MCP library spawns the server subprocess with a stripped-down
        default environment (HOME, PATH, SHELL, …) — it does NOT inherit
        PYTHONPATH.  We therefore embed PYTHONPATH explicitly in the ``env``
        block so the server can import ``kernelfoundry``.

        When running in a container, the kernelfoundry package is mounted at
        ``/kernelfoundry`` and the in-container interpreter is referenced by its
        absolute path (resolved from the image). The absolute path is required
        because Copilot rewrites a bare ``python -m <module>`` command into
        ``pipx run <module>``; a path with a separator bypasses that rewrite.
        ``comm_dir`` is identical on host and container thanks to same-path
        mounting.

        Additional MCP servers can be injected via ``extra_mcp_servers`` (set
        through the constructor). Each entry is merged after the built-in
        ``kernelfoundry-mcp`` server.
        """
        # set defaults for command and pythonpath
        if self._use_container:
            command = self._container_python
            pythonpath = str(Path(kernelfoundry.__file__).parent.parent.resolve())
        else:
            import kernelfoundry as _kf_module

            kf_parent = str(Path(_kf_module.__file__).parent.parent.resolve())
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            entries = [e for e in existing_pythonpath.split(os.pathsep) if e]
            abs_entries = [str(Path(e).resolve()) for e in entries]
            if kf_parent not in abs_entries:
                abs_entries.insert(0, kf_parent)
            command = sys.executable
            pythonpath = os.pathsep.join(abs_entries)

        servers = {
            "kernelfoundry-mcp": {
                "type": "stdio",
                "args": ["-m", "kernelfoundry.mcp_server", "_internal", str(comm_dir)],
            },
            **{name: dict(cfg) for name, cfg in self._extra_mcp_servers.items()},
        }

        # Add correct pythonpaths and commands to mcp server and collect required mounts
        mounts: set[tuple[str, str, str]] = set()
        for _, entry in servers.items():
            entry.pop("prompt", None)  # remove prompt if given in config

            if "command" not in entry:
                entry["command"] = command

            env = dict(entry.get("env") or {})
            if "PYTHONPATH" not in env:
                env["PYTHONPATH"] = pythonpath
            if "COPILOT_HOME" not in env:
                env["COPILOT_HOME"] = str(self._copilot_home)

            if self._use_container:
                mapped_entries: list[str] = []
                for original_entry in str(env["PYTHONPATH"]).split(os.pathsep):
                    if not original_entry:
                        continue
                    normalized_entry = original_entry.rstrip(os.sep)
                    base_name = normalized_entry.split(os.sep)[-1]
                    mapped_entry = f"/{base_name}"
                    mapped_entries.append(mapped_entry)
                    mounts.add((original_entry, mapped_entry, "ro"))
                env["PYTHONPATH"] = os.pathsep.join(mapped_entries)

            entry["env"] = env

            # Args may be a Hydra ListConfig.
            if "args" in entry and OmegaConf.is_config(entry["args"]):
                entry["args"] = OmegaConf.to_container(entry["args"], resolve=True)

        self._mcp_pythonpath_mounts = sorted(mounts) if self._use_container else []
        print("Provided MCP servers", servers)
        return {"mcpServers": servers}

    def _resolve_container_python(self) -> str:
        """Return the absolute path of the Python interpreter inside the image.

        Copilot rewrites a bare ``python -m <module>`` MCP command into
        ``pipx run <module>``; supplying an absolute interpreter path (which
        contains a path separator) bypasses that rewrite so the MCP server is
        launched as intended. Querying the image once yields that path.
        """
        base_cmd = self._container_image.runtime.get_run_cmd(self._container_image, reserved_host_memory_kb=None)
        probe_cmd = base_cmd + ["sh", "-c", "command -v python || command -v python3"]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=120)
            candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if candidates:
                return candidates[-1]
        except subprocess.SubprocessError as exc:
            logger.warning("Failed to resolve container python path: %s", exc)
        logger.warning("Falling back to 'python' for the in-container MCP server command")
        return "python"

    def _install_skills(self, copilot_home: Path) -> None:
        """Write the agent's skills into ``COPILOT_HOME/skills``.

        Each skill is written to ``COPILOT_HOME/skills/<name>/`` together with its bundled resources.
        """
        if not self._skills:
            return

        skills_dir = copilot_home / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill in self._skills:
            skill.save(skills_dir / skill.name)
        logger.info("Installed %d skill(s) into %s", len(self._skills), skills_dir)

    def _build_env(self) -> dict[str, str]:
        """Build the subprocess environment, setting COPILOT_HOME.

        Also ensures that the Python path entries needed to import
        ``kernelfoundry`` are preserved as absolute paths so that nested
        subprocesses (e.g. the MCP server spawned by fastmcp) can find the
        package regardless of their working directory.
        """
        env = os.environ.copy()
        env["COPILOT_HOME"] = str(self._copilot_home)
        env["COPILOT_OFFLINE"] = "true"
        # env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
        env["COPILOT_OTEL_FILE_EXPORTER_PATH"] = str(self._tmpdir_path / "copilot-otel.jsonl")
        env["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

        # Resolve relative PYTHONPATH entries to absolute paths so they survive
        # directory changes in child processes.
        existing = env.get("PYTHONPATH", "")
        absolute_entries = [str(Path(p).resolve()) for p in existing.split(os.pathsep) if p]
        # Also add the directory that contains the kernelfoundry package itself
        # (in case it wasn't already on PYTHONPATH but was found via editable install
        # or another mechanism on sys.path).
        import kernelfoundry as _kf_module

        kf_parent = str(Path(_kf_module.__file__).parent.parent.resolve())
        if kf_parent not in absolute_entries:
            absolute_entries.insert(0, kf_parent)

        if absolute_entries:
            env["PYTHONPATH"] = os.pathsep.join(absolute_entries)

        env.update(self._env_overrides)
        return env

    def _restore_session_state(self, copilot_home: Path, state: dict) -> None:
        """Restore *copilot_home* from a previously serialised state dict."""
        for rel_path, content in state.items():
            target = copilot_home / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(content))
