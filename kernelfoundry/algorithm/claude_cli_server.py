"""Inference server backed by the Claude Code CLI in headless mode.

Satisfies the same call contract as :class:`~kernelfoundry.algorithm.inference_server.InferenceServer`
so it can be dropped into ``config.inference`` via hydra ``_target_`` with no other changes:

    inference:
      _target_: kernelfoundry.algorithm.claude_cli_server.ClaudeCLIServer
      model_name: opus
      num_completions: 1

Why this exists: the stock InferenceServer reads ``os.environ["OPENAI_API_KEY"]`` (or the
Anthropic equivalent) at construction, so every run is billed against a metered API key. The
``claude`` CLI authenticates against a Claude subscription instead, and ``-p/--print`` makes it
non-interactive.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Auth variables that would override the subscription login in the child process.
_AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_URL",
)

# stop_reason values that mean the completion was cut off rather than finished.
TRUNCATION_STOP_REASONS = {"max_tokens", "refusal"}

# The CLI is a coding agent, not a completion endpoint. Left to itself it will reason about
# the kernel and then reach for Read/Bash to go look at the project, which ends the turn with
# stop_reason=tool_use and no text. This directive is the primary control; _DENIED_TOOLS and
# max_turns are the safety net and the recovery headroom.
_NO_TOOLS_DIRECTIVE = (
    "Answer directly from the prompt alone. Do not use any tools, do not read or write files, "
    "and do not run commands -- everything you need is in this message. Reply with the "
    "requested code and nothing else."
)

# Denied outright so a stray attempt cannot modify the working tree or run anything.
#
# The meta-tools at the end matter as much as the file and shell tools. ToolSearch loads
# deferred tools on demand, so leaving it open reinstates everything below it: with only the
# direct tools denied, the model reached ToolSearch and wrote to the working directory, and
# nothing was recorded in permission_denials because nothing it used was denied.
#
# Deny, do not allowlist. `--allowedTools` adds permissions rather than restricting to the
# named set: passing a single nonexistent tool name left Write and Bash usable.
_DENIED_TOOLS = (
    "Bash",
    "Edit",
    "Write",
    "MultiEdit",
    "NotebookEdit",
    "Read",
    "Glob",
    "Grep",
    "Task",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    # ToolSearch last, and only real tool names in this list: an unrecognized name appears to
    # invalidate the whole flag, which silently re-enables everything.
    "ToolSearch",
)


class ClaudeCLIError(RuntimeError):
    """Raised when the CLI cannot be run or returns something unusable."""


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process and everything it spawned.

    Killing only the direct child leaves its grandchildren running, and on a timeout those are
    exactly what keep the call alive.
    """
    if os.name == "nt":
        # taskkill walks the tree; Popen.kill does not.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logging.error("claude process %s survived a tree kill", proc.pid)


class ClaudeCLIServer:
    """Generate completions by invoking ``claude -p`` as a subprocess.

    Args:
        model_name: Value for ``--model``. An alias such as ``opus`` or ``sonnet`` tracks the
            latest model; a dated id pins one. ``None`` leaves the CLI default.
        num_completions: Completions to return per call. The CLI returns one response per
            invocation, so N completions cost N sequential invocations.
        max_tokens: Recorded for reporting only. The CLI exposes no output-token ceiling, so
            unlike the API path this cannot truncate a response.
        temperature: Accepted and ignored, because the CLI exposes no sampling controls.
            Diversity across branches comes from separate sessions and from the differing
            prompts the evolutionary loop constructs, not from temperature.
        timeout: Per-invocation timeout in seconds.
        max_retry: Retries per completion before giving up.
        claude_exe: Path or name of the CLI binary.
        disallow_tools: Run as a pure text completion with no tool use. Leave this on for the
            evolutionary loop, which wants a kernel back rather than an agent that edits files.
        verbose: Log each invocation.
    """

    def __init__(
        self,
        model_name: str | None = None,
        num_completions: int = 1,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int = 900,
        max_retry: int = 3,
        claude_exe: str = "claude",
        disallow_tools: bool = True,
        max_turns: int = 8,
        effort: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.model = model_name or "default"
        self.model_name = model_name
        self.num_completions = num_completions
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retry = max_retry
        self.claude_exe = claude_exe
        self.disallow_tools = disallow_tools
        self.max_turns = max_turns
        self.effort = effort
        self.verbose = verbose
        self.server_args = dict(kwargs)
        self.server_args.update(
            {"num_completions": num_completions, "max_tokens": max_tokens, "temperature": temperature}
        )

        if temperature not in (None, 0, 0.0):
            logging.warning(
                "ClaudeCLIServer received temperature=%s, which the CLI cannot set. Branch "
                "diversity comes from separate sessions and differing prompts instead.",
                temperature,
            )

        resolved = shutil.which(claude_exe)
        if resolved is None:
            raise ClaudeCLIError(
                f"Claude CLI {claude_exe!r} not found on PATH. Install it, or set claude_exe to its path."
            )
        self.claude_path = resolved
        logging.info("ClaudeCLIServer using %s (model=%s)", resolved, self.model)

    # ------------------------------------------------------------------
    # environment
    # ------------------------------------------------------------------
    def _child_env(self) -> dict:
        """Environment for the subprocess, with API-key auth removed.

        The CLI prefers ANTHROPIC_API_KEY over the subscription login when both are present and
        says so only on stderr, so an inherited key would quietly move every call onto metered
        billing.
        """
        env = {k: v for k, v in os.environ.items() if k not in _AUTH_ENV_VARS}
        return env

    # ------------------------------------------------------------------
    # message handling
    # ------------------------------------------------------------------
    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str, str]:
        """Split OpenAI-style messages into (system_prompt, user_prompt).

        Assistant turns are folded into the prompt with a label; the loop sends single-turn
        prompts in practice, so this only has to be lossless rather than clever.
        """
        system_parts, convo_parts = [], []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):  # tolerate block-style content
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            if role == "system":
                system_parts.append(str(content))
            elif role == "assistant":
                convo_parts.append(f"[previous assistant response]\n{content}")
            else:
                convo_parts.append(str(content))
        return "\n\n".join(system_parts), "\n\n".join(convo_parts)

    # ------------------------------------------------------------------
    # one invocation
    # ------------------------------------------------------------------
    def _invoke(self, system_prompt: str, user_prompt: str) -> tuple[str, dict, str]:
        """Run the CLI once. Returns (text, usage, stop_reason)."""
        cmd = [self.claude_path, "-p", "--output-format", "json"]
        if self.model_name:
            cmd += ["--model", self.model_name]
        if self.effort:
            cmd += ["--effort", self.effort]

        # --system-prompt replaces the agent system prompt, which is what a plain completion
        # wants; --append-system-prompt would keep it.
        full_system = system_prompt
        if self.disallow_tools:
            full_system = (system_prompt + "\n\n" + _NO_TOOLS_DIRECTIVE) if system_prompt else _NO_TOOLS_DIRECTIVE

        if self.disallow_tools:
            cmd += ["--strict-mcp-config"]
        cmd += ["--max-turns", str(self.max_turns)]
        if self.disallow_tools:
            # Last, and nothing after it. --disallowedTools is variadic, and with other flags
            # following it the denial stops taking effect -- the model then reaches Write and
            # Bash with nothing recorded in permission_denials.
            cmd += ["--disallowedTools", " ".join(_DENIED_TOOLS)]

        # stdout and stderr go to temporary FILES rather than pipes, and the child gets its own
        # process group. Both matter for the timeout to actually work:
        #
        #   subprocess.run(timeout=...) kills the child and then waits for its pipes to close.
        #   The CLI spawns grandchildren that inherit those pipes, so the pipes stay open after
        #   the child dies and the wait blocks forever -- the timeout silently does nothing and
        #   the caller hangs indefinitely. Files cannot be held open in that way, and killing
        #   the whole process group takes the grandchildren with it.
        with tempfile.TemporaryDirectory(prefix="claude_cli_") as td:
            out_path = Path(td) / "stdout"
            err_path = Path(td) / "stderr"
            # The CLI runs here rather than in the caller's directory, so nothing it writes can
            # land in the task or the repository.
            workdir = Path(td) / "cwd"
            workdir.mkdir()
            if full_system:
                # By file, never as an argument. On Windows `claude` resolves to npm's claude.CMD,
                # and cmd.exe cuts a batch file's arguments at the first newline: every flag
                # after a multi-line --system-prompt was silently dropped, so --disallowedTools
                # and --max-turns never applied and generation ran as a full agent.
                system_path = Path(td) / "system_prompt"
                system_path.write_text(full_system, encoding="utf-8")
                cmd[1:1] = ["--system-prompt-file", str(system_path)]
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

            with open(out_path, "w", encoding="utf-8") as out_f, open(err_path, "w", encoding="utf-8") as err_f:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=self._child_env(),
                    cwd=str(workdir),
                    creationflags=creationflags,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    proc.communicate(input=user_prompt, timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(proc)
                    raise

            returncode = proc.returncode
            stdout = out_path.read_text(encoding="utf-8", errors="replace")
            stderr = err_path.read_text(encoding="utf-8", errors="replace")

        proc = subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

        if proc.returncode != 0:
            # The CLI reports failures as JSON on stdout and often leaves stderr empty, so
            # both streams have to be surfaced or the error is invisible.
            raise ClaudeCLIError(
                f"claude exited {proc.returncode}. "
                f"stdout: {(proc.stdout or '').strip()[:1500]} | "
                f"stderr: {(proc.stderr or '').strip()[:800]}"
            )

        stdout = (proc.stdout or "").strip()
        if not stdout:
            raise ClaudeCLIError(f"claude produced no output. stderr: {(proc.stderr or '').strip()[:800]}")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCLIError(f"claude output was not JSON: {stdout[:800]}") from exc

        if payload.get("is_error"):
            raise ClaudeCLIError(f"claude reported an error: {str(payload.get('result'))[:800]}")

        # A completion is one turn. More than one means the CLI ran its agent loop despite the
        # no-tools directive, and `result` is then the
        # last turn of a transcript rather than an answer. Those turns read as ordinary prose
        # ("that run is already going in the background"), so nothing downstream recognises them
        # as a failure: the code extractor returns the longest code-like run it can find in
        # English, and the compiler then reports something like `unknown type name 'The'`.
        # Retrying costs one call and usually lands a real completion.
        turns = payload.get("num_turns")
        if isinstance(turns, int) and turns > 1:
            raise ClaudeCLIError(
                f"claude used {turns} turns, so this is an agent transcript rather than a "
                f"completion. Result began: {str(payload.get('result'))[:200]!r}"
            )

        usage = payload.get("usage") or {}
        return (
            payload.get("result") or "",
            {
                "input_tokens": usage.get("input_tokens", 0) or 0,
                "output_tokens": usage.get("output_tokens", 0) or 0,
            },
            payload.get("stop_reason") or "",
        )

    # ------------------------------------------------------------------
    # the InferenceServer contract
    # ------------------------------------------------------------------
    def __call__(self, messages: list[dict], single_question: bool = False, **kwargs):
        """Return ``(outputs, metadata)`` matching InferenceServer.

        Args:
            messages: OpenAI-style ``{"role", "content"}`` dicts.
            single_question: One completion, ignoring ``num_completions``.

        Returns:
            ``(list[str], dict)`` where the dict carries model, token counts and the indices of
            any truncated completions.
        """
        system_prompt, user_prompt = self._split_messages(messages)
        n = 1 if single_question else int(kwargs.get("n", self.num_completions) or 1)

        outputs: list[str] = []
        truncated: list[int] = []
        totals = {"input_tokens": 0, "output_tokens": 0}

        for i in range(n):
            last_exc: Exception | None = None
            for attempt in range(self.max_retry):
                try:
                    tic = time.time()
                    text, usage, stop_reason = self._invoke(system_prompt, user_prompt)
                    if self.verbose:
                        logging.info(
                            "ClaudeCLIServer completion %d/%d in %.1fs (%d output tokens)",
                            i + 1,
                            n,
                            time.time() - tic,
                            usage["output_tokens"],
                        )
                    outputs.append(text)
                    totals["input_tokens"] += usage["input_tokens"]
                    totals["output_tokens"] += usage["output_tokens"]
                    if stop_reason in TRUNCATION_STOP_REASONS:
                        truncated.append(len(outputs) - 1)
                    break
                except subprocess.TimeoutExpired as exc:
                    last_exc = exc
                    logging.warning(
                        "claude timed out after %ss (completion %d, attempt %d/%d)",
                        self.timeout,
                        i + 1,
                        attempt + 1,
                        self.max_retry,
                    )
                except ClaudeCLIError as exc:
                    last_exc = exc
                    logging.warning(
                        "claude call failed (completion %d, attempt %d/%d): %s",
                        i + 1,
                        attempt + 1,
                        self.max_retry,
                        exc,
                    )
                    time.sleep(2)
            else:
                # Carry the last failure's text: without it the caller sees only a count and
                # has to go digging in the log for what the CLI actually said.
                raise ClaudeCLIError(
                    f"claude failed {self.max_retry} times for completion {i + 1}/{n}. " f"Last error: {last_exc}"
                ) from last_exc

        if truncated:
            logging.error(
                "%d of %d completion(s) stopped at a truncation reason; generated code is " "likely incomplete.",
                len(truncated),
                len(outputs),
            )

        metadata = {
            "model": self.model,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "truncated_completions": truncated,
        }
        return outputs, metadata
