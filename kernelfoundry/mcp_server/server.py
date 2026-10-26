"""KernelFoundry MCP server allowing to use components of the pipeline as tools"""

from __future__ import annotations

import asyncio
import io
import re
import json
import logging
import os
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import httpx
import yaml
from fastmcp import Context, FastMCP
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

CONFIG_PATH = Path.home() / ".config" / "kernelfoundry" / "config.yml"
DEFAULT_LOCAL_UI_URL = "http://localhost:8885"

TERMINAL_STATUSES = {"COMPLETE", "FAIL", "CANCELED", "VALIDATED"}
SUCCESS_STATUSES = {"COMPLETE", "VALIDATED"}

mcp = FastMCP("kernelfoundry", instructions="Tools for working with kernelfoundry task directories.")

_internal_comm_path: str | None = None


class BuildAndTestResult(BaseModel):
    """Structured result returned by the ``build_and_test`` MCP tool."""

    success: bool
    job_id: int
    eval_log: str
    runtime_stats: dict[str, Any] = Field(default_factory=dict)
    speedup: float | str = "N/A"
    best_kernel_id: int | None = None
    error: str | None = None
    graph_url: str | None = None


async def _build_and_test_via_internal_comm(folder_path: str, ctx: Context) -> BuildAndTestResult:
    if _internal_comm_path is None:
        raise RuntimeError("Internal communication path is not configured.")

    comm_dir = Path(_internal_comm_path).expanduser().resolve()
    if not comm_dir.exists():
        raise FileNotFoundError(f"Internal communication directory does not exist: {comm_dir}")
    if not comm_dir.is_dir():
        raise NotADirectoryError(f"Internal communication path is not a directory: {comm_dir}")

    poll_interval = float(os.environ.get("KERNELFOUNDRY_INTERNAL_POLL_INTERVAL", "1.5"))
    max_wait = float(os.environ.get("KERNELFOUNDRY_INTERNAL_MAX_WAIT", "3600"))

    input_path = comm_dir / "input.json"
    input_done_path = comm_dir / "input.json.done"
    output_path = comm_dir / "output.json"
    output_done_path = comm_dir / "output.json.done"

    # Avoid consuming stale handshake files from an earlier invocation.
    for stale_path in (input_done_path, output_done_path):
        if stale_path.exists():
            stale_path.unlink()

    payload = {"folder_path": folder_path}

    input_path.write_text(json.dumps(payload), encoding="utf-8")
    input_done_path.write_text("", encoding="utf-8")
    await ctx.report_progress(progress=10, total=100)

    deadline = time.monotonic() + max_wait
    while not output_done_path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {max_wait:.0f}s waiting for internal output marker: {output_done_path}"
            )
        elapsed = max_wait - (deadline - time.monotonic())
        poll_progress = 10 + int(85 * min(elapsed / max_wait, 1.0))
        await ctx.report_progress(progress=poll_progress, total=100)
        await asyncio.sleep(poll_interval)

    if not output_path.exists():
        raise FileNotFoundError(f"Internal output marker exists but output file is missing: {output_path}")

    raw_output = output_path.read_text(encoding="utf-8")
    output_payload = json.loads(raw_output)
    if not isinstance(output_payload, dict):
        raise TypeError("output.json must contain a JSON object.")

    output_done_path.unlink()
    await ctx.report_progress(progress=100, total=100)
    return BuildAndTestResult.model_validate(output_payload)


async def _poll_job_status(
    client: httpx.AsyncClient,
    server_url: str,
    headers: dict[str, str],
    job_id: int,
    poll_interval: float,
    max_wait: float,
    ctx: Context | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait
    while True:
        resp = await client.get(f"{server_url}/api/job/{job_id}", headers=headers)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")
        if status in TERMINAL_STATUSES:
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {max_wait:.0f}s waiting for job {job_id}; last status: {status}")
        if ctx is not None:
            elapsed = max_wait - (deadline - time.monotonic())
            poll_progress = 10 + int(70 * min(elapsed / max_wait, 1.0))
            await ctx.report_progress(progress=poll_progress, total=100)
        await asyncio.sleep(poll_interval)


def _load_config() -> dict[str, Any]:
    """Read MCP server settings from the config file, then the environment.

    ``server_url`` selects the execution path. Left unset, ``build_and_test`` runs locally and no
    credentials are required; set, it submits to a KernelFoundry server and ``user`` and ``token``
    become mandatory.
    """
    config: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"Invalid config format in {CONFIG_PATH}; expected a YAML mapping.")
        config.update(file_config)

    env_map = {
        "server_url": "KERNELFOUNDRY_SERVER_URL",
        "user": "KERNELFOUNDRY_USER",
        "token": "KERNELFOUNDRY_TOKEN",
        "timeout": "KERNELFOUNDRY_TIMEOUT",
        "poll_interval": "KERNELFOUNDRY_POLL_INTERVAL",
        "max_wait": "KERNELFOUNDRY_MAX_WAIT",
    }
    for key, env_var in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            config[key] = value

    # If no server URL is configured we run the evaluator locally and do not need auth credentials.
    server_url = str(config.get("server_url", "")).strip()
    config["server_url"] = server_url
    if not server_url:
        return config

    missing = [k for k in ("user", "token") if not config.get(k)]
    if missing:
        raise ValueError(
            f"Missing required config keys: {', '.join(missing)}. "
            f"Provide them in {CONFIG_PATH} or via environment variables: " + ", ".join(env_map[k] for k in missing)
        )
    return config


def _format_server_request_error(error: Exception, server_url: str) -> str:
    if not isinstance(error, httpx.HTTPStatusError) or error.response.status_code != 403:
        return str(error)

    parsed_url = urlsplit(server_url)
    host = parsed_url.hostname
    if host and ":" in host:
        host = f"[{host}]"
    if host and parsed_url.port:
        host = f"{host}:{parsed_url.port}"
    host = host or "the server host from server_url"
    return (
        f"{error}. This may be a proxy issue. Add {host} to the NO_PROXY or no_proxy environment "
        "variable for the MCP server, then restart it."
    )


def _zip_folder(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(folder):
            # Do not include local KernelFoundry metadata from the task root.
            if Path(root).relative_to(folder).parts[:1] == (".kernelfoundry",):
                continue
            for name in files:
                abs_path = Path(root) / name
                rel_path = abs_path.relative_to(folder)
                zf.write(abs_path, arcname=rel_path.as_posix())
    return buffer.getvalue()


def _append_kernel_uuids(folder: Path, kernels: list[dict[str, Any]]) -> None:
    uuids = [str(k["uuid"]) for k in kernels if k.get("uuid")]
    if not uuids:
        return
    meta_dir = folder / ".kernelfoundry"
    kernels_path = meta_dir / "kernels.json"
    meta_dir.mkdir(parents=True, exist_ok=True)
    existing: list[str] = json.loads(kernels_path.read_text(encoding="utf-8")) if kernels_path.exists() else []
    kernels_path.write_text(json.dumps(existing + uuids, indent=2), encoding="utf-8")


def _get_last_kernel_uuid(folder: Path) -> str | None:
    kernels_path = folder / ".kernelfoundry" / "kernels.json"
    if not kernels_path.exists():
        return None
    loaded = json.loads(kernels_path.read_text(encoding="utf-8"))
    return loaded[-1] if isinstance(loaded, list) and loaded else None


def _read_task_and_job_name(folder: Path) -> tuple[str | None, str | None]:
    """Read ``task_name``/``job_name`` straight from the task folder's ``config.yaml``.

    Avoids constructing a full ``Task`` (which pulls in the optional ``algo`` extra) just
    to read two plain config fields.
    """
    for name in ("config.yaml", "config.yml"):
        config_path = folder / name
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            return config.get("task_name"), config.get("job_name")
    return None, None


def _build_graph_url(server_url: str, task_name: str | None, job_name: str | None) -> str | None:
    """Link into the KernelFoundry graph UI for the given task/job, or None if unavailable.

    When running fully locally (no ``server_url`` configured), falls back to the default
    local ``python -m kernelfoundry.gui`` address.
    """
    if not task_name or not job_name:
        return None
    base_url = (server_url or DEFAULT_LOCAL_UI_URL).rstrip("/")
    return f"{base_url}/graph?job_name={quote(str(job_name))}&task_name={quote(str(task_name))}"


async def _run_locally(folder: Path, ctx: Context, *, run_optimization: bool) -> BuildAndTestResult:
    """Shared local execution: validate-only or validate + optimize. Always returns the best DB kernel."""
    from hydra import compose, initialize_config_dir

    from kernelfoundry import CONFIG_DIR
    import kernelfoundry.eval_pipeline.database as db
    from kernelfoundry.eval_pipeline.task import Task
    from kernelfoundry.algorithm.utils.database_log_handler import DatabaseLogHandler
    from sqlalchemy import select
    import logging as _logging

    source_task, metadata = Task.create(folder)
    evolve_block = source_task.blocks.get("EVOLVE")
    if not evolve_block:
        raise ValueError(f"Task at {folder} is missing required EVOLVE block.")

    # _run_locally is only ever called when no server_url is configured, so the graph
    # link falls back to the default local `python -m kernelfoundry.gui` address.
    graph_url = _build_graph_url("", source_task.config.get("task_name"), source_task.config.get("job_name"))

    # Load base config using Hydra's config composition
    config_dir = CONFIG_DIR
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        base_config = compose(config_name="run")

    # Merge: base_config -> source_task.config
    merged_config = OmegaConf.merge(base_config, source_task.config)

    # Keep local validation behavior aligned with the CLI path.
    task_origin = source_task.config.get("task_origin", "custom")
    source_task.config["task_origin"] = task_origin

    await ctx.report_progress(progress=20, total=100)

    db.init(merged_config)
    task_db = source_task.to_database_task()
    task_id = task_db.id
    db.add_ignore_errors(task_db)

    job_id = db.add_job(task_origin=task_origin, status="INIT") or 0
    if job_id:
        db.update_job_status(job_id, status="INIT", task_id=task_id, config=source_task.config.copy())

    db_handler = DatabaseLogHandler(job_id, level=_logging.DEBUG)
    db_handler.setFormatter(_logging.Formatter("%(message)s"))
    _logging.getLogger().addHandler(db_handler)
    try:
        await ctx.report_progress(progress=40, total=100)

        if run_optimization:
            from kernelfoundry.algorithm.controller import Controller

            assert merged_config.get("max_iters", 0) > 0, "Cannot run controller with max_iters <= 0"
            controller = Controller(config=merged_config, job_id=job_id, task_id=task_id)
            await asyncio.to_thread(controller.run_single, source_task)
        else:
            from kernelfoundry.algorithm.utils.validate_task import validate_task

            await asyncio.to_thread(
                validate_task,
                source_task,
                merged_config,
                job_id,
                task_id,
                parent_uuid=_get_last_kernel_uuid(folder),
            )

        await ctx.report_progress(progress=90, total=100)

        with db.SessionRO() as session:
            kernels = session.execute(select(db.Kernel).where(db.Kernel.job_id == job_id)).scalars().all()

        if not kernels:
            await ctx.report_progress(progress=100, total=100)
            return BuildAndTestResult(
                success=True,
                job_id=int(job_id),
                eval_log="No kernels produced.",
                speedup="N/A",
                graph_url=graph_url,
            )

        best = max(
            kernels,
            key=lambda k: _kernel_performance_score(
                {"score": k.score, "speedup": k.improve_over_native, "runtime": k.runtime}
            ),
        )
        if best.uuid:
            _append_kernel_uuids(folder=folder, kernels=[{"uuid": best.uuid}])
        if run_optimization and best.output_code:
            _apply_output_code(folder=folder, output_code=str(best.output_code))
        await ctx.report_progress(progress=100, total=100)
        return BuildAndTestResult(
            success=True,
            job_id=int(job_id),
            eval_log=best.eval_log or "",
            runtime_stats=best.runtime_stats or {},
            speedup=best.improve_over_native if best.improve_over_native and best.improve_over_native > 0 else "N/A",
            best_kernel_id=best.id,
            graph_url=graph_url,
        )
    except Exception as e:
        await ctx.report_progress(progress=100, total=100)
        return BuildAndTestResult(
            success=False,
            job_id=int(job_id),
            eval_log=f"Local execution failed with exception:\n{traceback.format_exc()}",
            speedup="N/A",
            error=str(e),
            graph_url=graph_url,
        )
    finally:
        _logging.getLogger().removeHandler(db_handler)
        db_handler.close()


def _apply_output_code(folder: Path, output_code: str) -> None:
    """Splice optimized kernel code back between [EVOLVE_START] / [EVOLVE_END] markers.

    ``output_code`` is the string produced by ``blocks_to_str``: for a single EVOLVE
    block it is the raw code; for multiple blocks the files are separated by
    ``### Block: <filename> ###`` header lines.
    """

    # Parse multi-block format: ['', filename1, code1, filename2, code2, ...]
    parts = re.split(r"^### Block: (.+?) ###\n", output_code, flags=re.MULTILINE)
    if len(parts) > 1:
        it = iter(parts[1:])  # skip leading empty string
        blocks: dict[str, str] = dict(zip(it, it))
    else:
        # Single block — find the EVOLVE file by scanning the folder
        blocks = {}
        for candidate in sorted(folder.rglob("*")):
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            if "[EVOLVE_START]" in text and "[EVOLVE_END]" in text:
                blocks[str(candidate.relative_to(folder))] = output_code
                break

    for rel_filename, code in blocks.items():
        target = folder / rel_filename
        if not target.exists():
            continue
        original = target.read_text(encoding="utf-8")
        updated = re.sub(
            r"(\[EVOLVE_START\]\n)(.*?)(\n[^\n]*\[EVOLVE_END\])",
            lambda m, _code=code: m.group(1) + _code + m.group(3),
            original,
            flags=re.DOTALL,
        )
        target.write_text(updated, encoding="utf-8")


def _kernel_performance_score(kernel: dict[str, Any]) -> float:
    """Compute a performance score for a kernel (correctness + speedup or 1/runtime)"""
    perf_score = kernel.get("score") or 0
    speedup, runtime = kernel.get("speedup"), kernel.get("runtime")

    score = float(perf_score)
    if speedup is not None and speedup > 0:
        score += speedup
    elif (speedup is None or speedup == -1) and runtime is not None and runtime > 0:
        score += 1.0 / runtime
    return score


def _ensure_db() -> None:
    """Lazily initialize a direct DB connection, once per process.

    Unlike ``build_and_test``/``submit_task`` (which may delegate evaluation to a remote GPU
    worker via ``server_url``), DB reads always connect directly: ``db.init()`` already picks
    the shared Postgres instance when ``DB_IP`` is set in the environment, or falls back to
    local SQLite otherwise.
    """
    import kernelfoundry.eval_pipeline.database as db

    if db.SessionRO is not None:
        return

    from hydra import compose, initialize_config_dir

    from kernelfoundry import CONFIG_DIR

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="run")
    db.init(cfg)


async def _run_server_job(
    folder: Path,
    ctx: Context,
    *,
    endpoint: str,
    extra_form_data: dict[str, str] | None = None,
    pick_kernel: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Submit a task archive to the server, poll for completion, and return results.

    Only used when KernelFoundry is deployed on a server and ``server_url`` is configured in
    ``mcp.json``; otherwise the evaluation runs locally through :func:`_run_locally`.

    Args:
        folder: Resolved path to the task folder.
        ctx: MCP context for progress reporting.
        endpoint: The API endpoint path to POST to (e.g. ``/api/validate_job``).
        extra_form_data: Additional form fields beyond ``idsid`` (e.g. ``{"parent_uuid": ...}``).
        pick_kernel: Selects one kernel dict from the ``/api/kernels_by_job`` list.

    Returns:
        Dict with ``success``, ``job_id``, ``eval_log``, ``runtime_stats``,
        ``speedup``, ``chosen_kernel_id``, and ``graph_url`` keys.
    """
    config = _load_config()
    server_url = str(config.get("server_url", "")).rstrip("/")
    if not server_url:
        raise RuntimeError(
            "No server URL configured. " f"Set KERNELFOUNDRY_SERVER_URL or add server_url to {CONFIG_PATH}."
        )

    user = config["user"]
    token = config["token"]
    timeout = float(config.get("timeout", 1200.0))
    poll_interval = float(config.get("poll_interval", 5.0))
    max_wait = float(config.get("max_wait", 3600.0))

    task_name, job_name = _read_task_and_job_name(folder)
    graph_url = _build_graph_url(server_url, task_name, job_name)

    archive_bytes = _zip_folder(folder)

    await ctx.report_progress(progress=5, total=100)

    filename = f"{folder.name}.zip"
    headers = {"Authorization": f"Bearer {token}"}
    form_data: dict[str, str] = {"idsid": user}
    if extra_form_data:
        form_data.update(extra_form_data)

    eval_log: str | None = None
    runtime_stats: dict[str, Any] = {}
    speedup: str | float = "N/A"
    chosen_kernel_id: int | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 1. Submit the job to api (validate / submit)
        submit_resp = await client.post(
            f"{server_url}{endpoint}",
            headers=headers,
            files={"file": (filename, archive_bytes, "application/zip")},
            data=form_data,
        )
        try:
            submission = submit_resp.json()
        except ValueError:
            submission = {"raw_response": submit_resp.text}

        if submit_resp.status_code >= 400:
            raise RuntimeError(f"Job submission to {endpoint} failed ({submit_resp.status_code}): {submission}")

        job_id = submission.get("job_id")
        if job_id is None:
            raise RuntimeError(f"Server did not return a job_id. Response: {submission}")

        await ctx.report_progress(progress=10, total=100)

        # 2. Poll for job status until terminal state
        job_info = await _poll_job_status(
            client=client,
            server_url=server_url,
            headers=headers,
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            ctx=ctx,
        )
        final_status = job_info.get("status")

    await ctx.report_progress(progress=80, total=100)

    # Open a fresh client for result fetching — the previous connection pool may have
    # gone stale if the job ran for a long time.
    async with httpx.AsyncClient(timeout=timeout) as client:
        # 3. Get the kernels for the job
        kernels_resp = await client.get(
            f"{server_url}/api/kernels_by_job",
            headers=headers,
            params={"job_id": job_id},
        )
        kernels_resp.raise_for_status()
        kernels = kernels_resp.json().get("kernels", [])
        _append_kernel_uuids(folder=folder, kernels=kernels)

        await ctx.report_progress(progress=90, total=100)

        output_code: str | None = None
        if kernels:
            chosen_kernel = pick_kernel(kernels)
            chosen_kernel_id = chosen_kernel.get("id") if chosen_kernel else None
            output_code = chosen_kernel.get("output_code") if chosen_kernel else None
            if chosen_kernel_id is not None:
                log_resp = await client.get(
                    f"{server_url}/api/kernels/{chosen_kernel_id}/eval_log",
                    headers=headers,
                )
                if log_resp.status_code < 400:
                    eval_log = log_resp.json().get("eval_log")
                    runtime_stats = log_resp.json().get("runtime_stats", {})
                    speedup = log_resp.json().get("speedup", "N/A")
        else:
            logs_resp = await client.get(
                f"{server_url}/api/jobs/logs",
                headers=headers,
                params={"job_id": job_id},
            )
            logs_resp.raise_for_status()
            logs = logs_resp.json().get("logs", [])
            raw_logs = [raw_log for l in logs if (raw_log := l.get("raw_log"))]
            eval_log = "\n".join(raw_logs) if raw_logs else None

        if eval_log is None:
            eval_log = "Server error: Failed to retrieve evaluation log."

    await ctx.report_progress(progress=100, total=100)

    return {
        "success": final_status in SUCCESS_STATUSES,
        "job_id": job_id,
        "eval_log": eval_log,
        "runtime_stats": runtime_stats,
        "speedup": speedup,
        "chosen_kernel_id": chosen_kernel_id,
        "_output_code": output_code,
        "graph_url": graph_url,
    }


@mcp.tool()
async def search_kernels_in_db(
    task_name_contains: str | None = None,
    task_name: str | None = None,
    output_code_contains: str | None = None,
    output_code_terms: list[str] | None = None,
    output_language: str | None = None,
    gpu_arch_contains: str | None = None,
    gpu_arch: str | None = None,
    max_runtime: float | None = None,
    min_speedup: float | None = None,
    max_speedup: float | None = None,
    limit: int = 20,
    offset: int = 0,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Searches the shared KernelFoundry kernel database for previously generated kernels.

    Every task run through KernelFoundry (``submit_task``/``build_and_test``) writes one row
    per generated kernel to a shared database. Use this tool to find prior kernels matching a
    task or code pattern -- e.g. to see how other kernels solved a similar problem, or to check
    whether a given optimization construct has already been tried. Follow up on interesting
    matches with ``get_kernel_from_db(id)`` to fetch the full source and evaluation log.

    Always restricted to kernels with ``status == "correct"`` and ``speedup > 0.5`` (ratio over
    reference, e.g. 1.5 = 1.5x faster) -- below 1.0x since a kernel can still be
    good if compared against an especially strong native reference. Results are diversified
    across task_name (each task's best kernel first, then each task's second-best, and so
    on) rather than plain speedup order, since one job produces many near-identical
    iterations of the same task. All other filters are optional and combined with AND.

    Args:
        task_name: Case-insensitive exact match on the task name.
        task_name_contains: Case-insensitive substring match on the task name, e.g. "gemm".
        output_code_contains: Case-insensitive substring match on the generated kernel source,
            e.g. "dpas" to find kernels using that intrinsic.
        output_code_terms: Case-insensitive search terms; every term must occur in the generated
            kernel source.
        output_language: Exact match (case-insensitive), e.g. "sycl", "cuda", "triton", "cm" (C for
            Metal).
        gpu_arch_contains: Case-insensitive substring match on the GPU architecture string,
            e.g. "bmg" (matches both a plain "bmg" value and combos like "bmg,lnl").
        gpu_arch: Case-insensitive exact match on the GPU architecture string.
        max_runtime: Upper bound on runtime, in milliseconds.
        min_speedup: Lower bound on speedup ratio (on top of the always-applied >0.5x floor),
            e.g. 1.5 = 1.5x faster than native.
        max_speedup: Upper bound on speedup ratio, e.g. 1.5 = 1.5x faster than native.
        limit: Max rows to return, clamped to [1, 100]. Defaults to 20.
        offset: Zero-based row offset, for paging through results beyond `limit`.

    Returns:
        dict: A dictionary with the following keys:

            - success (bool): Whether the search completed successfully.
            - total_count (int): Total number of matching rows across all pages.
            - results (list[dict]): One compact entry per match with keys id, uuid, task_name,
              job_name, input_language, output_language, runtime (milliseconds), speedup
              (ratio, e.g. 1.5 = 1.5x faster than native), gpu_arch, timestamp, and
              output_code_preview (the code is truncated -- call ``get_kernel_from_db(id)``
              for the full source).
            - error (str | None): Error message if the search failed.
    """
    filters = {
        "output_language": output_language,
        "task_name": task_name,
        "gpu_arch_contains": gpu_arch_contains,
        "gpu_arch": gpu_arch,
        "task_name_contains": task_name_contains,
        "output_code_contains": output_code_contains,
        "output_code_terms": output_code_terms,
        "max_runtime": max_runtime,
        "min_speedup": min_speedup,
        "max_speedup": max_speedup,
    }

    if ctx is not None:
        await ctx.report_progress(progress=10, total=100)

    server_url = ""
    try:
        config = _load_config()
        server_url = str(config.get("server_url", "")).rstrip("/")

        if server_url:
            # if server_url is set, use the API on the server to query the database
            params: dict[str, Any] = {k: v for k, v in filters.items() if v is not None}
            params.update({"limit": limit, "offset": offset})
            headers = {"Authorization": f"Bearer {config['token']}"}
            async with httpx.AsyncClient(timeout=float(config.get("timeout", 60.0))) as client:
                resp = await client.get(f"{server_url}/api/kernels/search", params=params, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
            result = {
                "success": True,
                "total_count": payload.get("total_count", 0),
                "results": payload.get("results", []),
                "error": None,
            }
        else:
            import kernelfoundry.eval_pipeline.database.queries as db_queries

            _ensure_db()
            rows, total_count = db_queries.search_kernels(filters, limit=limit, offset=offset)
            result = {
                "success": True,
                "total_count": total_count,
                "results": [db_queries.kernel_to_preview_dict(row) for row in rows],
                "error": None,
            }

        if ctx is not None:
            await ctx.report_progress(progress=100, total=100)
        return result

    except Exception as e:
        logging.exception("search_kernels_in_db failed")
        if ctx is not None:
            await ctx.report_progress(progress=100, total=100)
        return {
            "success": False,
            "total_count": 0,
            "results": [],
            "error": _format_server_request_error(e, server_url),
        }


@mcp.tool()
async def get_kernel_from_db(
    kernel_id: int, include_profiler_data: bool = False, ctx: Context | None = None
) -> dict[str, Any]:
    """Fetches the full record for one kernel found via ``search_kernels_in_db``.

    Args:
        kernel_id: Database ID of the kernel (the "id" field from search_kernels_in_db results).
        include_profiler_data: If True, also include the detailed profiler timeline JSON
            blobs, which can be large. Defaults to False.

    Returns:
        dict: A dictionary with the following keys:

            - success (bool): Whether the lookup completed successfully.
                        - kernel (dict | None): Kernel metadata, output_code, eval_log, runtime_stats,
                            optimization_profile, and other compact fields, or None if no kernel with that
                            ID exists. Large prompt/answer text is omitted.
            - error (str | None): Error message if the lookup failed.
    """
    if ctx is not None:
        await ctx.report_progress(progress=10, total=100)

    server_url = ""
    try:
        config = _load_config()
        server_url = str(config.get("server_url", "")).rstrip("/")

        if server_url:
            headers = {"Authorization": f"Bearer {config['token']}"}
            async with httpx.AsyncClient(timeout=float(config.get("timeout", 60.0))) as client:
                resp = await client.get(
                    f"{server_url}/api/kernels/{kernel_id}",
                    params={"include_profiler_data": include_profiler_data},
                    headers=headers,
                )
                if resp.status_code == 404:
                    result = {"success": True, "kernel": None, "error": None}
                else:
                    resp.raise_for_status()
                    result = {"success": True, "kernel": resp.json(), "error": None}
        else:
            import kernelfoundry.eval_pipeline.database.queries as db_queries

            _ensure_db()
            kernel = db_queries.get_kernel(kernel_id)
            result = {
                "success": True,
                "kernel": db_queries.kernel_to_full_dict(kernel, include_profiler_data) if kernel is not None else None,
                "error": None,
            }

        if ctx is not None:
            await ctx.report_progress(progress=100, total=100)
        return result

    except Exception as e:
        logging.exception("get_kernel_from_db failed")
        if ctx is not None:
            await ctx.report_progress(progress=100, total=100)
        return {"success": False, "kernel": None, "error": _format_server_request_error(e, server_url)}


@mcp.tool()
async def submit_task(folder_path: str, ctx: Context) -> dict[str, Any]:
    """Submits a task for multi-iteration kernel optimization.

    Runs the full optimization loop (many LLM iterations), either locally or, if a
    ``server_url`` is configured, on the KernelFoundry server. Returns results for the
    best kernel found, ranked by composite performance score (perf_score + speedup).

    Unlike ``build_and_test``, this tool does not support internal communication and,
    when running locally, requires ``max_iters > 0`` in the task's config.

    Args:
        folder_path (str): Absolute or relative path to the folder containing the custom task.

    Returns:
        dict: A dictionary with the following keys:

            - success (bool): Whether the job completed successfully.
            - job_id (int): The ID of the submitted job.
            - eval_log (str): The evaluation log of the best kernel found.
            - runtime_stats (dict): Runtime statistics from the best kernel execution.
            - speedup (float | str): Runtime improvement of the best kernel or "N/A".
            - best_kernel_id (int | None): Database ID of the best kernel.
            - graph_url (str | None): Link to the KernelFoundry graph UI showing this
              task's full iteration history. Always share this link with the user after
              reporting results.

        Side effect:
            The best kernel's code is written back into the task folder, replacing
            the content between the ``[EVOLVE_START]`` and ``[EVOLVE_END]`` markers.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    await ctx.report_progress(progress=0, total=100)

    config = _load_config()
    if not str(config.get("server_url", "")).strip():
        return await _run_locally(folder, ctx, run_optimization=True)

    result = await _run_server_job(
        folder=folder,
        ctx=ctx,
        endpoint="/api/jobs",
        pick_kernel=lambda kernels: max(kernels, key=_kernel_performance_score),
    )
    result["best_kernel_id"] = result.pop("chosen_kernel_id")
    output_code = result.pop("_output_code", None)
    if output_code:
        _apply_output_code(folder=folder, output_code=str(output_code))
    return result


@mcp.tool()
async def build_and_test(folder_path: str, ctx: Context) -> BuildAndTestResult:
    """Builds and tests a task defined in a directory.

    If a server URL is configured, this submits the task archive to the KernelFoundry
    server and waits for results. Otherwise it evaluates the task locally.

    Note: this tool only builds and tests the current kernel, it does not run the full multi-iteration optimization loop.
    Optimization-related parameters in config.yaml are ignored.

    Returns a structured result containing success status, job ID, evaluation log,
    runtime statistics, speedup, best-kernel ID (optional), potential errors,
    and a URL to the KernelFoundry UI showing the graph of how the kernel evolved.
    If the URL is present and not None, share the link with the user after reporting results.

    Args:
        folder_path (str): Absolute or relative path to the folder containing the custom task.

    Returns:
        BuildAndTestResult: The build and test result with the following fields:

            - success (bool): Whether the job completed successfully.
            - job_id (int): The ID of the submitted job.
            - eval_log (str): The evaluation log from the job execution.
            - runtime_stats (dict): Runtime statistics from kernel execution.
            - speedup (float | str): Runtime improvement metric or "N/A" if not available.
            - error (str, optional): Error message if the job failed.
            - graph_url (str | None): Link to the KernelFoundry graph UI showing this
              task's full iteration history.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    if _internal_comm_path is not None:
        return await _build_and_test_via_internal_comm(folder_path=folder_path, ctx=ctx)

    await ctx.report_progress(progress=0, total=100)

    config = _load_config()
    if not str(config.get("server_url", "")).strip():
        return await _run_locally(folder, ctx, run_optimization=False)

    parent_uuid = _get_last_kernel_uuid(folder)
    result = await _run_server_job(
        folder=folder,
        ctx=ctx,
        endpoint="/api/validate_job",
        extra_form_data={"parent_uuid": parent_uuid} if parent_uuid else None,
        pick_kernel=lambda kernels: kernels[0],
    )
    del result["chosen_kernel_id"]
    result.pop("_output_code", None)
    return BuildAndTestResult.model_validate(result)


def main(_internal_comm_path_arg: str | None = None) -> None:
    """Start the KernelFoundry MCP server.

    Initializes logging and starts the Model Context Protocol server, which provides
    tools for building and testing kernel optimization tasks. Optionally configures
    internal communication via file-based handoff if a communication path is provided.

    Args:
        _internal_comm_path_arg (str | None): Optional path for file-based internal communication.
            If provided, the server will use this directory for handoff with another process
            instead of HTTP-based communication with a remote server. Defaults to None.
    """
    log_level = getattr(logging, "INFO", logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Optional internal communication path used for file-based handoff in build_and_test.
    global _internal_comm_path
    _internal_comm_path = _internal_comm_path_arg

    # submit_task requires a server URL and does not support internal communication,
    # so hide it when running in internal mode.
    if _internal_comm_path is not None:
        mcp.local_provider.remove_tool("submit_task")

    mcp.run()


if __name__ == "__main__":
    main()
