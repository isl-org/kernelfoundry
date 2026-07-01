"""KernelFoundry MCP server allowing to use components of the pipeline as tools"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastmcp import Context, FastMCP
from omegaconf import OmegaConf

CONFIG_PATH = Path.home() / ".config" / "kernelfoundry" / "config.yml"

TERMINAL_STATUSES = {"COMPLETE", "FAIL", "CANCELED", "VALIDATED"}
SUCCESS_STATUSES = {"COMPLETE", "VALIDATED"}

mcp = FastMCP("kernelfoundry", instructions="Tools for working with kernelfoundry task directories.")

_internal_comm_path: str | None = None


async def _build_and_test_via_internal_comm(folder_path: str, ctx: Context) -> dict[str, Any]:
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
    return output_payload


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
                zf.write(abs_path, arcname=str(rel_path))
    return buffer.getvalue()


def _append_kernel_uuids(folder: Path, kernels: list[dict[str, Any]]) -> None:
    uuids = [str(k.get("uuid")) for k in kernels if k.get("uuid")]
    if not uuids:
        return

    meta_dir = folder / ".kernelfoundry"
    kernels_path = meta_dir / "kernels.json"
    meta_dir.mkdir(parents=True, exist_ok=True)

    existing: list[str] = []
    if kernels_path.exists():
        try:
            loaded = json.loads(kernels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {kernels_path}: {e}") from e
        if not isinstance(loaded, list):
            raise TypeError(f"Expected JSON list in {kernels_path}, got {type(loaded).__name__}.")
        existing = [str(v) for v in loaded]

    existing.extend(uuids)
    kernels_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _get_last_kernel_uuid(folder: Path) -> str | None:
    kernels_path = folder / ".kernelfoundry" / "kernels.json"
    if not kernels_path.exists():
        return None
    loaded = json.loads(kernels_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or len(loaded) == 0:
        return None
    return loaded[-1]


async def _build_and_test_locally(folder: Path, ctx: Context) -> dict[str, Any]:
    import autoroot
    from hydra import compose, initialize_config_dir
    import kernelfoundry.eval_pipeline.database as db
    from kernelfoundry.algorithm.__main__ import validate_task
    from kernelfoundry.eval_pipeline.task import Task
    from kernelfoundry.algorithm.utils.database_log_handler import DatabaseLogHandler
    import logging as _logging

    source_task, metadata = Task.create(folder)
    evolve_block = source_task.blocks.get("EVOLVE")
    if not evolve_block:
        raise ValueError(f"Task at {folder} is missing required EVOLVE block.")

    # Load base config using Hydra's config composition
    config_dir = Path(autoroot.root) / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        base_config = compose(config_name="run")

    # Merge: base_config -> source_task.config
    merged_config = OmegaConf.merge(base_config, source_task.config)

    # Keep local validation behavior aligned with the CLI path.
    task_origin = source_task.config.get("task_origin", "custom")
    source_task.config["task_origin"] = task_origin

    await ctx.report_progress(progress=20, total=100)

    job_id: int | None = 0
    task_id: str | int | None = None

    db.init(merged_config)
    task_db = source_task.to_database_task()
    task_id = task_db.id
    db.add_ignore_errors(task_db)

    task_config = source_task.config.copy()
    job_id = db.add_job(task_origin=task_origin, status="INIT")
    if job_id is not None:
        db.update_job_status(job_id, status="INIT", task_id=task_id, config=task_config)
    else:
        job_id = 0

    try:
        await ctx.report_progress(progress=40, total=100)
        parent_uuid = _get_last_kernel_uuid(folder)

        db_handler = DatabaseLogHandler(job_id, level=_logging.DEBUG)
        db_handler.setFormatter(_logging.Formatter("%(message)s"))
        _logging.getLogger().addHandler(db_handler)
        try:
            validation = await asyncio.to_thread(
                validate_task,
                source_task,
                merged_config,
                job_id,
                task_id,
                parent_uuid=parent_uuid,
                return_output=True,
            )
        finally:
            _logging.getLogger().removeHandler(db_handler)
            db_handler.close()
        exec_result = validation["exec_result"]
        kernel_uuid = validation.get("kernel_uuid")
        if kernel_uuid:
            _append_kernel_uuids(folder=folder, kernels=[{"uuid": kernel_uuid}])

        eval_log = exec_result.eval_log
        if not eval_log:
            eval_log = validation.get("raw_logs") or "Local evaluation produced no eval_log and no raw logs."

        await ctx.report_progress(progress=100, total=100)
        return {
            # Local success means the validation job executed without internal server/job errors.
            "success": True,
            "job_id": int(job_id or 0),
            "eval_log": eval_log,
            "runtime_stats": exec_result.runtime_stats if isinstance(exec_result.runtime_stats, dict) else {},
            "speedup": exec_result.runtime_improvement if exec_result.runtime_improvement is not None else "N/A",
        }
    except Exception as e:
        raw_log = f"Local validation failed with exception:\n{traceback.format_exc()}"
        await ctx.report_progress(progress=100, total=100)
        return {
            "success": False,
            "job_id": int(job_id or 0),
            "eval_log": raw_log,
            "runtime_stats": {},
            "speedup": "N/A",
            "error": str(e),
        }


@mcp.tool()
async def build_and_test(folder_path: str, ctx: Context) -> dict[str, Any]:
    """Builds and tests a task defined in a directory.

    If a server URL is configured, this submits the task archive to the KernelFoundry
    server and waits for results. Otherwise it evaluates the task locally.

    Args:
        folder_path (str): Absolute or relative path to the folder containing the custom task.

    Returns:
        dict: A dictionary with the following keys:

            - success (bool): Whether the job completed successfully.
            - job_id (int): The ID of the submitted job.
            - eval_log (str): The evaluation log from the job execution.
            - runtime_stats (dict): Runtime statistics from kernel execution.
            - speedup (float | str): Runtime improvement metric or "N/A" if not available.
            - error (str, optional): Error message if the job failed.
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
    server_url = str(config.get("server_url", "")).rstrip("/")
    if not server_url:
        return await _build_and_test_locally(folder=folder, ctx=ctx)

    user = config["user"]
    token = config["token"]
    timeout = float(config.get("timeout", 1200.0))
    poll_interval = float(config.get("poll_interval", 5.0))
    max_wait = float(config.get("max_wait", 3600.0))

    archive_bytes = _zip_folder(folder)

    await ctx.report_progress(progress=5, total=100)

    filename = f"{folder.name}.zip"
    headers = {"Authorization": f"Bearer {token}"}
    parent_uuid = _get_last_kernel_uuid(folder)

    # return values
    eval_log: str | None = None
    runtime_stats: dict[str, Any] = {}
    speedup: str | float = "N/A"

    async with httpx.AsyncClient(timeout=timeout) as client:
        # 1. Submit the job to /api/validate_job
        submit_resp = await client.post(
            f"{server_url}/api/validate_job",
            headers=headers,
            files={"file": (filename, archive_bytes, "application/zip")},
            data={"idsid": user, "parent_uuid": parent_uuid} if parent_uuid else {"idsid": user},
        )
        try:
            submission = submit_resp.json()
        except ValueError:
            submission = {"raw_response": submit_resp.text}

        if submit_resp.status_code >= 400:
            raise RuntimeError(f"validate_job request failed ({submit_resp.status_code}): {submission}")

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
        if kernels:
            # 4. Get the eval log of the first kernel (if any)
            eval_log: str | None = None
            first_kernel_id = kernels[0].get("id") if kernels else None
            if first_kernel_id is not None:
                log_resp = await client.get(
                    f"{server_url}/api/kernels/{first_kernel_id}/eval_log",
                    headers=headers,
                )
                if log_resp.status_code < 400:
                    eval_log = log_resp.json().get("eval_log")
                    runtime_stats = log_resp.json().get("runtime_stats", {})
                    speedup = log_resp.json().get("speedup", "N/A")
        else:
            # without kernel we don't have an eval log -> Fetch the raw logs instead
            eval_log = []
            logs_resp = await client.get(
                f"{server_url}/api/jobs/logs",
                headers=headers,
                params={"job_id": job_id},
            )
            logs_resp.raise_for_status()
            logs = logs_resp.json().get("logs", [])
            for l in logs:
                raw_log = l.get("raw_log")
                if raw_log:
                    eval_log.append(raw_log)
            if eval_log:
                eval_log = "\n".join(eval_log) if eval_log else None
            else:
                eval_log = None

        if eval_log is None:
            eval_log = "Server error: Failed to retrieve evaluation log."

    await ctx.report_progress(progress=100, total=100)

    return {
        "success": final_status in SUCCESS_STATUSES,
        "job_id": job_id,
        "eval_log": eval_log,
        "runtime_stats": runtime_stats,
        "speedup": speedup,
    }


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
    mcp.run()


if __name__ == "__main__":
    main()
