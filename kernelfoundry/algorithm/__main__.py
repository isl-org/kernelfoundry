import argparse
import sys
import os
from pathlib import Path
import logging
import shutil
import json
import signal
from datetime import datetime, timezone

from dotenv import load_dotenv

# Loads DB/queue/LLM credentials from a .env file at the repository root. A no-op when there is
# no .env (e.g. an installed package with credentials set directly in the environment).
load_dotenv(os.environ.get("KERNELFOUNDRY_ENV_FILE", Path.cwd() / ".env"))

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.algorithm.utils.kernelbench_dataset import load_kernelbench_task
from kernelfoundry.algorithm.controller import Controller
from kernelfoundry.algorithm.utils.kernelbench_dataset import get_kernelbench_task_id
from kernelfoundry.eval_pipeline.utils.custom_task_helper import dict_to_yaml_str
from kernelfoundry.algorithm.utils.database_log_handler import DatabaseLogHandler
from kernelfoundry.algorithm.utils.validate_task import validate_task

from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError
import hydra
from hydra.core.hydra_config import HydraConfig

KERNELBENCH_TASK_ORIGINS = ("KernelBench", "robust_kbench")


def _configure_logging(config: DictConfig) -> None:
    logging_level = config.get("logging_level", "INFO").upper()
    logging.basicConfig(level=logging_level, format="%(message)s")


def _check_task_config(config: DictConfig) -> str:
    if config.get("task") is None and config.get("custom_task") is not None:
        config.task = config.custom_task
    assert config.task is not None, "task must be specified"
    assert config.task_origin is not None, "task_origin must be specified"
    return config.task_origin


def _setup_db_logging(task_origin: str, job_id, store_generated_kernels_in_db: bool):
    db_handler = None
    if store_generated_kernels_in_db:
        if job_id is None:
            job_id = db.add_job(task_origin=task_origin, status="INIT")
        if job_id is not None:
            db_handler = DatabaseLogHandler(job_id, level=logging.DEBUG)
            db_handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(db_handler)
            logging.info(f"Created job with {job_id=}")
    return job_id, db_handler


def _create_task(config: DictConfig, task_origin: str):
    if task_origin in KERNELBENCH_TASK_ORIGINS:
        # Assumes config.task is a KernelBench task_name (short version without .py).
        task = load_kernelbench_task(
            config,
            config.task,
            from_db=(task_origin == "robust_kbench"),
            as_custom_task=True,
            origin=task_origin,
        )
        metadata = {}
    else:
        task, metadata = Task.create(config.task)

    task_config = task.config.copy()
    task.config["task_origin"] = task_origin
    task.print_info(logging.info)
    task.validate()
    return task, metadata, task_config


def _store_task_and_get_id(task, task_origin: str, store_generated_kernels_in_db: bool):
    if task_origin not in KERNELBENCH_TASK_ORIGINS:
        task_db = task.to_database_task()
        task_id = task_db.id
        if store_generated_kernels_in_db and db.add_ignore_errors(task_db):
            logging.info(f"Added task with {task_id=} to database")
    else:
        task_id = get_kernelbench_task_id(task.config["task_name"])
    return task_id


def _merge_run_config(
    config: DictConfig,
    task_config: DictConfig,
    metadata,
    strip_experiment_override: bool,
) -> DictConfig:
    hydra_cfg = HydraConfig.get()
    cmdline_overrides_list = list(getattr(hydra_cfg.overrides, "task", []))
    if strip_experiment_override:
        cmdline_overrides_list = [
            override.lstrip("+") for override in cmdline_overrides_list if not override.startswith("experiment=")
        ]
    else:
        cmdline_overrides_list = [override.lstrip("+") for override in cmdline_overrides_list]
    cmdline_overrides = OmegaConf.from_dotlist(cmdline_overrides_list)

    OmegaConf.set_struct(config, True)
    merged_config = OmegaConf.merge(config, task_config, metadata.get("overrides", {}), cmdline_overrides)

    logging.info("Overrides applied:\n" + dict_to_yaml_str(metadata.get("overrides", {}), indent=2))
    logging.info("Command-line overrides:\n" + dict_to_yaml_str(cmdline_overrides, indent=2))
    logging.debug("Merged configuration:\n" + dict_to_yaml_str(OmegaConf.to_container(merged_config), indent=2))
    return merged_config


def _cleanup_task_path(config: DictConfig) -> None:
    if not config.get("clean_up_afterwards", False):
        return

    task_path = Path(config.task)
    if task_path.exists() and task_path.is_dir():
        shutil.rmtree(task_path)
        logging.info(f"Cleaned up task path: {task_path}")
    else:
        task_path.unlink()
        logging.info(f"Cleaned up task path: {task_path}")


def _close_db_handler(db_handler) -> None:
    if db_handler is not None:
        logging.getLogger().removeHandler(db_handler)
        db_handler.close()


def submit_task(config: DictConfig):
    """Run a kernel generation or validation task"""
    db_handler = None
    job_id = config.get("job_id", None)

    # Install signal handlers to raise KeyboardInterrupt on SIGTERM and SIGBREAK, so that the
    # finally block can clean up the database and logging handler.
    def _raise_interrupt(signum, _frame):
        raise KeyboardInterrupt(f"terminated by signal {signum}")

    for name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _raise_interrupt)
        except (OSError, ValueError):
            continue

    try:
        _configure_logging(config)

        db.init(config)

        task_origin = _check_task_config(config)

        # Check constraints for benchmark tasks
        if config.task_origin == "benchmark":
            assert config.get("validate", False), "validate must be set to true for benchmark tasks"
            assert config.get("max_iters", 0) == 0, "max_iters must be set to 0 for benchmark tasks"

        store_generated_kernels_in_db = config.get("store_generated_kernels_in_db", True)
        job_id, db_handler = _setup_db_logging(task_origin, job_id, store_generated_kernels_in_db)

        task, metadata, task_db_config = _create_task(config, task_origin)
        task_id = _store_task_and_get_id(task, task_origin, store_generated_kernels_in_db)

        if store_generated_kernels_in_db:
            db.update_job_status(job_id, status="INIT", task_id=task_id, config=task_db_config)

        merged_config = _merge_run_config(config, task.config, metadata, strip_experiment_override=True)

        task.config["task_origin"] = task_origin

        # TASK VALIDATION
        if merged_config.get("validate", True):
            parent_uuid = merged_config.get("parent_uuid", None)
            validate_task(task, merged_config, job_id, task_id, parent_uuid=parent_uuid)

        # KERNEL GENERATION
        if merged_config.max_iters > 0:

            # If saving is enabled, create the generated task folder structure
            save_best_kernel = merged_config.get("save_best_kernel", False)

            # Run task
            controller = Controller(config=merged_config, job_id=job_id, task_id=task_id)
            run_output = controller.run_single(task)

            # Save evolve blocks from the best generated kernel to disk
            if save_best_kernel:
                programs, best_uuid = run_output
                best_program = programs[best_uuid]
                best_exec_result = best_program.kernel_exec_result
                if best_exec_result is not None and best_exec_result.compiled and best_exec_result.correctness:
                    evolve_blocks = best_program.code
                    evolve_save_path = (
                        Path(merged_config.results_dir)
                        / merged_config.job_name
                        / merged_config.task_name
                        / "evolve.json"
                    )
                    evolve_save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(evolve_save_path, "w", encoding="utf-8") as f:
                        json.dump(evolve_blocks, f, indent=2, ensure_ascii=False)
                    logging.info(f"Saved evolve blocks to {evolve_save_path}")
                else:
                    logging.info(
                        "Best generated result is not valid (does not compile or is incorrect). "
                        "Skipping kernel save."
                    )
        else:
            db.update_job_progress(job_id, 1.0)
    except KeyboardInterrupt:
        # cancel job at interrupt
        logging.warning("Interrupted; marking job %s as CANCELED.", job_id)
        if db_handler is not None and job_id is not None:
            db.update_job_status(job_id, status="CANCELED")
        raise
    except Exception as e:
        if isinstance(e, ConfigKeyError):
            first_error_line = str(e).splitlines()[0]
            logging.error(
                f"Error merging configurations - invalid keys in custom task config that are not in base config: {first_error_line}. Check the documentation to use only valid keys."
            )
        else:
            logging.error(f"An error occurred while running the custom task: {e}")
        if db_handler is not None and job_id is not None:
            db.update_job_status(job_id, status="FAIL")
        raise e
    finally:
        _close_db_handler(db_handler)
        _cleanup_task_path(config)


def submit_agentic_task(config: DictConfig, env_overrides: dict[str, str] | None = None):
    """Run a task with the agentic controller."""
    db_handler = None
    job_id = config.get("job_id", None)
    try:
        _configure_logging(config)

        db.init(config)

        task_origin = _check_task_config(config)

        store_generated_kernels_in_db = config.get("store_generated_kernels_in_db", True)
        job_id, db_handler = _setup_db_logging(task_origin, job_id, store_generated_kernels_in_db)

        task, metadata, task_db_config = _create_task(config, task_origin)
        task_id = _store_task_and_get_id(task, task_origin, store_generated_kernels_in_db)

        if store_generated_kernels_in_db:
            db.update_job_status(job_id, status="INIT", task_id=task_id, config=task_db_config)

        merged_config = _merge_run_config(config, task.config, metadata, strip_experiment_override=False)

        task.config["task_origin"] = task_origin
        task.config["task_id"] = task_id

        if merged_config.max_iters > 0:
            controller = hydra.utils.instantiate(
                merged_config.controller,
                run_config=merged_config,
                job_id=job_id,
                task_id=task_id,
                env_overrides=env_overrides or None,
                _recursive_=False,
            )
            controller.run(task)

            if store_generated_kernels_in_db:
                db.update_job_status(job_id, status="COMPLETE", finished_at=datetime.now(timezone.utc))
        else:
            db.update_job_progress(job_id, 1.0)

    except Exception as e:
        if isinstance(e, ConfigKeyError):
            first_error_line = str(e).splitlines()[0]
            logging.error(
                f"Error merging configurations - invalid keys in custom task config that are not in base config: {first_error_line}. Check the documentation to use only valid keys."
            )
        else:
            logging.error(f"An error occurred while running the agentic task: {e}")
        if db_handler is not None and job_id is not None:
            db.update_job_status(job_id, status="FAIL", finished_at=datetime.now(timezone.utc))
        raise e
    finally:
        _close_db_handler(db_handler)
        _cleanup_task_path(config)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run.yaml")
def main(config: DictConfig) -> None:
    submit_task(config)


@hydra.main(version_base="1.3", config_path="../configs", config_name="run_agentic.yaml")
def agentic_main(config: DictConfig) -> None:
    submit_agentic_task(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m kernelfoundry.algorithm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Run the standard pipeline.").set_defaults(func=main)
    subparsers.add_parser("agentic", help="Run the agentic pipeline.").set_defaults(func=agentic_main)

    # Keep unknown args for Hydra, e.g. task=... max_iters=...
    args, hydra_args = parser.parse_known_args()

    # Remove CLI command args so Hydra only sees its own overrides.
    sys.argv = [sys.argv[0], *hydra_args]
    args.func()
