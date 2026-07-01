import autoroot
import os
import argparse
import sys
import uuid
from typing import Any
from pathlib import Path
import logging
import shutil
from datetime import datetime, timezone

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.algorithm.utils.kernelbench_dataset import load_kernelbench_task
from kernelfoundry.algorithm.controller import Controller, setup_logging
from kernelfoundry.algorithm.utils.kernelbench_dataset import get_kernelbench_task_id
from kernelfoundry.eval_pipeline.utils.custom_task_helper import dict_to_yaml_str
from kernelfoundry.algorithm.utils.database_log_handler import DatabaseLogHandler
from kernelfoundry.eval_pipeline.tasks.task_runner import TaskRunner
from kernelfoundry.algorithm.problem_logger import ProblemLogger
from kernelfoundry.algorithm.evaluator import Evaluator
from kernelfoundry.algorithm.schemas import Program
from kernelfoundry.algorithm.utils.validation_logs import collect_raw_logs_from_task

from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError
import hydra
from hydra.core.hydra_config import HydraConfig


def validate_task(
    task: Task,
    config: DictConfig,
    job_id: int,
    task_id: int,
    parent_uuid: str | None = None,
    return_output: bool = False,
) -> dict[str, Any]:
    db.update_job_status(job_id, "VALIDATING", started_at=datetime.now(timezone.utc))
    # setup logger
    setup_logging(config.logdir)
    # initialize task runner
    TaskRunner.init(use_queue=config.get("use_queue", True))
    # set up problem logger
    validate_logdir = os.path.join(config.logdir, "validate_" + config.task_name)
    os.makedirs(validate_logdir, exist_ok=True)
    problem_logger = ProblemLogger(0, config.get("task_name", ""), validate_logdir, 0)
    # initialize evaluator
    kernel_uuid = str(uuid.uuid4())
    evaluator = Evaluator(config, problem_logger, 0, kernel_uuid=kernel_uuid)

    # Keep execution config in sync with merged_config used by the controller path.
    task.config = OmegaConf.to_container(config, resolve=True)
    task.has_build_step = task.config.get("has_build_step", task.has_build_step)
    task.has_reference_build_step = task.config.get("has_reference_build_step", True)

    Controller.build_container_image_for_task(task=task)

    # use code between evolve tags as the code to validate
    current_evolve_code_blocks = task.blocks.get("EVOLVE")
    assert len(current_evolve_code_blocks) == 1, "Custom task must have exactly one EVOLVE block for validation"
    current_evolve_code = list(current_evolve_code_blocks.values())[0]
    new_task = task.with_blocks({"EVOLVE": current_evolve_code}, keep_test_result_reference=True)

    # RUN
    exec_result, new_task = evaluator.run(new_task)

    # add to database
    kernel = db.Kernel(
        uuid=kernel_uuid,
        task_name=config.task_name,
        job_name=config.job_name,
        parent_uuid=parent_uuid,
        input_code=next(iter(task.blocks["REFERENCE"].values())),
        output_code=current_evolve_code,
        input_language=config.prompt.get("reference_language", None),
        output_language=config.language,
        gpu_arch=(config.gpu_arch if isinstance(config.gpu_arch, str) else config.gpu_arch[0]),
        config=OmegaConf.to_container(config, resolve=True),
        task_id=task_id,
        job_id=job_id,
    )
    Program.populate_kernel_from_exec_result(kernel, exec_result)
    if db.add_ignore_errors(kernel):
        logging.info(f"Added validation result to database")

    # update job status
    db.update_job_status(job_id, "VALIDATED", finished_at=datetime.now(timezone.utc))

    if return_output:
        # collect raw logs for debugging
        raw_logs = collect_raw_logs_from_task(new_task)

        return {
            "kernel_uuid": kernel_uuid,
            "exec_result": exec_result,
            "task": new_task,
            "raw_logs": raw_logs,
            "job_id": job_id,
            "task_id": task_id,
        }


@hydra.main(version_base="1.3", config_path=f"{autoroot.root}/configs", config_name="run.yaml")
def main(config: DictConfig) -> None:
    db_handler = None
    job_id = config.get("job_id", None)
    try:
        # set logging level to INFO
        logging_level = config.get("logging_level", "INFO").upper()
        logging.basicConfig(level=logging_level, format="%(message)s")

        db.init(config)

        assert config.task is not None, "task must be specified"
        assert config.task_origin is not None, "task_origin must be specified"
        task_origin = config.task_origin

        # Check constraints for benchmark tasks
        if config.task_origin == "benchmark":
            assert config.get("validate", False), "validate must be set to true for benchmark tasks"
            assert config.get("max_iters", 0) == 0, "max_iters must be set to 0 for benchmark tasks"

        store_generated_kernels_in_db = config.get("store_generated_kernels_in_db", True)
        if store_generated_kernels_in_db:
            if job_id is None:
                job_id = db.add_job(task_origin=task_origin, status="INIT")
            if job_id is not None:
                # Add database logging handler
                db_handler = DatabaseLogHandler(job_id, level=logging.DEBUG)
                formatter = logging.Formatter("%(message)s")
                db_handler.setFormatter(formatter)
                logging.getLogger().addHandler(db_handler)
                logging.info(f"Created job with {job_id=}")

        if config.task_origin in ("KernelBench", "robust_kbench"):
            # assumes config.task is a KernelBench task_name (short version wo .py)
            task = load_kernelbench_task(
                config,
                config.task,
                from_db=(config.task_origin == "robust_kbench"),
                as_custom_task=True,
                origin=config.task_origin,
            )
            metadata = {}
        else:
            task, metadata = Task.create(config.task)

        task_config = task.config.copy()
        task.config["task_origin"] = task_origin

        task.print_info(logging.info)
        task.validate()

        # Store task in database
        if task_origin not in ["KernelBench", "robust_kbench"]:
            task_db = task.to_database_task()
            task_id = task_db.id
            if store_generated_kernels_in_db:
                if db.add_ignore_errors(task_db):
                    logging.info(f"Added task with {task_id=} to database")
        else:
            task_id = get_kernelbench_task_id(task.config["task_name"])

        if store_generated_kernels_in_db:
            db.update_job_status(job_id, status="INIT", task_id=task_id, config=task_config)

        # merge configs
        # Get command-line overrides from Hydra to ensure they have highest priority
        hydra_cfg = HydraConfig.get()
        cmdline_overrides_list = list(getattr(hydra_cfg.overrides, "task", []))
        # Remove Hydra's "+" prefix from overrides
        cmdline_overrides_list = [override.lstrip("+") for override in cmdline_overrides_list]
        cmdline_overrides = OmegaConf.from_dotlist(cmdline_overrides_list)

        # set run config as the base - disallow new keys from task.config or overrides to prevent typos
        OmegaConf.set_struct(config, True)
        # Merge order: config (defaults from run.yaml) -> task.config -> overrides -> cmdline_overrides
        merged_config = OmegaConf.merge(config, task.config, metadata.get("overrides", {}), cmdline_overrides)

        logging.info("Overrides applied:\n" + dict_to_yaml_str(metadata.get("overrides", {}), indent=2))
        logging.info("Command-line overrides:\n" + dict_to_yaml_str(cmdline_overrides, indent=2))
        logging.debug("Merged configuration:\n" + dict_to_yaml_str(OmegaConf.to_container(merged_config), indent=2))

        task.config["task_origin"] = task_origin

        # TASK VALIDATION
        if merged_config.get("validate", True):
            parent_uuid = merged_config.get("parent_uuid", None)
            validate_task(task, merged_config, job_id, task_id, parent_uuid=parent_uuid)

        # KERNEL GENERATION
        if merged_config.max_iters > 0:
            controller = Controller(config=merged_config, job_id=job_id, task_id=task_id)
            controller.run_single(task)
        else:
            db.update_job_progress(job_id, 1.0)
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
        if db_handler is not None:
            logging.getLogger().removeHandler(db_handler)
            db_handler.close()
        # Clean up task path content
        if config.get("clean_up_afterwards", False):
            task_path = Path(config.task)
            if task_path.exists() and task_path.is_dir():
                shutil.rmtree(task_path)
                logging.info(f"Cleaned up task path: {task_path}")
            else:
                task_path.unlink()
                logging.info(f"Cleaned up task path: {task_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m kernelfoundry.algorithm")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run"],
        help="Subcommand to execute. Use 'run' to start the pipeline.",
    )

    # Keep unknown args for Hydra, e.g. task=... max_iters=...
    args, hydra_args = parser.parse_known_args()
    if args.command != "run":
        parser.error("missing command: use 'run' (e.g. python -m kernelfoundry.algorithm run)")

    # Remove CLI command args so Hydra only sees its own overrides.
    sys.argv = [sys.argv[0], *hydra_args]
    main()
