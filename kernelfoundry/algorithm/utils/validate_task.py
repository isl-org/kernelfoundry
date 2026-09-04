import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.algorithm.controller import Controller, setup_logging
from kernelfoundry.algorithm.evaluator import Evaluator
from kernelfoundry.algorithm.problem_logger import ProblemLogger
from kernelfoundry.algorithm.schemas import Program
from kernelfoundry.algorithm.utils.validation_logs import collect_raw_logs_from_task
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.eval_pipeline.tasks.task_runner import TaskRunner
from kernelfoundry.eval_pipeline.utils.custom_task_helper import blocks_to_str
from omegaconf import DictConfig, OmegaConf


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
    TaskRunner.init(use_queue=config.get("use_queue", True), gpu_arch=config.gpu_arch)
    # set up problem logger
    validate_logdir = os.path.join(config.logdir, "validate_" + config.task_name)
    os.makedirs(validate_logdir, exist_ok=True)
    problem_logger = ProblemLogger(0, config.get("task_name", ""), validate_logdir, 0)
    # initialize evaluator
    kernel_uuid = str(uuid.uuid4())
    evaluator = Evaluator(config, problem_logger, 0, kernel_uuid=kernel_uuid)

    # Keep execution config in sync with merged_config used by the controller path.
    task.apply_config(config)

    Controller.build_container_image_for_task(task=task)

    # use code between evolve tags as the code to validate
    current_evolve_code_blocks = task.blocks.get("EVOLVE", {})
    assert len(current_evolve_code_blocks) >= 1, "Custom task must have at least one EVOLVE block for validation"
    new_task = task.with_blocks({"EVOLVE": current_evolve_code_blocks}, keep_test_result_reference=True)
    current_evolve_code = blocks_to_str(current_evolve_code_blocks)

    # RUN
    exec_result, new_task = evaluator.run(new_task)

    if config.get("save_exec_result", False):
        save_exec_result_path = Path(config.results_dir) / config.job_name / config.task_name / "validate.json"
        save_exec_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_exec_result_path, "w", encoding="utf-8") as f:
            json.dump(exec_result.to_dict(), f, indent=2)
        logging.info(f"Saved execution result to {save_exec_result_path}")

    # add to database
    kernel = db.Kernel(
        uuid=kernel_uuid,
        task_name=config.task_name,
        job_name=config.job_name,
        parent_uuid=parent_uuid,
        input_code=blocks_to_str(task.blocks["REFERENCE"]) if task.blocks.get("REFERENCE") else "",
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
        logging.info("Added validation result to database")

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
