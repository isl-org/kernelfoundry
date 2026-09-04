"""Controller for kernel generation and evaluation using LLMs and prompt evolution."""

import os
from typing import override, Optional
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import numpy as np
import time
from collections import defaultdict
import logging
from datetime import datetime, timezone
import shutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from kernelfoundry.algorithm.schemas import EvalResult, Program
from kernelfoundry.algorithm.utils.score import select_best_solution
from kernelfoundry.algorithm.utils.load_from_db import load_best_kernel
from kernelfoundry.algorithm.problem_logger import ProblemLogger
import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.eval_pipeline.tasks.task_runner import TaskRunner
from kernelfoundry.eval_pipeline.utils.custom_task_helper import blocks_to_str
from kernelfoundry.algorithm.evaluator import Evaluator
from kernelfoundry.algorithm.prompts.prompt_evolution_integration import PromptEvolutionMixin
from kernelfoundry.algorithm.answer_processor import AnswerProcessor
from kernelfoundry.algorithm.prompts.feedback_llm import FeedbackHelper
from kernelfoundry.algorithm.prompts.prompt_constructor import PromptConstructor, get_system_prompt
from kernelfoundry.algorithm.prompts.optimization_aware import (
    build_exploration_prompt,
    get_optimization_guidance_for_parent,
)
from kernelfoundry.algorithm.utils.token_usage import zero_token_usage


def setup_logging(logdir: str, logging_level: int | str | None = None):
    """Setup logging including raw mode that stores logs in database.

    Args:
        logdir (str): Directory where controller.log file will be stored.
        logging_level (int | str | None, optional): Logging level to use. Defaults to None.
    """
    if logging_level is None:
        logging_level = logging.getLogger().level

    def raw_root(message, *args, **kwargs):
        logging.log(RAW, message, *args, **kwargs)

    # Define the custom level
    RAW = 25  # between INFO (20) and WARNING (30)
    logging.addLevelName(RAW, "RAW")
    logging.raw = raw_root

    # Set up logging to console and file
    log_file = Path(logdir) / "controller.log"
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging_level)

    # Create formatter
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging_level)


class Controller(PromptEvolutionMixin):
    """Controller class that steers the kernel generation and evaluation process."""

    def __init__(
        self,
        config,
        job_id: int | None,
        task_id: int | None,
        logging_level: int | str | None = None,
        resource_allocator=None,
    ):
        """The controller steers the kernel generation and evaluation process for a Task problem.

        Args:
            config: configuration for the controller
            job_id (int | None): ID of the associated job in the database, if database logging is enabled
            task_id (int | None): ID of the associated task in the database, if database logging is enabled
            logging_level (int | str | None): logging level to use for logging output
        """
        self.config = config
        self.job_id = job_id
        self.task_id = task_id
        setup_logging(self.config.logdir, logging_level=logging_level)
        # track failures to avoid running 100 trials with a broken setup
        self.failures = 0
        self.max_failures = config.get("max_failures", 5)
        # evolve_mode: if >1 branches
        self.evolve_mode = config.get("branches_per_iteration", 1) > 1

        TaskRunner.init(use_queue=config.get("use_queue", True), gpu_arch=config.gpu_arch)

        self.resource_allocator = resource_allocator
        self.answer_processor = AnswerProcessor(
            language=config.language,
            diff_format=config.prompt.diff_format,
            postprocess_code=config.postprocess_code,
            postprocess_code_config=config.postprocess_code_config,
        )

        self.reference_language = config.prompt.get("reference_language", "Pytorch")

        self.llm_server = hydra.utils.instantiate(config.inference)
        print(f"Initiated LLM with config {config.inference}")
        self._token_usage_totals = zero_token_usage()

        self.feedback_helper = FeedbackHelper(
            use_feedback_llm=config.use_feedback_llm,
            language=config.language,
            server_config=config.feedback_llm_config,
            use_docs_via_keywords=config.use_docs_via_keywords,
        )
        print(f"Initiated feedback LLM with config: {config.feedback_llm_config}")

        self.prompt_constructor = PromptConstructor(
            config.language,
            config.gpu_arch,
            config.prompt,
            reference_language=self.reference_language,
            mode=config.mode,
            use_feedback_llm=config.use_feedback_llm,
        )
        print(f"Initiated prompt constructor: language: {config.language}, use diff: {config.prompt.diff_format}")

        if self.evolve_mode:
            self.program_database = hydra.utils.instantiate(config.database)
            print("Initiated program database with config:", config.database)
        else:
            self.program_database = None

        # create logdir
        self.config.logdir = os.path.join(self.config.logdir, self.config.job_name)
        os.makedirs(self.config.logdir, exist_ok=True)

        # Initialize prompt evolution system (from PromptEvolutionMixin)
        self.setup_prompt_evolution()

    def sample_evolve_programs(self):
        """Sample parent code, inspirations, top program from the database"""
        parent, inspirations = self.program_database.sample()
        # Get parent artifacts if available
        parent_console_log = parent.get_artifact()

        if self.config.use_feedback_llm:
            # Get feedback from feedback LLM -> not saving the feedback in parent because we want varying feedback!
            parent_feedback = self.feedback_helper.get_feedback(parent.code, parent_console_log)
            assert len(parent_feedback) == 1, "Feedback LLM should return exactly one feedback message"
            parent_artifacts = parent_feedback[0]
            # # Other option: input console log and feedback
            # parent_artifacts = {"Console log": parent_console_log}
            # parent_artifacts["Feedback"] = parent_feedback[0]
        else:
            parent_artifacts = parent_console_log
        parent.feedback = parent_artifacts

        # Get island-specific programs for context
        parent_island = parent.metadata.get("island")
        if parent_island is None:
            parent_island = self.program_database.current_island

        island_programs = [
            self.program_database.programs[pid]
            for pid in self.program_database.islands[parent_island]
            if pid in self.program_database.programs and self.program_database.programs[pid].metrics
        ]

        # Sort by metrics for top programs (if they have combined_score)
        if island_programs:
            try:
                island_programs.sort(key=lambda p: p.metrics.get("combined_score", 0), reverse=True)
            except (KeyError, TypeError):
                # If sorting fails, keep original order
                pass

        # Use config values for limits instead of hardcoding
        num_top = getattr(self.program_database, "num_top_programs", None)
        num_diverse = getattr(self.program_database, "num_diverse_programs", None)

        # Ensure we have integer values, not None
        if num_top is None:
            print("WARNING: num_top_programs is None, using default 1")
            num_top = 1
        if num_diverse is None:
            print("WARNING: num_diverse_programs is None, using default 0")
            num_diverse = 0

        # Extra safety: ensure they're actually integers
        num_top = int(num_top)
        num_diverse = int(num_diverse)

        island_top_programs = island_programs[: num_top + num_diverse] if island_programs else []

        # Ensure we have at least one program - use parent as fallback
        if not island_top_programs:
            island_top_programs = [parent]

        # final parts of the prompt:
        sampled_program_prompts = {
            "top_program": island_top_programs[0],
            "last_program": parent,
            "inspirations": inspirations,
        }
        return sampled_program_prompts, parent

    def evolve_prompt_and_inference(
        self, problem_name: str, ref_arch_src: str, trial: int, task: Optional["Task"] = None
    ) -> (str, str, Program, str, str):
        """
        Run a single iteration in a worker process
        Args:
            problem_name: Operation name
            ref_arch_src: Reference source code
            trial: Trial number (to pass to LLM server)
            task: Custom task object that can provide the initial implementation

        Returns:
            llm_response: Generated code from LLM
            prompt: Prompt that was used for generation
            parent: Parent program object
            model_used: LLM model that was used
        """
        iteration_start = time.time()

        # sample parent and inspirations
        if not self.program_database.is_empty():
            sampled_program_prompts, parent = self.sample_evolve_programs()
            parent.task = task  # attach the custom task here
        elif task is not None:
            # A custom task with an initial implementation can serve as parent
            parent = self.create_program0(task)
            sampled_program_prompts = dict(last_program=parent)
        else:
            sampled_program_prompts, parent = {}, None

        # Determine target optimization profile before prompt construction so RAG can use it.
        parent_is_correct = parent.kernel_exec_result.correctness if parent and parent.kernel_exec_result else False
        vector_parent = parent if parent_is_correct else None
        target_optimization_profile, optimization_strategy = self._get_target_optimization_profile(vector_parent)

        # Sample evolved prompt content early (for template injection)
        evolvable_content = None
        prompt_evolution_session_id = None
        if self.prompt_evolution_enabled and self.meta_prompting_enabled:
            prompt_program, session_id = self.sample_evolved_prompt()
            if prompt_program is not None:
                evolvable_content = prompt_program.template_overrides
                prompt_evolution_session_id = session_id
                print(f"Sampled evolved prompt (id={prompt_program.id[:8]}, gen={prompt_program.generation})")

        # Build prompt with evolved content
        prompt = self.prompt_constructor(
            reference_src=ref_arch_src,
            problem_name=problem_name,
            evolvable_content=evolvable_content,
            target_optimization_profile=target_optimization_profile,
            **sampled_program_prompts,
        )

        # Apply additional optimization-aware prompting if enabled (appends optimization guidance)
        if getattr(self.config, "use_optimization_aware_prompting", False):
            prompt, additional_session_id = self._apply_optimization_aware_prompting(
                prompt,
                parent,
                target_optimization_profile=target_optimization_profile,
                strategy=optimization_strategy,
            )
            # Combine session IDs for meta-prompting
            if additional_session_id and prompt_evolution_session_id:
                prompt_evolution_session_id = f"{prompt_evolution_session_id}+{additional_session_id}"
            elif additional_session_id:
                prompt_evolution_session_id = additional_session_id

        # Generate code modification (sync wrapper for async)
        messages = [
            {"role": "system", "content": get_system_prompt(self.config.language)},
            {"role": "user", "content": prompt},
        ]
        if self.resource_allocator is None:
            llm_response, metadata = self.llm_server(
                messages=messages,
                trial=trial,
            )
        else:
            with self.resource_allocator.reserve_server() as server_id:
                llm_response, metadata = self.llm_server(
                    messages=messages,
                    trial=trial,
                    server_id=server_id,
                )
        assert (
            len(llm_response) == 1
        ), f"LLM must return one answer, received {len(llm_response)} from {metadata['model']}"
        llm_response = llm_response[0]

        # add token usage to total
        self._token_usage_totals += {
            "input_tokens": metadata["input_tokens"],
            "output_tokens": metadata["output_tokens"],
        }

        print("Time for prompt sampling and inference:", time.time() - iteration_start)
        return llm_response, prompt, parent, metadata["model"], prompt_evolution_session_id

    def _apply_optimization_aware_prompting(
        self,
        prompt: str,
        parent: Optional[Program],
        target_optimization_profile: Optional[dict] = None,
        strategy: Optional[str] = None,
    ) -> tuple:
        """
        Apply optimization-aware prompting enhancements, including evolved prompt genes.

        Args:
            prompt: Base prompt
            parent: Parent program (if available)

        Returns:
            Tuple of (enhanced_prompt, session_id) where session_id is for prompt evolution tracking
        """
        session_id = None

        enable_esimd = getattr(self.config, "enable_esimd_exploration", False)
        if not target_optimization_profile:
            return prompt, session_id

        effective_strategy = strategy or "diversify"

        # For explicit SIMD upgrades, prepend taxonomy before required optimizations.
        if effective_strategy == "esimd_upgrade":
            from kernelfoundry.algorithm.prompts.optimization_aware import get_optimization_taxonomy_prompt

            if self.config.language.lower() == "cuda":
                explicit_simd_dimensions = ["tensor_core"]
            else:
                explicit_simd_dimensions = ["esimd"]

            explicit_simd_taxonomy = get_optimization_taxonomy_prompt(
                include_antipatterns=False,
                include_performance_hints=True,
                dimensions=explicit_simd_dimensions,
                include_esimd=True,
                backend=self.config.language,
            )
            prompt += "\n" + explicit_simd_taxonomy
            if self.config.language.lower() == "cuda":
                print(">>> Tensor Core taxonomy included in prompt (requires #include <mma.h>)")
            else:
                print(">>> ESIMD taxonomy included in prompt (requires #include <sycl/ext/intel/esimd.hpp>)")

        # Use a single selected target profile for both parent-guided and exploration-guided prompting.
        prompt = build_exploration_prompt(
            prompt,
            [target_optimization_profile],
            include_esimd=enable_esimd,
            backend=self.config.language,
        )
        print(
            "Applied optimization-aware target profile: "
            f"memory={target_optimization_profile.get('memory_opt', 0)}, "
            f"compute={target_optimization_profile.get('compute_opt', 0)}, "
            f"parallelism={target_optimization_profile.get('parallelism_opt', 0)}, "
            f"esimd={target_optimization_profile.get('esimd_opt', 0)} "
            f"(strategy: {effective_strategy})"
        )

        if self.prompt_evolution_enabled:
            prompt, session_id = self.apply_evolved_prompting(
                prompt, parent, strategy=effective_strategy, include_esimd=enable_esimd
            )
            if session_id:
                print(f"Applied evolved prompt genes (session: {session_id[:8]}...)")

        return prompt, session_id

    def _get_target_optimization_profile(self, parent: Optional[Program]) -> tuple[Optional[dict], Optional[str]]:
        """Select the optimization profile (vector) to explore next.

        Returns:
            Tuple of (target_profile, strategy). target_profile contains
            memory_opt/compute_opt/parallelism_opt/esimd_opt coordinates.
        """
        exploration_strategy = getattr(self.config, "exploration_strategy", "mutate")
        use_exploration_prompts = getattr(self.config, "use_exploration_prompts", True)
        enable_esimd = getattr(self.config, "enable_esimd_exploration", False)
        guidance_exploration_rate = float(getattr(self.config, "guidance_exploration_rate", 0.0))

        # Clamp into [0, 1] for robustness.
        guidance_exploration_rate = min(1.0, max(0.0, guidance_exploration_rate))

        underexplored = self.program_database.get_underexplored_regions(n=self.config.branches_per_iteration)

        # Optional override: use an underexplored region even when a parent exists.
        if parent and underexplored and use_exploration_prompts and np.random.random() < guidance_exploration_rate:
            target = underexplored[np.random.randint(len(underexplored))]
            print(f"Selected underexplored target profile override (rate={guidance_exploration_rate:.2f})")
            return target, "diversify"

        if parent:
            parent_profile = self.program_database.get_parent_optimization_profile(parent)
            if parent_profile:
                strategy = self._select_evolution_strategy(parent_profile, exploration_strategy, enable_esimd)
                target = get_optimization_guidance_for_parent(
                    parent_profile,
                    strategy=strategy,
                    include_esimd=enable_esimd,
                )
                return target, strategy

        if underexplored and use_exploration_prompts:
            target = underexplored[np.random.randint(len(underexplored))]
            return target, "diversify"

        return None, None

    def _select_evolution_strategy(
        self,
        parent_profile: dict,
        base_strategy: str,
        enable_esimd: bool = False,
    ) -> str:
        """
        Select the best evolution strategy based on parent profile and exploration state.

        This implements intelligent strategy selection that considers:
        - Current ESIMD level (if low, may suggest esimd_upgrade)
        - Overall optimization levels
        - Randomization for exploration diversity

        Args:
            parent_profile: Parent's optimization profile dict
            base_strategy: Default strategy from config
            enable_esimd: Whether ESIMD exploration is enabled

        Returns:
            Selected strategy string
        """
        import random

        esimd_level = parent_profile.get("esimd_opt", 0)
        memory_level = parent_profile.get("memory_opt", 0)
        compute_level = parent_profile.get("compute_opt", 0)
        parallelism_level = parent_profile.get("parallelism_opt", 0)

        # Calculate total non-ESIMD optimization level
        non_esimd_total = memory_level + compute_level + parallelism_level

        # Intelligent strategy selection for ESIMD exploration
        if enable_esimd:
            # If kernel has any optimization but no ESIMD, consider esimd_upgrade
            # Lower threshold to encourage more ESIMD exploration
            if esimd_level == 0:
                # Higher chance for well-optimized kernels
                if non_esimd_total >= 6:  # Very well optimized
                    if random.random() < 0.6:  # 60% chance
                        return "esimd_upgrade"
                elif non_esimd_total >= 3:  # Moderately optimized
                    if random.random() < 0.4:  # 40% chance
                        return "esimd_upgrade"
                elif non_esimd_total >= 1:  # Some optimization
                    if random.random() < 0.2:  # 20% chance to explore early
                        return "esimd_upgrade"

            # If already using ESIMD but at low level, chance to upgrade
            elif esimd_level in [1, 2]:
                if random.random() < 0.35:  # 35% chance to push higher
                    return "esimd_upgrade"

        # Otherwise use the base strategy
        return base_strategy

    def create_program0(self, task: Task) -> Program:
        """Create initial Program object for a Task.

        Args:
            task (Task): Task instance containing problem details
        Returns:
            Program: initial Program object
        """
        code = task.blocks["EVOLVE"]
        program0 = Program(id="", is_program0=True, code=code, language=self.config.language, task=task)
        return program0

    def standard_prompt_and_inference(
        self, parent_program: Program, task_name: str, ref_arch_src: str, problem_logger: ProblemLogger
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Construct prompt and do inference with the LLM server
        Args:
            parent_program: Program object of the parent program
            # problem_logger: Helper object that saves and loads files for the current job
            task_name: op name of KernelBench problem
            ref_arch_src: Pytorch code that we want to kernelize

        Returns:
            tuple[list[str], list[str], list[str]]: list of LLM answers, list of prompts used, list of models used
        """
        # prompt construction
        # feedback_helper returns a) None if first trial, b) console output if no feedback LLM, c) list of feedback
        parent_code, eval_feedback = self.feedback_helper.load_parent_and_get_feedback(
            problem_logger, parent_program=parent_program
        )

        # make prompt from console output/feedback, prior implementation and reference code
        prompt_list = []  # multiple prompts if feedback LLM was used with num_completions>0
        for feedback in eval_feedback:
            # attach feedback to parent
            parent_program.feedback = feedback
            # construct prompt
            prompt = self.prompt_constructor(
                ref_arch_src,
                task_name,
                last_program=parent_program,
                second_ref_code=None,
            )
            prompt_list.append(prompt)

        custom_kernel_out_list = []

        # inference -> make new list to ensure that we have the same number of prompts as LLM outputs
        prompt_list_new, model_list = [], []
        for prompt in prompt_list:
            messages = [
                {"role": "system", "content": get_system_prompt(self.config.language)},
                {"role": "user", "content": prompt},
            ]
            trial = problem_logger.trial  # trial to pass to the LLM server (for warmstart)
            if self.resource_allocator is None:
                program_list, metadata = self.llm_server(
                    messages=messages,
                    trial=trial,
                )
            else:
                with self.resource_allocator.reserve_server() as server_id:
                    program_list, metadata = self.llm_server(
                        messages=messages,
                        trial=trial,
                        server_id=server_id,
                    )
            prompt_list_new.extend([prompt for _ in range(len(program_list))])
            model_list.extend([metadata["model"]] * len(program_list))
            # token_usage already reflects the full request (including all completions);
            # add it once to avoid double counting when num_completions > 1.
            self._token_usage_totals += {
                "input_tokens": metadata["input_tokens"],
                "output_tokens": metadata["output_tokens"],
            }
            custom_kernel_out_list.extend(program_list)

        return custom_kernel_out_list, prompt_list_new, model_list

    @override
    def run_single(self, task: Task) -> tuple[dict[str, Program], str]:
        """Generate and evaluate kernels for a Task problem and repeat until unit tests pass or the max
        number of iterations is reached.

        Args:
            task (Task): Task instance containing problem details

        Returns:
            Returns a tuple containing:
                dict[str, Program]: Dictionary mapping uuid their Program instances
                str: uuid of the best Program found
        """
        try:
            return self._run_single(task, self.task_id, self.job_id)
        except Exception as e:
            logging.error(f"Job {self.job_id} failed with error: {e}", exc_info=True)
            raise

    @staticmethod
    def build_container_image_for_task(task: Task) -> Task:
        """Build the container image for the given Task if use_container is True and update the config.

        Args:
            task (Task): Task instance containing problem details. Note that this function updates
                the config of the custom task in-place and returns the same custom task for convenience.
        Returns:
            Task: Modified custom task that contains the image ids in the
                config for each language and gpu arch combination
        """
        if task.config.get("use_container", False):
            if task.config.get("container_image") is None:
                # The task does not specify any image to use
                # -> assume the task has a dockerfile or we use a default image
                build_image_ans = TaskRunner.build_image(task)
                # Extract image IDs and store as {language: {gpu_arch: image_id}} in config
                task.config["container_image"] = {
                    lang: {
                        arch: br.result.output_data["image"]
                        for arch, br in arch_map.items()
                        if br is not None and br.result is not None and br.result.output_data is not None
                    }
                    for lang, arch_map in build_image_ans.items()
                }
            logging.info(f"Using container image(s): {task.config.get('container_image')}")
        return task

    def _run_single(self, task: Task, task_id: str, job_id: int | None) -> tuple[dict[str, Program], str]:
        tic_all = time.time()
        config = self.config

        job_name = config.get("job_name", "")
        task_name = config.get("task_name", "")
        level = config.get("level", 99)
        logging.info(f"========= Starting {job_name=}, {task_name=}) ========")

        # attach the config for this job to the custom task after saving to the database
        # The task object is passed to the workers which get access to the config for the job this way
        task.apply_config(config)

        if self.evolve_mode:
            # workaround for to ensure that the program database is problem-specific - reinitiate for every new problem
            self.program_database.setup(
                config.language, task_name, task_id=task_id, gpu_arch=config.gpu_arch, output_dir=config.logdir
            )

        reference = blocks_to_str(task.blocks["REFERENCE"]) if task.blocks.get("REFERENCE") else ""
        if not reference and config.get("test_reference", True):
            logging.warning(
                "No REFERENCE block found in task, but test_reference=True. "
                "The LLM will receive no reference code in its prompt."
            )

        program0 = self.create_program0(task)
        # This is the raw output of the reference for the same task used to compute the runtime improvement
        # if there is no baseline data
        reference_test_result, build_result_reference = None, None
        programs = {program0.id: program0}
        results = defaultdict(list)  # for logging to the results.json file

        parent = program0  # Initialize to program0 for first iteration

        # Update Job status to RUN and set started_at timestamp
        if job_id is not None:
            db.update_job_status(job_id, "RUN", started_at=datetime.now(timezone.utc))

        # Build the container images before starting the main loop
        task = self.build_container_image_for_task(task)

        # Start main iteration loop
        for trial in range(config.max_iters):
            problem_logger = ProblemLogger(level, task_name, config.logdir, trial)
            logging.info(f"--------- Trial {trial}---------")
            tic_trial = tic = time.time()
            # 2) QUERY LLM: Query server with constructed prompt
            if trial == 0 and (config.start_from_best or config.kernels_iter_0_path is not None):
                # Loading kernel from prior job (best / best from specific job / specific path)
                if config.start_from_best:
                    generated_kernel, eval_log = load_best_kernel(task_name, config.language)
                elif "best-" in config.kernels_iter_0_path:
                    job = (
                        config.kernels_iter_0_path.split("best-")[-1] if config.kernels_iter_0_path != "best" else None
                    )  # Start with best kernel from prior job
                    generated_kernel, eval_log = load_best_kernel(task_name, config.language, job_name=job)
                else:
                    generated_kernel, eval_log = problem_logger.load_kernel_from_other_run(config.kernels_iter_0_path)
                if generated_kernel is None:
                    break
                # Option 1: re-evaluate the kernel (ignore eval_log, evaluate again)
                custom_kernel_out_list = [generated_kernel]
                prompt_list, model_list = ["No prompt - kernel taken from prior job!"], ["prior_generation"]
                child_list = [
                    self.answer_processor(llm_response, trial, parent) for llm_response in custom_kernel_out_list
                ]
                prompt_evolution_session_ids = [None]  # No evolution for prior job
                # # Option 2: directly continue to next iteration
                # problem_logger.save_gen_kernel(generated_kernel)
                # problem_logger.save_stdout(eval_log)
                # continue
            elif self.evolve_mode:
                custom_kernel_out_list, prompt_list, child_list, model_list = [], [], [], []
                prompt_evolution_session_ids = []  # Track session IDs for prompt evolution

                def task_function():
                    return self.evolve_prompt_and_inference(task_name, reference, trial=trial, task=task)

                max_workers = min(config.branches_per_iteration, os.cpu_count())
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(task_function) for _ in range(config.branches_per_iteration)]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            llm_response, prompt, parent, model_used, session_id = future.result()
                            parent.task = program0.task  # set custom task attribute
                            custom_kernel_out_list.append(llm_response)
                            prompt_list.append(prompt)
                            model_list.append(model_used)
                            prompt_evolution_session_ids.append(session_id)
                            child_list.append(self.answer_processor(llm_response, trial, parent))
                        except Exception as exc:
                            logging.error(f"LLM inference generated an exception: {exc}", exc_info=True)
                            prompt_evolution_session_ids.append(None)  # Keep list aligned
                assert len(prompt_list) > 0 and len(custom_kernel_out_list) > 0, "too many inference fails!"
            else:
                custom_kernel_out_list, prompt_list, model_list = self.standard_prompt_and_inference(
                    parent, task_name, reference, problem_logger
                )
                child_list = [
                    self.answer_processor(llm_response, trial, parent) for llm_response in custom_kernel_out_list
                ]
                prompt_evolution_session_ids = [None] * len(custom_kernel_out_list)  # No evolution for standard mode

            # update token usage for job in database
            if job_id is not None:
                db.update_job_token_usage(
                    job_id,
                    input_tokens=self._token_usage_totals.input_tokens,
                    output_tokens=self._token_usage_totals.output_tokens,
                )

            problem_logger.log_prompt_list(prompt_list)
            logging.info(f"Time for generating all kernels {time.time() - tic}")

            # 3) EVALUATION: evaluate every generated kernel
            logging.info(f"Evaluating {len(custom_kernel_out_list)} solutions")
            tic = time.time()

            if reference_test_result is not None:
                # Attach the reference test result to the task of each child
                # The reference test result is used as a fallback to compute the runtime improvement
                for child in child_list:
                    child.task.test_result_reference = reference_test_result
                    child.task.build_result_reference = build_result_reference  # add build if available

            eval_results, eval_tasks = zip(*self.evaluate_batch(child_list, problem_logger))
            problem_logger.log_eval_results(eval_results)
            logging.info(f"Time for evaluating all kernels {time.time() - tic}")

            if reference_test_result is None:
                # get the reference test result from just evaluated tasks once
                # This will get the first non-None reference test result found or None if none exist
                reference_test_result = next(
                    (et.test_result_reference for et in eval_tasks if et.test_result_reference is not None), None
                )
                build_result_reference = next(
                    (et.build_result_reference for et in eval_tasks if et.build_result_reference is not None), None
                )

            # Report fitness for prompt evolution (if enabled and in evolve mode)
            if self.evolve_mode and self.prompt_evolution_enabled:
                for k, (eval_result, session_id) in enumerate(zip(eval_results, prompt_evolution_session_ids)):
                    if session_id and eval_result:
                        # Compute fitness using unified performance scoring function
                        kernel_score = EvalResult.compute_performance_score(eval_result)
                        # Get the kernel code for this child (needed for meta-prompting evolution) + eval log
                        artifact_path = problem_logger.stdout_path_part + f"_v{k}.txt"
                        with open(artifact_path, "r") as inf:
                            eval_log = inf.read()
                        if child_list[k].code is not None:  # if extraction failed, do not report fitness
                            kernel_code = child_list[k].code + "\n ====== EVAL LOG =====\n" + eval_log
                            self.report_prompt_fitness(
                                session_id=session_id,
                                kernel_score=kernel_score,
                                kernel_code=kernel_code,
                                kernel_metrics={
                                    "correctness": 1.0 if getattr(eval_result, "correctness", False) else 0.0,
                                    "compiled": 1.0 if getattr(eval_result, "compiled", False) else 0.0,
                                    "runtime": getattr(eval_result, "runtime", None),
                                },
                            )
                # Periodically save prompt evolution state
                if trial % 10 == 0:
                    self.save_prompt_evolution_state()

            # add eval result to program database:
            for k, (child, eval_result, eval_task, prompt, llm_model) in enumerate(
                zip(child_list, eval_results, eval_tasks, prompt_list, model_list)
            ):
                artifact_path = problem_logger.stdout_path_part + f"_v{k}.txt"
                child.add_eval_results(eval_result, artifact_path)
                # only for evolve mode (and if kernel extraction did not fail): add to database
                if self.evolve_mode and child.code is not None:
                    self.program_database.add(child, iteration=trial)
                    self.program_database.increase_island_counter_and_switch()
                    # Check migration
                    if self.program_database.should_migrate():
                        logging.info("Performing migration")
                        self.program_database.migrate_programs()
                        self.program_database.log_island_status()

                kernel = db.Kernel(
                    task_name=task_name,
                    job_name=job_name,
                    input_code=reference,
                    input_language=self.reference_language,
                    prompt=prompt,
                    trial=trial,
                    version=k,
                    language_model=llm_model,
                    # only a single architecture is supported for now
                    gpu_arch=config.gpu_arch if isinstance(config.gpu_arch, str) else config.gpu_arch[0],
                    config=OmegaConf.to_container(config, resolve=True),
                    task_id=task_id,
                    job_id=job_id,
                )
                if self.config.get("store_generated_kernels_in_db", True):
                    child.update_Kernel(kernel)
                    if db.add_ignore_errors(kernel):
                        logging.info(f"Added kernel version {k} to database")

                programs[child.id] = child
            # # to visualize updated map elites grid:
            # self.program_database.save_grid_visualization(iteration=trial, include_esimd=False)

            # find best performing kernel
            selected_kernel_index = select_best_solution(eval_results)
            selected_kernel_raw = custom_kernel_out_list[selected_kernel_index]

            # 4) LOG RESULT
            result_list = "\n".join([str(res) for res in eval_results])
            logging.info(f"Results:\n{result_list}")
            logging.info(f"Stdout saved at {problem_logger.stdout_path_part}_v{selected_kernel_index}.txt")
            # copy best stdout to a standardized _best.txt file if it exists
            selected_stdout = f"{problem_logger.stdout_path_part}_v{selected_kernel_index}.txt"
            if os.path.exists(selected_stdout):
                shutil.copy(selected_stdout, f"{problem_logger.stdout_path_part}_best.txt")
                logging.info(f"Stdout saved at {problem_logger.stdout_path_part}_best.txt")
            else:
                logging.warning(
                    "No stdout was written for the selected version (%s), so there is no _best.txt "
                    "for the next trial to read. This normally means evaluation failed before it "
                    "produced any output; see the errors above for why.",
                    selected_stdout,
                )

            kernel_exec_result = eval_results[selected_kernel_index]
            kernel_exec_result.metadata["scores_versions"] = [res.perf_score for res in eval_results]

            if not self.evolve_mode:
                # set parent
                parent = child_list[selected_kernel_index]

            # log results (save to runs dir)
            problem_logger.save_gen_kernel(selected_kernel_raw)

            # make results dict, add to results list, and save to disk if it's not parallel mode
            res_dict = problem_logger.log_result(kernel_exec_result, results, save=self.resource_allocator is None)

            if config.stop_once_correct and kernel_exec_result.compiled and kernel_exec_result.correctness:
                break

            logging.info(f"Time total for trial {trial} {time.time() - tic_trial}")

            # Update job progress after completing the trial
            if job_id is not None:
                progress = (trial + 1) / config.max_iters
                db.update_job_progress(job_id, progress)

        logging.info(f"Time total for problem {task_name} {time.time() - tic_all}")

        # get the uuid of the best program
        programs_list = list(programs.values())
        programs_with_result = [prog for prog in programs_list if prog.kernel_exec_result is not None]
        program_eval_results = [prog.kernel_exec_result for prog in programs_with_result]
        best_result_idx = select_best_solution(program_eval_results)

        # Update Job status to COMPLETE
        if job_id is not None:
            db.update_job_status(job_id, "COMPLETE", finished_at=datetime.now(timezone.utc))

        return programs, programs_with_result[best_result_idx].id

    def evaluate_single(
        self,
        program: Program,
        problem_logger: ProblemLogger,
        version: int = 0,
    ) -> tuple[EvalResult, Task]:
        """Evaluate a single kernel program.

        Args:
            program (Program): Program object containing the kernel code
            problem_logger (ProblemLogger): Helper object for logging the individual outputs
            version (int, optional): Version index of the kernel. Defaults to 0.

        Returns:
            EvalResult: Evaluation result of the kernel
        """
        evaluator = Evaluator(self.config, problem_logger, version, kernel_uuid=program.id)
        task = program.task.with_blocks({"EVOLVE": program.code}, keep_test_result_reference=True)
        result = evaluator.run(task)
        # track failure cases: timeout or extraction error
        eval_result = result[0]
        if "timed out" in eval_result.eval_log or eval_result.perf_score == 0.0:
            self.failures += 1
            if self.failures >= self.max_failures:
                raise RuntimeError(f"Maximum number of failures ({self.max_failures}) reached. Aborting.")
        assert result is not None, "Evaluator returned None - this may be due to test_custom=False in config"
        return result

    @override
    def evaluate_batch(
        self,
        programs: list[Program],
        problem_logger: ProblemLogger,
    ) -> list[tuple[EvalResult, Task]]:
        """Evaluate a batch of kernels

        Args:
            programs (list[Program]): List of kernels as strings
            problem_logger (ProblemLogger): helper object for logging the individual outputs

        Returns:
            List of KernelEvalResults
        """
        # eval timeout: for each program, allow 5 times test timeout, times 2 (reference + custom), compile parallel
        build_timeout, test_timeout = self.config.get("build_timeout", 200), self.config.get("test_timeout", 120)
        timeout_evaluation = build_timeout + len(programs) * 5 * 2 * test_timeout + 600  # slack for wait time in queue
        # Evaluate each program in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.get("max_workers", 8)) as executor:
            futures = {
                executor.submit(self.evaluate_single, program, problem_logger, k): k
                for k, program in enumerate(programs)
            }
            eval_results = [None] * len(programs)
            for future in concurrent.futures.as_completed(futures, timeout_evaluation):
                k = futures[future]
                try:
                    eval_results[k] = future.result()
                except Exception as exc:
                    logging.error(f"Evaluation of program version {k} generated an exception: {exc}", exc_info=True)
                    # Create a failed result
                    eval_results[k] = (
                        EvalResult(compiled=False, correctness=False, perf_score=0.0, metadata={"error": str(exc)}),
                        programs[k].task,
                    )

        return eval_results
