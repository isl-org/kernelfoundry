"""AgenticController: runs a CopilotCLIAgent for a fixed number of iterations."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Optional
import hydra
import numpy as np

from kernelfoundry.algorithm.copilot_cli_agent import CopilotCLIAgent
from kernelfoundry.algorithm.schemas import EvalResult, Program
import kernelfoundry.eval_pipeline.database as db
from kernelfoundry.eval_pipeline.task import Task, TestResult
from kernelfoundry.eval_pipeline.tasks.task_runner import TaskRunner
from kernelfoundry.eval_pipeline.utils.custom_task_helper import blocks_to_str
from kernelfoundry.eval_pipeline.utils.container import Image, get_container_runtime, select_container_image
from kernelfoundry.algorithm.prompts.feedback_llm import FeedbackHelper
from kernelfoundry.algorithm.prompts.prompt_constructor import PromptConstructor, get_system_prompt
from kernelfoundry.algorithm.prompts.prompt_evolution_integration import PromptEvolutionMixin
from kernelfoundry.algorithm.prompts.optimization_aware import (
    build_exploration_prompt,
    get_optimization_guidance_for_parent,
)
from kernelfoundry.algorithm.utils.logging import setup_logging
from kernelfoundry.algorithm.utils.skills import Skill, SkillLibrary
from kernelfoundry.eval_pipeline.profiler_feedback import (
    get_profiler_feedback_class,
    get_reference_language_for_profiling,
)
from kernelfoundry.algorithm.utils.token_usage import zero_token_usage

# Extra ``languages`` metadata values that should match a given task language
# when filtering skills. The task language maps to the set of accepted values.
LANGUAGE_ALIASES: dict[str, list[str]] = {
    "opencl": ["opencl", "ocl"],
}

# def _build_prompt(task: Task) -> str:
#     """Build a simple prompt from the task's USER_INSTRUCTIONS block, falling back to a generic one."""
#     user_instructions = task.blocks.get("USER_INSTRUCTIONS", {})
#     if user_instructions:
#         return "\n\n".join(user_instructions.values())

#     language = task.config.get("language", "kernel")
#     return (
#         f"Optimize the {language} code in this workspace for better performance. "
#         f"Use the build_and_test tool to evaluate your changes after each modification."
#         f"Explain your strategy and reasoning in detail and summarize the changes you do each step."
#     )


class AgenticController(PromptEvolutionMixin):
    """Runs a :class:`~kernelfoundry.algorithm.copilot_cli_agent.CopilotCLIAgent`
    for a fixed number of iterations.

    Args:
        run_config: Hydra configuration object (must have ``logdir`` and ``max_iters``).
        job_id: Job identifier passed through to the agent.
        task_id: Task identifier (stored for reference).
        copilot_exe: Path or name of the Copilot CLI binary.
        env_overrides: Extra environment variables forwarded to the Copilot subprocess.
        agent_timeout: Per-iteration timeout in seconds.
        logging_level: Logging level for the file handler added by this controller.
        extra_mcp_servers: Optional additional MCP server definitions merged into
            the agent configuration.
    """

    def __init__(
        self,
        run_config,  # Parameter name must not be "config" to avoid a conflict with hydra.instantiate
        job_id: int | None,
        task_id: int | None,
        copilot_exe: str = "copilot",
        env_overrides: dict[str, str] | None = None,
        agent_timeout: float = 3600.0,
        logging_level: int | str | None = None,
        extra_mcp_servers: dict[str, dict] | None = None,
    ):
        self.config = run_config
        self.job_id = job_id
        self.task_id = task_id
        self.copilot_exe = copilot_exe
        self.env_overrides = env_overrides
        self.agent_timeout = agent_timeout
        self.extra_mcp_servers = dict(extra_mcp_servers or {})
        self.skills = self._load_skills()

        setup_logging(Path(run_config.logdir) / "controller.log", logging_level=logging_level)
        TaskRunner.init(use_queue=run_config.get("use_queue", True), gpu_arch=run_config.gpu_arch)

        # self.answer_processor = AnswerProcessor(
        #     language=config.language,
        #     diff_format=config.prompt.diff_format,
        #     postprocess_code=config.postprocess_code,
        #     postprocess_code_config=config.postprocess_code_config,
        # )

        self.reference_language = self.config.prompt.get("reference_language", "Pytorch")

        self.llm_server = hydra.utils.instantiate(self.config.inference)
        logging.info(f"Initiated LLM with config {self.config.inference}")

        self.feedback_helper = FeedbackHelper(
            use_feedback_llm=self.config.use_feedback_llm,
            language=self.config.language,
            server_config=self.config.feedback_llm_config,
            use_docs_via_keywords=self.config.use_docs_via_keywords,
        )
        logging.info(f"Initiated feedback LLM with config: {self.config.feedback_llm_config}")

        mcp_prompt_list = [
            str(server_cfg.get("prompt")) for server_cfg in self.extra_mcp_servers.values() if server_cfg.get("prompt")
        ]

        self.prompt_constructor = PromptConstructor(
            self.config.language,
            self.config.gpu_arch,
            self.config.prompt,
            reference_language=self.reference_language,
            mode=self.config.mode,
            use_feedback_llm=self.config.use_feedback_llm,
            mcp_prompt_list=mcp_prompt_list,
        )
        logging.info(
            f"Initiated prompt constructor: language: {self.config.language}, use diff: {self.config.prompt.diff_format}"
        )

        self.program_database = hydra.utils.instantiate(self.config.database)
        logging.info("Initiated program database with config:", self.config.database)

        # Initialize prompt evolution system (from PromptEvolutionMixin)
        self.setup_prompt_evolution()

    def _load_skills(self) -> list[Skill]:
        """Load skills from ``config.skills.directories`` filtered by language.

        Relative directories are resolved against the repository root. For
        OpenCL tasks, skills tagged with either ``opencl`` or ``ocl`` are
        included.
        """
        skills_config = self.config.get("skills", None)
        directories = skills_config.get("directories", None) if skills_config else None
        language = self.config.get("language")
        if not directories or not language:
            return []

        languages_include = LANGUAGE_ALIASES.get(language.lower(), [language])
        skills: list[Skill] = []
        for directory in directories:
            directory_path = Path(directory)
            if not directory_path.is_absolute():
                directory_path = Path.cwd() / directory_path
            if not directory_path.is_dir():
                logging.warning("Skills directory does not exist: %s", directory_path)
                continue
            library = SkillLibrary.load(directory_path)
            skills.extend(library.filter(languages_include=languages_include))

        logging.info(
            "Checked the KernelFoundry skills database for skills for language %r: %d skill(s) loaded",
            language,
            len(skills),
        )
        return skills

    @staticmethod
    def build_container_image_for_task(task: Task) -> Task:
        """Build the container image for the given Task if ``use_container`` is True.

        Updates ``task.config["container_image"]`` in-place and returns the task.
        """
        if task.config.get("use_container", False):
            if task.config.get("container_image") is None:
                build_image_ans = TaskRunner.build_image(task)
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

    @staticmethod
    def _resolve_agent_container_image(task: Task) -> Image | None:
        """Resolve the container image used to run the Copilot CLI agent.

        Returns ``None`` when ``use_container`` is disabled.
        """
        # mirrors the image-resolution logic in kernelfoundry.eval_pipeline.tasks.test_custom_task`.
        # The container_image config entry may be a plain image name or a mapping
        # keyed by language and GPU architecture (with an ``"all"`` fallback).

        if not task.config.get("use_container", False):
            return None
        container_image = task.config.get("container_image")
        language = (task.config.get("language") or "").lower()
        gpu_arch = task.config.get("gpu_arch")
        if isinstance(gpu_arch, (list, tuple)):
            gpu_arch = gpu_arch[0]
        if isinstance(container_image, dict):
            container_image = select_container_image(container_image, language, gpu_arch)
        if not container_image:
            raise ValueError(f"No container image configured for language={language!r} gpu_arch={gpu_arch!r}")
        paths = task.config.get("paths", {}) or {}
        runtime = get_container_runtime()(
            registry=paths.get("container_registry"),
            allowed_registries=paths.get("allowed_container_registries"),
        )
        image = runtime.get_image(container_image)
        if image is None:
            raise RuntimeError(f"Container image not found: {container_image}")
        return image

    def sample_evolve_programs(self):
        """Sample parent code, inspirations, top program from the database"""
        parent, inspirations = self.program_database.sample()
        # Get parent artifacts if available
        parent_console_log = parent.get_artifact()

        if self.config.use_feedback_llm:
            # Get feedback from feedback LLM -> not saving the feedback in parent because we want varying feedback!
            parent_feedback = self.feedback_helper.get_feedback(parent.code_as_str, parent_console_log)
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

    def create_program0(self, task: Task) -> Program:
        """Create initial Program object for a Task.

        Args:
            task: Task instance containing problem details.

        Returns:
            Initial Program object built from the EVOLVE block.
        """
        code = task.blocks["EVOLVE"]
        program0 = Program(id="", is_program0=True, code=code, language=self.config.language, task=task)
        return program0

    @staticmethod
    def _select_evolution_strategy(
        parent_profile: dict,
        base_strategy: str,
        enable_esimd: bool = False,
    ) -> str:
        """Select the best evolution strategy based on parent profile and exploration state."""
        import random

        esimd_level = parent_profile.get("esimd_opt", 0)
        memory_level = parent_profile.get("memory_opt", 0)
        compute_level = parent_profile.get("compute_opt", 0)
        parallelism_level = parent_profile.get("parallelism_opt", 0)

        non_esimd_total = memory_level + compute_level + parallelism_level

        if enable_esimd:
            if esimd_level == 0:
                if non_esimd_total >= 6:
                    if random.random() < 0.6:
                        return "esimd_upgrade"
                elif non_esimd_total >= 3:
                    if random.random() < 0.4:
                        return "esimd_upgrade"
                elif non_esimd_total >= 1:
                    if random.random() < 0.2:
                        return "esimd_upgrade"
            elif esimd_level in [1, 2]:
                if random.random() < 0.35:
                    return "esimd_upgrade"

        return base_strategy

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

        guidance_exploration_rate = min(1.0, max(0.0, guidance_exploration_rate))

        underexplored = self.program_database.get_underexplored_regions(n=self.config.branches_per_iteration)

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

    def _apply_optimization_aware_prompting(
        self,
        prompt: str,
        parent: Optional[Program],
        target_optimization_profile: Optional[dict] = None,
        strategy: Optional[str] = None,
    ) -> tuple:
        """Apply optimization-aware prompting enhancements, including evolved prompt genes.

        Args:
            prompt: Base prompt.
            parent: Parent program (if available).
            target_optimization_profile: Optimization profile dict to target.
            strategy: Evolution strategy name.

        Returns:
            Tuple of (enhanced_prompt, session_id) where session_id is for prompt evolution tracking.
        """
        session_id = None

        enable_esimd = getattr(self.config, "enable_esimd_exploration", False)
        if not target_optimization_profile:
            return prompt, session_id

        effective_strategy = strategy or "diversify"

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

    def evolve_prompt(
        self, problem_name: str, ref_arch_src: str, trial: int, task: Optional["Task"] = None
    ) -> tuple[str, Optional[Program]]:
        """
        Construct a fresh prompt for a single branch and report the sampled parent.

        Args:
            problem_name: Operation name
            ref_arch_src: Reference source code
            trial: Trial number (to pass to LLM server)
            task: Custom task object that can provide the initial implementation

        Returns:
            A ``(prompt, parent)`` pair: the prompt used for generation, and the parent
            program sampled from the database (``None`` if no parent was available). The
            parent's ``id`` identifies the agent to fork in order to continue the search
            from it.
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

        print("Time for prompt sampling:", time.time() - iteration_start)
        return prompt, parent

    def evaluate_reference(self, task: Task) -> tuple[Task, dict[str, EvalResult]]:
        if not task.config.get("test_reference", True):
            logging.info("Skipping reference evaluation as per config")
            return task, {}

        # backup original config
        original_config = task.config.copy()
        # alter config to run reference test only
        task.config["has_build_step"] = False
        task.config["language"] = get_reference_language_for_profiling(task.config)
        task.config["test_custom"] = False  # only test reference, not custom result
        gpu_arch = task.config["gpu_arch"].split(",")
        reference_gpu_arch = task.config.get("reference_gpu_arch", "").split(",")
        assert (
            len(reference_gpu_arch) <= 1
        ), "Only one reference gpu arch is supported if reference runs on other hardware"
        if reference_gpu_arch:
            gpu_arch = reference_gpu_arch
            task.config["gpu_arch"] = gpu_arch[0]

        # build
        logging.info(f"Building reference for gpu_archs {gpu_arch} with language {task.config['language']}")
        if task.has_reference_build_step:
            built_task = TaskRunner.build_custom_task(task)
            task.build_result_reference = built_task.build_result_reference
            if task.build_result_reference is not None:
                assert task.build_result_reference.result.returncode == 0, f"Reference build failed, cannot proceed"
        logging.build_test_result(task.build_result_reference, prefix="build_reference")
        # run reference on the specified gpu(s)
        test_results = {}
        for arch in gpu_arch:
            test_results[arch] = TaskRunner.test_custom_task(task, gpu_arch=arch)["reference"]
            logging.build_test_result(test_results[arch], prefix="test_reference")
        # restore config
        task.config = original_config
        # disable reference build and test for later since we already did it here
        task.config["test_reference"] = False

        # merge test results into a single EvalResult for the reference
        task.test_result_reference = TestResult.merge(test_results)

        return task, test_results

    def run(self, task: Task) -> list[tuple[Program, EvalResult]]:
        """Create an agent for *task* and run it for ``config.max_iters`` iterations.

        Args:
            task: The task to optimize.

        Returns:
            All ``(Program, EvalResult)`` pairs collected across all iterations,
            in the order they were produced.
        """
        tic_all = time.time()
        config = self.config

        job_name = config.get("job_name", "")
        task_name = config.get("task_name", "")
        level = config.get("level", 99)
        logging.info(f"========= Starting {job_name=}, {task_name=}) ========")

        # attach the config for this job to the custom task after saving to the database
        # The task object is passed to the workers which get access to the config for the job this way
        task.apply_config(config)

        reference = blocks_to_str(task.blocks["REFERENCE"]) if task.blocks.get("REFERENCE") else ""
        if not reference and task.config.get("test_reference", True):
            logging.warning(
                "No REFERENCE block found in task, but test_reference=True. "
                "The LLM will receive no reference code in its prompt."
            )

        # Update Job status to RUN and set started_at timestamp
        if self.job_id is not None:
            db.update_job_status(self.job_id, "RUN", started_at=datetime.now(timezone.utc))

        task = self.build_container_image_for_task(task)
        task, _ = self.evaluate_reference(task)

        agent_container_image = self._resolve_agent_container_image(task)

        all_results: list[tuple[Program, EvalResult]] = []
        token_usage_total = zero_token_usage()

        # Maps every produced program id to the agent that produced it, so that a
        # later iteration can fork the right agent given a sampled parent's id.
        program_to_agent: dict[str, CopilotCLIAgent] = {}

        def _make_fresh_agent(branch: int) -> CopilotCLIAgent:
            return CopilotCLIAgent(
                task=task,
                job_id=self.job_id,
                task_id=self.task_id,
                config=task.config,
                container_image=agent_container_image,
                copilot_exe=self.copilot_exe,
                env_overrides=self.env_overrides,
                branch=branch,
                skills=self.skills,
                extra_mcp_servers=self.extra_mcp_servers,
            )

        for i in range(config.max_iters):
            logging.info("Iteration %d / %d", i + 1, config.max_iters)

            # ---- Selection ----
            # Sample a parent and build a fresh prompt for each branch, then pick
            # which agent to fork. This is sequential because it reads/samples
            # from the program database. Exactly config.branches_per_iteration
            # agents are selected, so only that many are active this iteration.
            branch_specs: list[tuple[CopilotCLIAgent, str]] = []
            for branch in range(config.branches_per_iteration):
                prompt, parent = self.evolve_prompt(task_name, reference, i, task=task)

                parent_id = parent.id if parent is not None else None
                source_agent = program_to_agent.get(parent_id) if parent_id else None
                if source_agent is not None:
                    agent = source_agent.fork(branch=branch, parent_program=parent)
                    logging.info(f"Iteration {i + 1} branch {branch}: forking agent of parent {parent_id}")
                else:
                    agent = _make_fresh_agent(branch)
                    logging.info(f"Iteration {i + 1} branch {branch}: starting fresh agent (parent={parent_id})")
                branch_specs.append((agent, prompt))

            # ---- Execution ----
            # Run the selected branch agents in parallel and collect their results.
            iteration_results: list[tuple[Program, EvalResult]] = []
            max_workers = min(config.branches_per_iteration, os.cpu_count())
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_agent = {
                    executor.submit(agent.run, prompt, iteration=i, timeout=self.agent_timeout): agent
                    for agent, prompt in branch_specs
                }
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        results = future.result()
                    except TimeoutError:
                        logging.error(f"Iteration {i + 1} timed out after {self.agent_timeout:.0f}s")
                        continue
                    except Exception as e:
                        logging.error(f"Iteration {i + 1} failed with exception: {e}", exc_info=True)
                        continue
                    iteration_results.extend(results)
                    token_usage_total += agent.token_usage
                    if self.job_id is not None:
                        db.update_job_token_usage(
                            self.job_id,
                            input_tokens=token_usage_total.input_tokens,
                            output_tokens=token_usage_total.output_tokens,
                            cached_input_tokens=token_usage_total.cached_input_tokens,
                        )
                    # Associate every program this agent produced with the agent so
                    # that a future iteration can fork it via the sampled parent id.
                    for program, _eval_result in results:
                        program_to_agent[program.id] = agent

            # ---- Bookkeeping ----
            # Register the new programs in the evolution database so the next
            # iteration's sample_evolve_programs() can propose them as parents.
            for program, _eval_result in iteration_results:
                if getattr(program, "code", None) is not None:
                    self.program_database.add(program, iteration=i)

            all_results.extend(iteration_results)
            logging.info("Iteration %d produced %d result(s)", i + 1, len(iteration_results))

            # Update job progress after completing the iteration
            if self.job_id is not None:
                db.update_job_progress(self.job_id, (i + 1) / config.max_iters)

        logging.info(f"Time total for problem {task_name} {time.time() - tic_all}")

        return all_results
