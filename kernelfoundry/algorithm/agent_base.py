from __future__ import annotations

from abc import ABC, abstractmethod
import uuid
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING
from kernelfoundry.eval_pipeline.task import Task
from kernelfoundry.algorithm.schemas import EvalResult, Program
from kernelfoundry.eval_pipeline.utils.container import Image
from kernelfoundry.eval_pipeline.utils.custom_task_helper import blocks_to_str
import kernelfoundry.eval_pipeline.database as db
import logging

if TYPE_CHECKING:
    from kernelfoundry.algorithm.utils.skills import Skill


class EvaluateFunctionResult(NamedTuple):
    tool_response: dict
    eval_result: EvalResult | None
    program: Program | None


class BuildAndTestHandler:
    """Handler for evaluating agent-produced kernel folders.

    Subclass and override :meth:`call` to customise evaluation logic, or
    compose a separate object for any post-session teardown logic.

    Example::

        from kernelfoundry.algorithm.agent_base import AgentBase, BuildAndTestHandler

        class MyHandler(BuildAndTestHandler):
            def call(self, task, folder_path, job_id, task_id, prompt,
                     iteration, branch, llm_model, session_log,
                     previous_program=None):
                result = super().call(
                    task, folder_path, job_id, task_id, prompt,
                    iteration, branch, llm_model, session_log,
                    previous_program,
                )
                # Optionally modify result here.
                return result

        agent = MyAgent(task, job_id, build_test_handler=MyHandler())
    """

    def call(
        self,
        task: Task,
        folder_path: str | Path,
        job_id: int,
        task_id: str,
        prompt: str,
        iteration: int,
        branch: int,
        llm_model: str,
        session_log: str,
        previous_program: Program | None = None,
        agent_session_id: str | None = None,
    ) -> EvaluateFunctionResult:
        """Evaluate an agent-produced folder and return an :class:`EvaluateFunctionResult`.

        Extracts the EVOLVE block from *folder_path*, combines it with *task* via
        :func:`~kernelfoundry.eval_pipeline.task.Task.with_blocks`, then runs the
        :class:`~kernelfoundry.algorithm.evaluator.Evaluator` on the result.

        Args:
            task: The base task whose EVOLVE block is replaced by the content in
                *folder_path*.
            folder_path: Path to the folder produced by the agent (as passed to
                the ``build_and_test`` MCP tool).
            job_id: Job ID associated with the current agent run.
            task_id: Task identifier for the current run.
            prompt: The prompt that was given to the agent.
            iteration: The current iteration number passed to the agent's run function.
            branch: The branch identifier.
            llm_model: The language model used by the agent, for logging purposes.
            session_log: The session log for the current agent run.
            previous_program: The previously generated program, if any.
            agent_session_id: The session identifier of the agent that produced this kernel, if any.

        Returns:
            An :class:`EvaluateFunctionResult` whose ``tool_response`` dict contains
            the keys expected by the ``build_and_test`` tool:

            * ``success`` (bool) — whether the kernel passed the correctness check
            * ``job_id`` (int) — the supplied *job_id*
            * ``eval_log`` (str) — the condensed evaluation log
            * ``runtime_stats`` (dict) — detailed runtime statistics from the evaluation
            * ``speedup`` (float) — runtime improvement compared to the reference implementation
        """
        import tempfile
        from kernelfoundry.algorithm.evaluator import Evaluator
        from kernelfoundry.algorithm.problem_logger import ProblemLogger

        try:
            source_task, _ = Task.create(Path(folder_path))
            evolve_block = source_task.blocks.get("EVOLVE")
            new_task = task.with_blocks({"EVOLVE": evolve_block}, keep_test_result_reference=True)
        except Exception as e:
            return EvaluateFunctionResult(
                tool_response={
                    "success": False,
                    "job_id": job_id,
                    "eval_log": "Failed to extract EVOLVE block. Make sure that the code is properly tagged with [EVOLVE_START] and [EVOLVE_END].",
                    "runtime_stats": {},
                    "speedup": -1.0,
                },
                eval_result=None,
                program=None,
            )

        with tempfile.TemporaryDirectory(prefix="kf_eval_") as logdir:
            problem_logger = ProblemLogger(level=0, problem_id=job_id, logdir=logdir, trial=0)
            evaluator = Evaluator(config=new_task.config, problem_logger=problem_logger, version=0)
            eval_result, _ = evaluator.run(new_task)

        kernel_uuid = str(uuid.uuid4())

        evolve_code = blocks_to_str(evolve_block)

        if task.config.get("store_generated_kernels_in_db", True):
            reference = blocks_to_str(task.blocks["REFERENCE"]) if task.blocks.get("REFERENCE") else ""
            reference_language = task.config.get("prompt", {}).get("reference_language", "Pytorch")
            kernel = db.Kernel(
                task_name=task.config["task_name"],
                job_name=task.config["job_name"],
                input_code=reference,
                input_language=reference_language,
                prompt=prompt,
                trial=iteration,
                version=branch,
                language_model=llm_model,
                # only a single architecture is supported for now
                gpu_arch=(
                    task.config["gpu_arch"] if isinstance(task.config["gpu_arch"], str) else task.config["gpu_arch"][0]
                ),
                config=task.config,
                task_id=task_id,
                job_id=job_id,
                output_code=evolve_code,
                output_language=task.config.get("language"),
                answer=session_log,
                uuid=kernel_uuid,
                parent_uuid=previous_program.id if previous_program else None,
                agent_session_id=agent_session_id,
            )
            logging.info(
                f"task_id={task.config.get('task_id')} Generated kernel version {kernel.version} with UUID {kernel.uuid}"
            )

            Program.populate_kernel_from_exec_result(kernel, eval_result)

            if db.add_ignore_errors(kernel):
                logging.info(f"Added kernel version {branch} to database")

        metadata = {}
        if previous_program:
            metadata = {
                "changes": "Full rewrite",
                "parent_metrics": previous_program.metrics,
                "island": previous_program.metadata.get("island", None),
            }
        program = Program(
            id=kernel_uuid,
            code=evolve_block,
            raw_llm_code=evolve_code,
            language=task.config.get("language"),
            iteration_found=iteration,
            parent_id=previous_program.id if previous_program else None,
            task=new_task,
            generation=previous_program.generation + 1 if previous_program else 0,
            metadata=metadata,
        )
        program.add_eval_results(eval_result)

        return EvaluateFunctionResult(
            tool_response={
                "success": eval_result.correctness,
                "job_id": job_id,
                "eval_log": eval_result.eval_log,
                "runtime_stats": eval_result.runtime_stats,
                "speedup": eval_result.runtime_improvement,
            },
            eval_result=eval_result,
            program=program,
        )


class AgentBase(ABC):
    """Base class for agents to work on a task.

    The agent autonomously generates solutions to the given task and evaluates
    them using the build_and_test tool provided by the MCP server.

    Example usage pattern of the agent::

        from kernelfoundry.algorithm.agent_base import AgentBase, BuildAndTestHandler

        # Subclass BuildAndTestHandler to customise evaluation or post-session behaviour
        class MyHandler(BuildAndTestHandler):
            def call(self, task, folder_path, job_id, task_id, prompt,
                     iteration, branch, llm_model, session_log,
                     previous_program=None):
                result = super().call(
                    task, folder_path, job_id, task_id, prompt,
                    iteration, branch, llm_model, session_log,
                    previous_program=previous_program,
                )
                # Optionally modify result here.
                return result

            def session_end(self, session_log):
                print(session_log)

        # Initialize the agent with a task and job ID
        agent = MyAgent(task, job_id, build_test_handler=MyHandler())

        ans = agent.run("Optimize the kernel for better performance.")

        agent2 = agent.fork()  # Create a new instance continuing from the same session state
        # These can run in separate threads to explore in parallel.
        agent.run("Continue your optimization efforts.")
        agent2.run("Focus on reducing memory bandwidth usage.")

    """

    def __init__(
        self,
        task: Task,
        job_id: int,
        task_id: str,
        config: dict,
        container_image: Image | None = None,
        initial_session_state: dict | None = None,
        build_test_handler: BuildAndTestHandler | None = None,
        branch: int = 0,
        parent_session_uuid: str | None = None,
        parent_program: Program | None = None,
        skills: list[Skill] | None = None,
    ):
        """Initialize the agent with a starting point task and job ID.

        Args:
            task (Task): The task for the agent to work on.
            job_id (int): The job ID associated with this agent.
            config (dict): The main configuration dictionary for the job.
            container_image (Image | None): Optional image for running the agent in a containerized environment.
            initial_session_state (dict | None): Optional session state to restore from a previous run,
                as returned by :meth:`session_state`.  When provided, the agent continues from that
                state rather than starting a fresh session.
            build_test_handler: A :class:`BuildAndTestHandler` instance whose :meth:`~BuildAndTestHandler.call`
                method is invoked after every ``build_and_test`` tool call.
                Defaults to a plain :class:`BuildAndTestHandler` instance.
            branch: An integer identifier for the branch used for logging.
            parent_session_uuid: The session UUID of the parent agent, if any.
            parent_program: The program used as the parent for the next evaluation, if any.
            skills: Optional list of :class:`~kernelfoundry.algorithm.utils.skills.Skill`
                instances to make available to the agent. Any filtering is expected
                to be done by the caller.
        """
        self._task = task
        self._job_id = job_id
        self._task_id = task_id
        self._config = config
        self._container_image = container_image
        self._initial_session_state = dict(initial_session_state or {})
        self._handler = build_test_handler if build_test_handler is not None else BuildAndTestHandler()
        self._skills = list(skills) if skills is not None else []
        self._branch = branch
        self._session_uuid = str(uuid.uuid4())
        self._parent_session_uuid = parent_session_uuid
        self.set_parent_program(parent_program)

    def set_parent_program(self, program: Program | None) -> None:
        """Set the parent program used as the parent for the next evaluation."""
        self._parent_program = program

    @abstractmethod
    def run(self, prompt: str, iteration: int) -> list[tuple[Program, EvalResult]]:
        """Run the agent with the given prompt and return a list of (Program, EvalResult) tuples.

        Args:
            prompt (str): The input prompt for the agent to process.
            iteration (int): The current iteration number, starting from 0.

        Returns:
            list[tuple[Program, EvalResult]]: A list of tuples containing the generated
                program for each build_and_test call and the respective evaluation result.
        """
        pass

    @abstractmethod
    def fork(self, branch: int, parent_program: Program | None = None) -> AgentBase:
        """Create a new instance of the agent with the same session state.

        Args:
            branch (int): An integer identifier for the branch used for logging.
            parent_program (Program | None): The program to set as the parent for the
                forked agent's next evaluation, if any.

        This function must not be called while the agent is running.
        """
        pass

    @abstractmethod
    def session_state(self) -> dict:
        """Return the current session state of the agent as a dictionary.

        The returned dict is JSON serializable and should contain all
        necessary information to restore the agent's state in a new instance.
        Note that the state is implementation-specific and may include binary data
        using Base64 encoding or similar approaches.
        """
        pass
