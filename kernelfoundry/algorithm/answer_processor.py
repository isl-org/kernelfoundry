"""AnswerProcessor class for postprocessing LLM responses into Program objects."""

import uuid

from kernelfoundry.algorithm.schemas import Program
from kernelfoundry.algorithm.utils.extract_code import (
    extract_code_from_tags,
    extract_cpp_code_heuristic,
    apply_diff,
    extract_diffs,
    format_diff_summary,
)
from kernelfoundry.algorithm.utils.code_editing import CodeEditing


class AnswerProcessor:
    """Takes a response from an LLM and postprocess it (extract code, edit code, transform to program)"""

    def __init__(self, language: str, diff_format: bool, postprocess_code: bool, postprocess_code_config):
        """Initialize the answer processor for LLM responses.

        Args:
            language (str): Programming language of the kernel.
            diff_format (bool): Whether to use diff format for code changes.
            postprocess_code (bool): Whether to apply code postprocessing (e.g. fix SYCL queue).
            postprocess_code_config: Configuration for code postprocessing.
        """
        self.language = language
        self.diff_format = diff_format
        self.postprocess_code = postprocess_code
        self.postprocess_code_config = postprocess_code_config

    def __call__(self, llm_response: str, iteration: int, parent: Program | None) -> Program:
        """Transforms an LLM output into a Program object

        Args:
            llm_response (str): Raw output of the LLM
            iteration (int): Current iteration
            parent (Program | None): Program that was used as input to generate the current version

        Returns:
            Program: new Program object storing code, parent id, etc
        """
        # Parse response based on evolution mode
        if self.diff_format and parent is not None:
            # # prior version: load parent code from file and extract code
            # parent_code = extract_code_from_tags(problem_logger.read_prior_gen_code(), self.config.language.lower())
            diff_blocks = extract_diffs(llm_response)
            if len(diff_blocks) == 0:
                child_code = "fail-placeholder"
            else:
                child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            child_code = extract_code_from_tags(llm_response, self.language.lower())
            # Use fallback heuristic for C++/CUDA/SYCL as last resort
            if child_code is None and self.language.lower() in ("cpp", "cuda", "sycl"):
                child_code = extract_cpp_code_heuristic(llm_response)
            changes_summary = "Full rewrite"

        # Remove evolution tags that should not appear in the final code
        if child_code:
            child_code = child_code.replace("[EVOLVE_START]", "").replace("[EVOLVE_END]", "")

        if self.postprocess_code and child_code:
            if self.postprocess_code_config.fix_sycl_queue_and_wait and self.language == "SYCL":
                code_editing = CodeEditing(child_code)
                code_editing.replace_queue_with_torch_queue_and_remove_wait()
                child_code = code_editing.source_code

        child_id = str(uuid.uuid4())

        # Create child program
        if parent is None:
            # first iteration
            child_program = Program(
                id=child_id,
                code=child_code,
                raw_llm_code=llm_response,
                language=self.language,
                iteration_found=iteration,
            )
        else:
            # has a parent
            child_program = Program(
                id=child_id,
                code=child_code,
                raw_llm_code=llm_response,
                language=self.language,
                parent_id=parent.id,
                generation=parent.generation + 1,
                iteration_found=iteration,
                task=parent.task,
                metadata={
                    "changes": changes_summary,
                    "parent_metrics": parent.metrics,
                    "island": parent.metadata.get("island", None),
                },
                # metrics will be passed later
            )
        return child_program
