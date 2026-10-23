"""AnswerProcessor class for postprocessing LLM responses into Program objects."""

import logging
import uuid

from kernelfoundry.algorithm.schemas import Program
from kernelfoundry.algorithm.utils.extract_code import (
    extract_code_from_tags,
    extract_cpp_code_heuristic,
    extract_multi_block_code,
)
from kernelfoundry.algorithm.utils.code_editing import CodeEditing
from kernelfoundry.eval_pipeline.utils.custom_task_helper import block_labels

logger = logging.getLogger(__name__)


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
        assert not diff_format, "Diff format option is not supported anymore and will be removed in the future."
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
        changes_summary = "Full rewrite"
        all_blocks = [(fp, i) for fp, contents in parent.code.items() for i in range(len(contents))] if parent else []

        if len(all_blocks) > 1:
            # Multiple EVOLVE blocks: expect one labeled, fenced code section per block.
            child_code_dict = self._extract_multi_block_code(llm_response, parent.code, all_blocks)
        else:
            # Parse response based on evolution mode
            child_code = extract_code_from_tags(llm_response, self.language.lower())
            # Use fallback heuristic for C++/CUDA/SYCL as last resort
            if child_code is None and self.language.lower() in ("cpp", "cuda", "sycl"):
                child_code = extract_cpp_code_heuristic(llm_response)

            child_code = self._clean_and_postprocess(child_code)

            # Map extracted string back to block dict structure
            if parent is not None:
                fp = all_blocks[0][0]
                child_code_dict = {fp: [child_code]} if child_code else {fp: [""]}
            else:
                child_code_dict = {"generated": [child_code]} if child_code else {}

        child_id = str(uuid.uuid4())

        # Create child program
        if parent is None:
            # first iteration
            child_program = Program(
                id=child_id,
                code=child_code_dict,
                raw_llm_code=llm_response,
                language=self.language,
                iteration_found=iteration,
            )
        else:
            # has a parent
            child_program = Program(
                id=child_id,
                code=child_code_dict,
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

    def _clean_and_postprocess(self, code: str | None) -> str | None:
        """Strips stray evolution tags and applies configured code postprocessing to one code block."""
        if not code:
            return code
        code = code.replace("[EVOLVE_START]", "").replace("[EVOLVE_END]", "")
        if self.postprocess_code and self.postprocess_code_config.fix_sycl_queue_and_wait and self.language == "SYCL":
            code_editing = CodeEditing(code)
            code_editing.replace_queue_with_torch_queue_and_remove_wait()
            code = code_editing.source_code
        return code

    def _extract_multi_block_code(
        self,
        llm_response: str,
        parent_code: dict[str, list[str]],
        all_blocks: list[tuple[str, int]],
    ) -> dict[str, list[str]]:
        """Extracts one code block per EVOLVE region from a multi-block-formatted LLM response.

        Blocks the LLM didn't return a labeled section for are kept unchanged from the parent,
        rather than wiped, since a missing label most likely means the model chose not to touch
        that region rather than that it intentionally emptied it.
        """
        labels = block_labels(parent_code)
        use_cpp_fallback = self.language.lower() in ("cpp", "cuda", "sycl")
        extracted = extract_multi_block_code(llm_response, labels, self.language.lower(), use_cpp_fallback)

        if not extracted:
            logger.warning(
                f"AnswerProcessor could not find any of the {len(labels)} expected labeled EVOLVE "
                f"block(s) ({labels}) in the LLM response. The model likely did not follow the "
                "multi-block '### Block: <label> ###' response format at all.\n"
                f"Raw LLM response:\n{llm_response}"
            )
            return None

        missing = [label for label in labels if label not in extracted]
        if missing:
            logger.warning(
                "LLM response did not contain a labeled '### Block: <label> ###' section for %d/%d "
                "expected EVOLVE block(s): %s. Keeping those block(s) unchanged from the parent.",
                len(missing),
                len(labels),
                missing,
            )

        child_code_dict = {fp: list(contents) for fp, contents in parent_code.items()}
        for (fp, idx), label in zip(all_blocks, labels):
            code = extracted.get(label)
            child_code_dict[fp][idx] = self._clean_and_postprocess(code) if code else parent_code[fp][idx]
        return child_code_dict
