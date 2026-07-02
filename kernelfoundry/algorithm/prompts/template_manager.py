"""Template rendering utilities for prompt assembly in kernel generation."""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import omegaconf
from jinja2 import Environment, PackageLoader, select_autoescape

from kernelfoundry.algorithm.schemas import Program
from kernelfoundry.eval_pipeline.utils.gpu_specs import ARCH_TO_NAME, ARCH_TO_SPECS
from kernelfoundry.algorithm.prompts.languages import KERNEL_OPTIMIZATION_TIPS

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Template manager with optional evolvable prompt components.

    Provides standard template construction and optionally supports:
    - Evolved optimization tips and strategies
    - Session tracking for fitness attribution
    - Automatic evolution based on kernel performance
    """

    def __init__(
        self,
        gpu_arch: str | list,
        language: str,
        ref_language: str,
        n_tips: int = 2,
        include_top: bool = True,
        include_inspirations: bool = True,
        use_hardware_prompt: bool = True,
        allow_templated: bool = False,
        template_example: Optional[str] = None,
        prompt_template_fn: str = "main_prompt.j2",
    ):
        """
        Initialize the template manager.

        Args:
            gpu_arch: Target GPU architecture(s)
            language: Target language (SYCL, CUDA, triton)
            ref_language: Reference language (Pytorch, CUDA, description)
            n_tips: Number of optimization tips to include
            include_top: Whether to include top program in prompt
            include_inspirations: Whether to include inspiration programs
            use_hardware_prompt: Whether to include hardware specs
            allow_templated: Whether to allow templated kernels
            template_example: Example of templated kernel format
            output_dir: Directory for saving evolution database (optional)
            llm_server: LLM server for mutation operations (optional)
            prompt_template_fn: Filename of the main prompt template
        """
        self.gpu_arch = gpu_arch
        self.language = language
        self.ref_language = ref_language
        assert n_tips < min([len(v) for _, v in KERNEL_OPTIMIZATION_TIPS.items()])
        self.n_tips = n_tips
        self.include_top = include_top
        self.include_inspirations = include_inspirations
        self.use_hardware_prompt = use_hardware_prompt

        # if allowing templated kernels or just standard optimization strategies
        self.allow_templated = allow_templated
        self.template_example = template_example

        env = Environment(loader=PackageLoader("kernelfoundry.algorithm.prompts"), autoescape=select_autoescape())
        self.prompt_template = env.get_template(prompt_template_fn)

    def construct_prompt(
        self,
        reference_code: str,
        last_program: Optional[Program] = None,
        prior_versions: List[Program] = [],
        top_program: Optional[Program] = None,
        rag_input: str = "",
        second_ref_code: Optional[str] = None,
        is_feedback: bool = False,
        evolvable_content: Optional[Dict[str, str]] = None,
        is_backward: bool = False,
    ) -> str:
        """Construct prompt from optional components.

        This method maintains compatibility with the existing API.

        Args:
            evolvable_content: Optional dict of evolved content for template regions.
                Keys should match template variables without the ``evolved_`` prefix:
                - optimization_philosophy
                - optimization_strategies
                - common_pitfalls
                - analysis_guidance
            is_backward: whether the operation is a backward pass
        """
        # check if any of the prior kernels is templated already
        any_templated_already = self._check_for_templated(prior_versions, last_program, top_program)
        algorithmic_strategies = self._format_optimization_strategies()

        # Prepare inspirations
        input_inspirations = [
            self._program_to_dict(prior_program)
            for prior_program in prior_versions
            # exclude top program and last program because included anyways
            if prior_program.code
            not in [
                top_program.code if top_program else None,
                last_program.code if last_program else None,
            ]
        ]

        # Get status: translation (last is None), error (last was error), or correct (last was correct)?
        status = self._determine_status(last_program)

        include_top = self.include_top
        if top_program is None:
            include_top = False
        elif last_program is not None and top_program.code == last_program.code:
            include_top = False

        user_instructions = ""
        if last_program and last_program.task:
            user_instructions = "\n".join(last_program.task.blocks.get("USER_INSTRUCTIONS", {}).values())

        # Get evolvable content (from parameter or stored state)
        evolved = evolvable_content or getattr(self, "_evolvable_content", {}) or {}

        # main function name
        function_name = "backward" if is_backward else "forward"

        # Render template
        filled_template = self.prompt_template.render(
            language=self.language,
            language_lower=self.language.lower(),
            # reference
            ref_code=reference_code,
            ref_language=self.ref_language,
            # second reference if available
            second_ref_code=second_ref_code,
            # rag_input (e.g. kernel examples, instructions for extensions)
            rag_input=rag_input,
            # inspirations
            include_inspirations=self.include_inspirations,
            inspirations=input_inspirations,
            # top
            include_top=include_top,
            top_program=self._program_to_dict(top_program),
            # last
            parent=self._program_to_dict(last_program),
            is_feedback=is_feedback,
            # hardware
            include_hardware_specs=self.use_hardware_prompt,
            hardware_specs=self._format_hardware_specs(),
            # final task description
            status=status,  # translate, error or correct
            # optimization strategies
            allow_templated=self.allow_templated,
            templated_example=self.template_example,
            any_templated_already=any_templated_already,
            algorithmic_strategies=algorithmic_strategies,
            # user instructions
            user_instructions=user_instructions,
            # Evolved content from meta-prompting
            evolved_optimization_philosophy=evolved.get("optimization_philosophy"),
            evolved_optimization_strategies=evolved.get("optimization_strategies"),
            evolved_common_pitfalls=evolved.get("common_pitfalls"),
            evolved_analysis_guidance=evolved.get("analysis_guidance"),
            function_name=function_name,
        )
        # remove empty lines
        filled_template = re.sub(r"\n\s*\n+", "\n\n", filled_template)
        return filled_template

    def _format_optimization_strategies(self):
        """Sample and format optimization tips for inclusion in the prompt."""
        tips_prompt = ""
        # assert self.n_tips > 0
        if self.n_tips > 0:
            # sample a selection of <self.n_tips> Tips for the language and inclucde in prompt
            tips = np.random.choice(KERNEL_OPTIMIZATION_TIPS[self.language], self.n_tips, replace=False)
            for i, tip in enumerate(tips):
                tips_prompt += f"{i+1}. {tip}\n"
        return tips_prompt

    def _format_hardware_specs(self):
        """Format configured GPU architecture specifications for prompt context."""
        gpu_arch_list = self.gpu_arch.split(",")
        hw_prompt_part = ""
        for arch in gpu_arch_list:
            assert arch in ARCH_TO_NAME, f"Unknown architecture gpu_arch: {arch}"
            hw_name = ARCH_TO_NAME[arch]
            specs = ARCH_TO_SPECS[arch]
            formatted_specs = ", ".join([f"{key}: {value}" for key, value in specs.items()])
            hw_prompt_part += f"**{hw_name}** with specs: " + formatted_specs
        return hw_prompt_part

    def _program_to_dict(self, program: Program | None):
        """Convert program to dictionary"""
        if program is not None:
            program_as_dict = {
                "code": program.code,
                "result": (
                    "(Result: " + program.kernel_exec_result.format_for_prompt() + ")"
                    if program.kernel_exec_result and program.kernel_exec_result.perf_score >= 0
                    # case where Kernelbench task and mode=standard -> prior results not loaded
                    else ""
                ),
                "console_output": program.kernel_exec_result.eval_log if program.kernel_exec_result else None,
                "feedback": program.feedback,
                "is_program0": program.is_program0,
                "task_origin": program.task.config.get("task_origin", None) if program.task else None,
            }
            # TODO Check if we use the to_dict from Program and remove this one?
            if program.kernel_exec_result is not None and program.kernel_exec_result.eval_log is not None:
                program_as_dict["console_output"] = program.kernel_exec_result.eval_log

            return program_as_dict
        else:
            return None

    def _check_for_templated(
        self,
        prior_versions: List[Program],
        last_program: Optional[Program],
        top_program: Optional[Program],
    ) -> bool:
        """Check if any of the versions included in the prompt are already templated"""
        included_versions = [p for p in prior_versions if self.include_inspirations]
        if last_program:
            included_versions.append(last_program)
        if self.include_top and top_program:
            included_versions.append(top_program)
        return any(
            program.is_templated or "forward_templated" in program.code
            for program in included_versions
            if program is not None and program.code is not None
        )

    def _determine_status(self, last_program: Optional[Program]) -> str:
        """Determine prompt status based on last program."""
        if last_program is None:
            return "translate"
        elif last_program.is_program0:
            return "translate"
        elif "correctness_success" in last_program.metrics:
            return "correct" if last_program.metrics["correctness_success"] else "error"
        else:
            if last_program.kernel_exec_result and last_program.kernel_exec_result.eval_log:
                return "correct" if "compiles and is correct" in last_program.kernel_exec_result.eval_log else "error"
            return "error"
