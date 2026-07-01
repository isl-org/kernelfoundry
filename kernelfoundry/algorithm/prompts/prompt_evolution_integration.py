"""
Interface to integrate prompt evolution with controller.

This module provides integration between prompt evolution systems and the
kernel generation controller, enabling co-evolution of prompts and kernels.

**Supports holistic evolution mode:**

Holistic (Science-CodeEvolve): Prompts evolve as complete programs
   - Uses MetaPromptingManager
   - Full prompt evolution via SEARCH/REPLACE diffs
   - Direct fitness attribution (prompt.fitness = solution.fitness)

**Integration points:**

1. Controller initialization: Setup prompt evolution manager(s)
2. Prompt construction: Sample evolved content during generation
3. Fitness reporting: Update fitness after kernel evaluation
4. Evolution trigger: Periodically evolve prompts based on performance

**Usage:**

In the controller's __init__:

    from kernelfoundry.algorithm.prompts.prompt_evolution_integration import PromptEvolutionMixin

    class Controller(PromptEvolutionMixin):
        def __init__(self, config, ...):
            super().__init__(config, ...)
            self.setup_prompt_evolution()

In evolve_prompt_and_inference:

    prompt, session_id = self.construct_evolved_prompt(...)
    # ... generate kernel ...
    self.report_prompt_fitness(session_id, kernel_score)
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

# Lazy import for meta-prompting to avoid circular dependencies
_meta_prompting_module = None


def _get_meta_prompting_module():
    """Lazy import of meta_prompting module."""
    global _meta_prompting_module
    if _meta_prompting_module is None:
        from kernelfoundry.algorithm.prompts import meta_prompting as mp

        _meta_prompting_module = mp
    return _meta_prompting_module


if TYPE_CHECKING:
    from kernelfoundry.algorithm.schemas import Program
    from kernelfoundry.algorithm.prompts.meta_prompting import MetaPromptingManager, PromptProgram
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PromptEvolutionMode:
    """Prompt evolution mode constants."""

    HOLISTIC = "holistic"
    DISABLED = "disabled"


class PromptEvolutionMixin:
    """
    Mixin class to add prompt evolution capabilities to the Controller.

    This mixin provides:
    - Initialization of holistic prompt manager
    - Unified interface for prompt sampling and fitness reporting
    - Automatic mode selection based on configuration

    Usage:
        class Controller(PromptEvolutionMixin):
            def __init__(self, config):
                # ... existing init ...
                self.setup_prompt_evolution()
    """

    # These attributes should be set by the inheriting class
    config: "DictConfig"
    llm_server: Callable

    def setup_prompt_evolution(self):
        """
        Initialize prompt evolution system(s).

        Call this after the controller's main initialization.
        Reads configuration from self.config.prompt for evolution settings.
        """
        from omegaconf import OmegaConf

        # Initialize state
        self._prompt_evolution_enabled = False
        self._prompt_evolution_mode = PromptEvolutionMode.DISABLED
        self._meta_prompting_manager: Optional["MetaPromptingManager"] = None
        self._pending_prompt_sessions: Dict[str, Dict[str, Any]] = {}

        # Check if prompt config exists
        prompt_config = getattr(self.config, "prompt", None)
        if prompt_config is None:
            logger.info("Prompt config not found, using static prompts")
            return

        # Check meta-prompting configuration
        meta_prompting_config = getattr(prompt_config, "meta_prompting", None)

        # Determine evolution mode
        if meta_prompting_config and getattr(meta_prompting_config, "enabled", False):
            mode = getattr(meta_prompting_config, "mode", PromptEvolutionMode.HOLISTIC)
            self._prompt_evolution_mode = mode
            self._prompt_evolution_enabled = True
        else:
            logger.info("Prompt evolution not enabled")
            return

        # Convert config to dict for managers
        prompt_config_dict: Dict[str, Any] = {}
        if hasattr(prompt_config, "_content"):
            container = OmegaConf.to_container(prompt_config, resolve=True)
            if isinstance(container, dict):
                prompt_config_dict = {str(k): v for k, v in container.items()}
        elif isinstance(prompt_config, dict):
            prompt_config_dict = {str(k): v for k, v in prompt_config.items()}

        # Check if ESIMD exploration is enabled (from global config)
        enable_esimd = getattr(self.config, "enable_esimd_exploration", False)

        # Get output directory
        logdir = getattr(self.config, "logdir", "runs")
        output_subdir = "prompt_evolution"
        if meta_prompting_config:
            output_subdir = getattr(meta_prompting_config, "output_subdir", "prompt_evolution")
        output_dir_general = os.path.join(logdir, output_subdir)
        os.makedirs(output_dir_general, exist_ok=True)
        output_dir = os.path.join(output_dir_general, self.config.get("task_name", "unknown_op"))
        os.makedirs(output_dir, exist_ok=True)

        # Get language from config
        language = getattr(self.config, "language", "SYCL")

        # Initialize managers based on mode
        if self._prompt_evolution_mode == PromptEvolutionMode.HOLISTIC:
            meta_config_dict: Dict[str, Any] = {}
            if meta_prompting_config is not None:
                if hasattr(meta_prompting_config, "_content"):
                    container = OmegaConf.to_container(meta_prompting_config, resolve=True)
                    if isinstance(container, dict):
                        meta_config_dict = {str(k): v for k, v in container.items()}
                elif isinstance(meta_prompting_config, dict):
                    meta_config_dict = {str(k): v for k, v in meta_prompting_config.items()}

            self._init_meta_prompting_manager(
                config=meta_config_dict,
                output_dir=output_dir,
                language=language,
            )
            logger.info(f"Meta-prompting (holistic) initialized: mode={self._prompt_evolution_mode}")

        print(f"Prompt evolution enabled: mode={self._prompt_evolution_mode}")

    def _init_meta_prompting_manager(
        self,
        config: Dict[str, Any],
        output_dir: str,
        language: str,
    ):
        """Initialize the meta-prompting manager."""
        mp = _get_meta_prompting_module()

        self._meta_prompting_manager = mp.create_meta_prompting_manager(
            config=config,
            output_dir=output_dir,
            llm_server=self.llm_server,
            language=language,
        )

    @property
    def prompt_evolution_enabled(self) -> bool:
        """Check if any prompt evolution is enabled."""
        return getattr(self, "_prompt_evolution_enabled", False)

    @property
    def prompt_evolution_mode(self) -> str:
        """Get current prompt evolution mode."""
        return getattr(self, "_prompt_evolution_mode", PromptEvolutionMode.DISABLED)

    @property
    def meta_prompting_enabled(self) -> bool:
        """Check if meta-prompting (holistic) is enabled."""
        return self._prompt_evolution_mode == PromptEvolutionMode.HOLISTIC

    def sample_evolved_prompt(
        self,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Optional["PromptProgram"], str]:
        """
        Sample an evolved prompt for kernel generation (holistic/meta-prompting).

        Args:
            rng: Optional random generator for reproducibility

        Returns:
            Tuple of (PromptProgram, session_id for tracking)
        """
        if not self.meta_prompting_enabled or self._meta_prompting_manager is None:
            return None, ""

        return self._meta_prompting_manager.sample_prompt(rng=rng)

    def get_evolvable_content(
        self,
        prompt_program: Optional["PromptProgram"] = None,
    ) -> Dict[str, str]:
        """
        Get evolvable content from a prompt program for template injection.

        Args:
            prompt_program: PromptProgram from meta-prompting (optional)

        Returns:
            Dict of region_name -> content for Jinja2 template
        """
        if prompt_program is None:
            if self._meta_prompting_manager is None:
                return {}
            # Get from best prompt
            best_prompt = self._meta_prompting_manager.database.get_best_prompt()
            if best_prompt is None:
                return {}
            return best_prompt.template_overrides

        return prompt_program.template_overrides

    def report_prompt_fitness(
        self,
        session_id: str,
        kernel_score: float,
        kernel_id: Optional[str] = None,
        kernel_code: Optional[str] = None,
        kernel_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Report kernel fitness to update associated prompt evolution state.

        Call this after kernel evaluation.

        Args:
            session_id: Session ID from sample_evolved_prompt
            kernel_score: Combined fitness score of the generated kernel
            kernel_id: Optional ID of the kernel
            kernel_code: Optional code of the kernel (for meta-prompting context)
            kernel_metrics: Optional detailed metrics for fine-grained attribution
        """
        if not self.prompt_evolution_enabled or not session_id:
            return

        # Holistic/meta-prompting session
        if self._meta_prompting_manager is not None:
            self._meta_prompting_manager.report_fitness(
                session_id=session_id,
                kernel_score=kernel_score,
                kernel_id=kernel_id,
                kernel_code=kernel_code,
                kernel_metrics=kernel_metrics,
            )
            logger.debug(f"Reported fitness {kernel_score:.4f} to meta-prompting (session {session_id[:12]})")

    def get_prompt_evolution_statistics(self) -> Dict[str, Any]:
        """Get combined statistics about prompt evolution."""
        stats = {
            "enabled": self.prompt_evolution_enabled,
            "mode": self._prompt_evolution_mode,
        }

        if self._meta_prompting_manager is not None:
            stats["meta_prompting"] = self._meta_prompting_manager.get_statistics()

        return stats

    def save_prompt_evolution_state(self):
        """Save the prompt evolution database(s)."""
        self._meta_prompting_manager.save()
        logger.info("Saved meta-prompting state")

    def apply_evolved_prompting(
        self,
        base_prompt: str,
        parent: Optional["Program"],
        strategy: str = "mutate",
        include_esimd: bool = False,
        skip_meta_prompting: bool = True,
    ) -> Tuple[str, str]:
        """
        Enhanced version of prompt construction with evolution.

        This method extends the optimization-aware prompting by incorporating
        evolved prompt components from holistic systems.

        NOTE: When using template-based meta-prompting (evolvable content injected
        during prompt construction), set skip_meta_prompting=True to avoid double-
        sampling. The holistic content is already in the prompt via template variables.

        Args:
            base_prompt: The base prompt before enhancements
            parent: Parent program being evolved
            strategy: Evolution strategy
            include_esimd: Whether to include ESIMD guidance
            skip_meta_prompting: If True, skip holistic evolution (already applied via template)

        Returns:
            Tuple of (enhanced_prompt, session_id)
        """
        if not self.prompt_evolution_enabled:
            return base_prompt, ""

        session_id = ""
        enhanced_prompt = base_prompt

        # Apply holistic evolution only if not already injected via template
        if not skip_meta_prompting and self.meta_prompting_enabled and self._meta_prompting_manager is not None:
            prompt_program, meta_session_id = self.sample_evolved_prompt()
            if prompt_program is not None:
                # Get evolvable content for template
                evolvable_content = prompt_program.template_overrides
                if evolvable_content:
                    # Format evolved content as sections
                    sections = []
                    for region_name, content in evolvable_content.items():
                        if content:
                            sections.append(f"## {region_name.replace('_', ' ').title()}\n{content}")

                    if sections:
                        evolved_section = "\n\n".join(sections)
                        enhanced_prompt = (
                            base_prompt + "\n\n---\n## Evolved Optimization Guidance\n\n" + evolved_section
                        )

                session_id = meta_session_id
                logger.debug(
                    f"Applied holistic evolution: prompt={prompt_program.id[:8]}, session={meta_session_id[:12]}"
                )

        return enhanced_prompt, session_id

    def force_prompt_evolution(self, num_prompts: int = 1) -> int:
        """
        Force immediate prompt evolution.

        Useful for testing or manual triggering.

        Args:
            num_prompts: Number of prompts to evolve

        Returns:
            Number of prompts successfully evolved
        """
        evolved = 0

        if self._meta_prompting_manager is not None:
            new_prompts = self._meta_prompting_manager.force_evolution(num_prompts)
            evolved += len(new_prompts)

        return evolved


def integrate_prompt_evolution_with_controller(controller_class):
    """
    Decorator to add prompt evolution capabilities to a controller class.

    Usage::

        @integrate_prompt_evolution_with_controller
        class Controller:
            ...

    Dynamic usage::

        Controller = integrate_prompt_evolution_with_controller(Controller)
    """
    # Check if already integrated
    if hasattr(controller_class, "_prompt_evolution_integrated"):
        return controller_class

    # Store original __init__
    original_init = controller_class.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Setup prompt evolution after original init
        if hasattr(self, "setup_prompt_evolution"):
            self.setup_prompt_evolution()

    # Add mixin methods
    for attr_name in dir(PromptEvolutionMixin):
        if not attr_name.startswith("_") or attr_name in ["_prompt_evolution_enabled", "_prompt_evolution_manager"]:
            attr = getattr(PromptEvolutionMixin, attr_name)
            if callable(attr) and not hasattr(controller_class, attr_name):
                setattr(controller_class, attr_name, attr)

    controller_class.__init__ = new_init
    controller_class._prompt_evolution_integrated = True

    return controller_class


# =============================================================================
# HELPER FUNCTIONS FOR PROMPT CONSTRUCTION
# =============================================================================


def build_evolved_optimization_prompt(
    base_prompt: str,
    meta_prompting_manager: Optional["MetaPromptingManager"] = None,
    strategy: str = "mutate",
    include_esimd: bool = False,
    parent_profile: Optional[Dict[str, int]] = None,
) -> Tuple[str, str]:
    """
    Build an optimization prompt with evolved components.

    Standalone function for use outside of the controller.

    Args:
        base_prompt: Base prompt text
        meta_prompting_manager: MetaPromptingManager instance (can be None)
        strategy: Evolution strategy
        include_esimd: Whether to include ESIMD guidance
        parent_profile: Optional parent's optimization profile

    Returns:
        Tuple of (enhanced_prompt, session_id)
    """
    enhanced_prompt = base_prompt
    session_id = ""

    # Apply holistic evolution
    if meta_prompting_manager is not None and meta_prompting_manager.enabled:
        prompt_program, meta_session = meta_prompting_manager.sample_prompt()
        if prompt_program is not None:
            content = meta_prompting_manager.get_evolvable_content(prompt_program)
            if content:
                sections = []
                for name, text in content.items():
                    if text:
                        sections.append(f"## {name.replace('_', ' ').title()}\n{text}")
                if sections:
                    enhanced_prompt = base_prompt + "\n\n---\n" + "\n\n".join(sections)
            session_id = meta_session

    # Add profile context if available
    if parent_profile:
        profile_text = (
            f"\nParent optimization levels: "
            f"Memory={parent_profile.get('memory_opt', 0)}, "
            f"Compute={parent_profile.get('compute_opt', 0)}, "
            f"Parallelism={parent_profile.get('parallelism_opt', 0)}"
        )
        if include_esimd:
            profile_text += f", ESIMD={parent_profile.get('esimd_opt', 0)}"
        enhanced_prompt = enhanced_prompt + profile_text

    return enhanced_prompt, session_id


def create_prompt_evolution_callback(
    meta_manager: Optional["MetaPromptingManager"] = None,
) -> Callable[[str, float, Optional[Dict]], None]:
    """
    Create a callback function for reporting kernel fitness.

    Useful for async or callback-based evaluation pipelines.

    Args:
        meta_manager: MetaPromptingManager instance

    Returns:
        Callback function(session_id, score, metrics) -> None
    """

    def callback(
        session_id: str,
        kernel_score: float,
        kernel_metrics: Optional[Dict[str, float]] = None,
        kernel_code: Optional[str] = None,
    ):
        if not session_id:
            return

        meta_manager.report_fitness(
            session_id=session_id,
            kernel_score=kernel_score,
            kernel_code=kernel_code,
            kernel_metrics=kernel_metrics,
        )

    return callback
