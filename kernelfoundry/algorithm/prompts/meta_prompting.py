"""
Meta-Prompting database and manager for evolving parts of the prompt.

This module implements Science-CodeEvolve style meta-prompting where prompts
evolve as complete programs rather than as modular genes.

**Architecture**

- PromptProgram: A complete prompt with evolvable regions
- HolisticPromptDatabase: MAP-Elites style database for prompt diversity
- MetaPrompter: LLM-based prompt mutation via SEARCH/REPLACE diffs
- MetaPromptingManager: Integration layer with the kernel generation pipeline

**Evolution strategy**

1. Prompts contain designated evolvable regions (PROMPT-BLOCK-START/END markers)
2. An auxiliary LLM (meta-prompter) generates SEARCH/REPLACE diffs to evolve prompts
3. Prompt fitness equals the best solution fitness it ever generated
4. Selection uses same strategies as programs (roulette, tournament, etc.)

**Key difference from gene-based evolution**

- Gene-based: Prompts assembled from independent components
- Holistic: Prompts evolve as coherent units with internal structure
- Both approaches are complementary and can be used together

**Thread safety**

- All public methods acquire locks before accessing shared state
- Database operations are atomic

Inspired by: Science-CodeEvolve (https://github.com/inter-co/science-codeevolve)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from threading import RLock, Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from jinja2 import Environment, PackageLoader, select_autoescape

# =============================================================================
# CONSTANTS
# =============================================================================


# Markers for evolvable regions in prompts
PROMPT_BLOCK_START = "# PROMPT-BLOCK-START"
PROMPT_BLOCK_END = "# PROMPT-BLOCK-END"

# Alternative markers for inline evolution (less intrusive)
EVOLVE_START = "{{EVOLVE:"
EVOLVE_END = "}}"


class PromptEvolutionMode(Enum):
    """Modes for prompt evolution."""

    HOLISTIC = auto()  # Science-CodeEvolve style - full prompt evolution
    GENE_BASED = auto()  # Original KernelGen style - modular genes
    HYBRID = auto()  # Both: holistic for structure, genes for components


class MetaPromptStrategy(Enum):
    """Strategies for meta-prompting."""

    IMPROVE = auto()  # Make prompt more effective based on results
    SPECIALIZE = auto()  # Specialize for specific optimization patterns
    GENERALIZE = auto()  # Make prompt more broadly applicable
    SIMPLIFY = auto()  # Reduce prompt complexity
    ELABORATE = auto()  # Add more detailed guidance


# =============================================================================
# META-PROMPTING TEMPLATES
# =============================================================================


META_PROMPT_SYSTEM_TEMPLATE_FILE = "meta_prompting_system.j2"
META_PROMPT_USER_TEMPLATE_FILE = "meta_prompting_user.j2"


# =============================================================================
# PROMPT PROGRAM DATA STRUCTURE
# =============================================================================


@dataclass
class PromptProgram:
    """
    A complete prompt that can evolve like a program.

    Unlike gene-based prompts where components are assembled, this represents
    the full prompt as a coherent unit with designated evolvable regions.

    """

    id: str
    """Unique identifier for this prompt"""
    template_overrides: Dict[str, str] = field(default_factory=dict)
    """Dict mapping region_name -> evolved text"""
    base_template_name: str = "main_prompt.j2"
    """Name of the Jinja2 template this is based on"""
    fitness: float = 0.0
    """Best fitness score from kernels generated with this prompt"""
    generation: int = 0
    """Evolution generation (0 = seed)"""
    parent_id: Optional[str] = None
    """ID of parent prompt if evolved"""
    child_scores: List[Tuple[float, float]] = field(default_factory=list)  # (score, timestamp)
    """History of child kernel scores for selection"""
    usage_count: int = 0
    """Number of times this prompt was used"""
    creation_time: float = field(default_factory=time.time)
    """Timestamp of prompt creation"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional tracking information"""

    # Link to best child solution (for meta-prompting context)
    best_child_id: Optional[str] = None
    best_child_code: Optional[str] = None
    best_child_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    @property
    def content_hash(self) -> str:
        """Generate a hash of the evolvable content for deduplication."""
        content = json.dumps(self.template_overrides, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def mean_child_score(self) -> float:
        """Mean score of children generated with this prompt."""
        if not self.child_scores:
            return 0.0
        return sum(s for s, _ in self.child_scores) / len(self.child_scores)

    def update_fitness(
        self,
        child_score: float,
        child_id: Optional[str] = None,
        child_code: Optional[str] = None,
        child_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Update prompt fitness based on child solution performance.

        Following Science-CodeEvolve: prompt.fitness = max(child_scores)
        """
        timestamp = time.time()
        self.child_scores.append((child_score, timestamp))
        self.usage_count += 1

        # Update fitness if this child is better
        logging.info("Updating fitness for prompt: child_score=%.4f, current_fitness=%.4f", child_score, self.fitness)
        assert child_code is not None
        if child_score > self.fitness:
            self.fitness = child_score
            self.best_child_id = child_id
            if child_code:
                # Store truncated code for context (avoid memory bloat)
                self.best_child_code = child_code[:5000] if len(child_code) > 5000 else child_code
            if child_metrics:
                self.best_child_metrics = child_metrics

        # Trim history to prevent unbounded growth
        max_history = 100
        if len(self.child_scores) > max_history:
            self.child_scores = self.child_scores[-max_history:]

    def get_evolvable_content(self) -> str:
        """Get the combined evolvable content for meta-prompting display."""
        if not self.template_overrides:
            return ""

        sections = []
        for region_name, content in sorted(self.template_overrides.items()):
            sections.append(f"[{region_name}]\n{content}")

        return "\n\n".join(sections)

    def apply_diff(self, diff: str) -> Optional["PromptProgram"]:
        """
        Apply a SEARCH/REPLACE diff to create a child prompt.

        Supports two formats:
        1. With section name: <<<<<<< SEARCH [section_name]
        2. Without section name: <<<<<<< SEARCH (searches all sections)

        Args:
            diff: The diff text from meta-prompting

        Returns:
            New PromptProgram if diff applied successfully, None otherwise
        """
        new_overrides = dict(self.template_overrides)

        # Parse SEARCH/REPLACE blocks - support optional [section_name] after SEARCH
        # Pattern matches: <<<<<<< SEARCH [section_name]\n or <<<<<<< SEARCH\n
        pattern = r"<<<<<<< SEARCH(?:\s*\[(\w+)\])?\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(pattern, diff, re.DOTALL)

        if not matches:
            logging.warning("No valid SEARCH/REPLACE blocks found in diff")
            return None

        applied = 0
        for section_name, search_text, replace_text in matches:
            search_text = search_text.strip()
            replace_text = replace_text.strip()

            if section_name:
                # Target specific section
                if section_name in new_overrides:
                    if search_text in new_overrides[section_name]:
                        new_overrides[section_name] = new_overrides[section_name].replace(search_text, replace_text, 1)
                        applied += 1
                    else:
                        logging.debug(f"Search text not found in section {section_name}")
            else:
                # Search all sections
                for region_name, content in new_overrides.items():
                    if search_text in content:
                        new_overrides[region_name] = content.replace(search_text, replace_text, 1)
                        applied += 1
                        break

        if applied == 0:
            logging.warning("No SEARCH/REPLACE operations were applied (text not found)")
            return None

        # Create child prompt
        child = PromptProgram(
            id=str(uuid.uuid4()),
            template_overrides=new_overrides,
            base_template_name=self.base_template_name,
            generation=self.generation + 1,
            parent_id=self.id,
            metadata={"diff_applied": True, "num_changes": applied},
        )

        return child

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "id": self.id,
            "template_overrides": self.template_overrides,
            "base_template_name": self.base_template_name,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "child_scores": self.child_scores,
            "usage_count": self.usage_count,
            "creation_time": self.creation_time,
            "metadata": self.metadata,
            "best_child_id": self.best_child_id,
            "best_child_code": self.best_child_code,
            "best_child_metrics": self.best_child_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptProgram":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            template_overrides=data.get("template_overrides", {}),
            base_template_name=data.get("base_template_name", "main_prompt.j2"),
            fitness=data.get("fitness", 0.0),
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            child_scores=[(s, t) for s, t in data.get("child_scores", [])],
            usage_count=data.get("usage_count", 0),
            creation_time=data.get("creation_time", time.time()),
            metadata=data.get("metadata", {}),
            best_child_id=data.get("best_child_id"),
            best_child_code=data.get("best_child_code"),
            best_child_metrics=data.get("best_child_metrics", {}),
        )


# =============================================================================
# DEFAULT EVOLVABLE REGIONS
# =============================================================================


def get_default_evolvable_regions(language: str = "SYCL") -> Dict[str, str]:
    """
    Get default content for evolvable prompt regions.

    These are the starting points for meta-prompting evolution.

    Args:
        language: Target kernel language (SYCL, CUDA, triton)

    Returns:
        Dict mapping region_name -> default content
    """
    regions_sycl = {
        "optimization_philosophy": f"""When optimizing {language} kernels, follow this philosophy:
1. **Understand before optimizing**: Analyze the reference implementation to identify the core computation pattern.
2. **Memory first**: GPU performance is often memory-bound. Focus on memory access patterns before compute optimizations.
3. **Simplicity when possible**: Complex optimizations can backfire. Prefer clear, maintainable code that the compiler can optimize.""",
        "optimization_strategies": """
- **Coalesced memory access**: Ensure coalesced memory access: adjacent work-items should access adjacent memory locations
- **Shared local memory**: Use local/shared memory to cache data reused within a work-group
- **Vector loads**: Consider vectorized loads/stores (float4, sycl::vec) for bandwidth efficiency
- **Operation fusion**: Fuse operations to reduce kernel launch overhead and improve data locality
- **Loop unrolling**: Unroll small, fixed-size loops to enable instruction-level parallelism
- **FMA operations**: Use FMA (fused multiply-add) operations where applicable
- Choose work-group sizes that maximize occupancy (typically 64-256)
- Use sub-group operations for efficient intra-group communication""",
        "common_pitfalls": """
1. **Bank conflicts**: When using local memory, ensure access patterns don't cause bank conflicts. Consider padding.
2. **Thread divergence**: Avoid conditionals where different work-items take different branches within a sub-group.
3. **Over-synchronization**: Excessive barriers serialize execution. Use only when necessary.
4. **SYCL dimension limit (CRITICAL)**: `sycl::range<N>`, `sycl::nd_range<N>`, `sycl::id<N>`, `sycl::local_accessor<T,N>`, and `sycl::accessor<T,N>` ALL require N ∈ {{1,2,3}}. Using N=4 or higher causes static_assert compilation failures. For 4D+ data (e.g., batch×channel×D×H×W in Conv3d), linearize dimensions: use `sycl::range<1>(total)` or `sycl::range<3>(batch*channel, H, W)`.
5. **Missing kernel names**: Always name kernels explicitly: `parallel_for<class MyKernel>(...)`
6. **Queue synchronization**: Be explicit about when to wait for queue completion.""",
        "analysis_guidance": """
Before writing code, analyze:
1. **Data flow**: What data is read? What is written? What is reused?
2. **Computation pattern**: Is this element-wise, reduction, scan, stencil, matrix operation?
3. **Bottleneck identification**: Will this be memory-bound or compute-bound?
4. **Parallelization strategy**: How should work be divided among work-items and work-groups?

After analysis, explain your optimization strategy before presenting the code.""",
    }

    regions_cuda = {
        "optimization_philosophy": f"""When optimizing {language} kernels, follow this philosophy:
1. **Understand before optimizing**: Analyze the reference implementation to identify the core computation pattern.
2. **Memory first**: GPU performance is often memory-bound. Focus on memory access patterns before compute optimizations.
3. **Simplicity when possible**: Complex optimizations can backfire. Prefer clear, maintainable code that the compiler can optimize.""",
        "optimization_strategies": f"""
- **Coalesced memory access**: adjacent threads should access adjacent memory locations
- **Shared memory usage**: use shared memory to cache data reused within a thread block
- **Minimize global memory transactions**: maximize data reuse
- **Operation fusion**: fuse operations to reduce kernel launch overhead and improve data locality
- **Loop unrolling**: unroll small, fixed-size loops to enable instruction-level parallelism
- **FMA operations**: use FMA (fused multiply-add) ops where applicable
- **Tensor cores**: make use of tensor cores using the WMMA API for matrix ops where applicable
- **Warp-level primitives**: use warp-level primitives for efficient intra-block communication
- **Balanced workload**: ensure balanced workload distribution across threads""",
        "common_pitfalls": """
1. **Bank conflicts**: When using shared memory, ensure access patterns don't cause bank conflicts. Consider padding.
2. **Thread divergence**: Avoid conditionals where different threads within a warp take different branches.
3. **Over-synchronization**: Excessive __syncthreads() calls serialize execution. Use only when necessary.
4. **Grid/block dimension limits**: Blocks are limited to 1024 threads total, and each dimension has specific limits (x: 1024, y: 1024, z: 64).""",
        "analysis_guidance": """
Before writing code, analyze:
1. **Data flow**: What data is read? What is written? What is reused?
2. **Computation pattern**: Is this element-wise, reduction, scan, stencil, matrix operation?
3. **Bottleneck identification**: Will this be memory-bound or compute-bound?
4. **Parallelization strategy**: How should work be divided among threads and thread blocks?

After analysis, explain your optimization strategy before presenting the code.""",
    }

    regions_opencl = {
        "optimization_philosophy": f"""When optimizing {language} kernels, follow this philosophy:
1. **Understand before optimizing**: Analyze the reference implementation to identify the core computation pattern.
2. **Memory first**: GPU performance is often memory-bound. Focus on memory access patterns before compute optimizations.
3. **Simplicity when possible**: Complex optimizations can backfire. Prefer clear, maintainable code that the compiler can optimize.""",
        "optimization_strategies": """
- **Coalesced memory access**: adjacent work-items should access adjacent memory locations in global memory
- **Local memory usage**: use local memory to cache data reused within a work-group
- **Vector loads/stores**: consider native vector types (e.g., float4, half8) for bandwidth efficiency
- **Operation fusion**: fuse operations to reduce kernel launch overhead and improve data locality
- **Loop unrolling**: unroll small, fixed-size loops to enable instruction-level parallelism
- **FMA operations**: use fused multiply-add operations where applicable
- Choose work-group sizes that maximize occupancy and memory throughput
- Use subgroup operations when available for efficient intra-group communication""",
        "common_pitfalls": """
1. **Bank conflicts**: When using local memory, avoid access patterns that cause bank conflicts. Consider padding.
2. **Thread divergence**: Avoid divergent branches within a subgroup/warp where possible.
3. **Over-synchronization**: Excessive barriers serialize execution. Use synchronization only when required.
4. **Address space mistakes**: Ensure correct use of `__global`, `__local`, and `__private` memory spaces.
5. **NDRange mismatch**: Ensure global and local sizes match the kernel indexing logic and boundary checks.
6. **Queue synchronization**: Be explicit about event dependencies and command queue synchronization.""",
        "analysis_guidance": """
Before writing code, analyze:
1. **Data flow**: What data is read? What is written? What is reused?
2. **Computation pattern**: Is this element-wise, reduction, scan, stencil, matrix operation?
3. **Bottleneck identification**: Will this be memory-bound or compute-bound?
4. **Parallelization strategy**: How should work be divided among work-items and work-groups?

After analysis, explain your optimization strategy before presenting the code.""",
    }

    regions_triton = {
        "optimization_philosophy": f"""When optimizing {language} kernels, follow this philosophy:
1. **Understand before optimizing**: Analyze the reference implementation to identify the core computation pattern.
2. **Blocks over scalars**: Triton programs operate on blocks of data, not individual elements. Design your algorithm around block-level operations from the start.
3. **Simplicity when possible**: Triton's JIT compiler handles many low-level details. Prefer clear block operations over manual micro-optimizations.""",
        "optimization_strategies": """
- **Blocked coalesced loads**: Use `tl.load(ptr + offsets, mask=offsets < N, other=0.0)` where `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` for coalesced access
- **Power-of-2 BLOCK_SIZE**: Always use powers of 2 (64, 128, 256, 512, 1024) declared as `tl.constexpr`
- **tl.dot for matrix ops**: Use `tl.dot(a, b)` for matrix multiply — it automatically dispatches to Tensor Cores for float16/bfloat16 inputs
- **Block-level reductions**: Use `tl.max(x, axis=0)`, `tl.sum(x, axis=0)` instead of serial loops
- **Kernel fusion**: Combine element-wise operations (e.g., bias add + activation) in a single kernel to eliminate intermediate global memory writes
- **Software pipelining**: Set `num_stages=3` (or higher) to overlap memory loads with compute for GEMM-like kernels
- **Auto-tuning**: Use `@triton.autotune` with multiple `triton.Config` entries to find optimal `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages`
- **L2 swizzling**: Apply GROUP_SIZE_M reordering for GEMM to improve L2 cache hit rate across program blocks""",
        "common_pitfalls": """
1. **Missing boundary mask**: Always use `mask=offsets < N` and `other=0.0` in `tl.load`/`tl.store` when N is not a multiple of BLOCK_SIZE — unmasked out-of-bounds access causes undefined behavior.
2. **Non-power-of-2 BLOCK_SIZE**: Triton requires power-of-2 block sizes for static shape analysis. Using e.g. BLOCK_SIZE=100 causes errors.
3. **BLOCK_SIZE not constexpr**: All compile-time constants (BLOCK_SIZE, BLOCK_M, BLOCK_N, etc.) must be annotated as `tl.constexpr` in the kernel signature.
4. **Small tl.dot dimensions**: `tl.dot` requires both input dimensions to be divisible by 16. BLOCK_K < 16 causes errors or falls back to scalar path.
5. **Mixed float precision in tl.dot**: Ensure both inputs have the same dtype. Convert with `.to(tl.float16)` before calling `tl.dot`; accumulate in float32.
6. **Forgetting num_warps/num_stages**: Default `num_warps=4, num_stages=2` may be suboptimal. For GEMM, `num_warps=8, num_stages=3` is typically better.
7. **Non-contiguous access patterns**: Strided accesses across programs break memory coalescing. Ensure contiguous block indexing: `pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`.""",
        "analysis_guidance": """
Before writing code, analyze:
1. **Data flow**: What data is read? What is written? What is reused across the block loop?
2. **Computation pattern**: Is this element-wise (1D grid), reduction (per-row), matrix multiply (2D grid + inner K loop), or attention (outer M loop, inner N loop)?
3. **Bottleneck identification**: Memory-bound (maximize BLOCK_SIZE, minimize launches) or compute-bound (maximize arithmetic intensity with tl.dot)?
4. **Grid structure**: 1D grid for element-wise/reductions; 2D grid for GEMM; persistent kernel for streaming workloads.
5. **Block size choice**: For element-wise: 1024; for reductions: 256–512; for GEMM: BLOCK_M=BLOCK_N=128, BLOCK_K=64.

After analysis, explain your tiling strategy (grid shape, block sizes, accumulator layout) before presenting the code.""",
    }

    # Return appropriate regions based on language
    if language.upper() == "CUDA":
        return regions_cuda
    elif language.upper() == "SYCL":
        return regions_sycl
    elif language.upper() == "OCL":
        return regions_opencl
    elif language.upper() == "TRITON":
        return regions_triton
    else:
        raise ValueError(f"Unsupported language: {language}. Supported: CUDA, SYCL, OCL, triton")


# =============================================================================
# HOLISTIC PROMPT DATABASE
# =============================================================================


class HolisticPromptDatabase:
    """
    Database for holistic prompt evolution following Science-CodeEvolve design.

    Key differences from gene-based PromptEvolutionDatabase:
    - Prompts stored as complete units, not assembled from genes
    - Selection based on prompt-level fitness (best child score)
    - Genealogy tracking for evolution history
    - Migration support for island-based evolution

    Thread-safe with proper locking.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        language: str = "SYCL",
    ):
        """
        Initialize the holistic prompt database.

        Args:
            config: Configuration dictionary
            output_dir: Directory for persistence
            language: Target kernel language
        """
        self.config = config or {}
        self.output_dir = output_dir
        self.language = language

        # Database storage
        self._prompts: Dict[str, PromptProgram] = {}
        self._lock = RLock()

        # Selection parameters
        self._exploration_rate = self.config.get("exploration_rate", 0.2)
        self._selection_temperature = self.config.get("selection_temperature", 1.0)

        # Tracking
        self._best_prompt_id: Optional[str] = None
        self._total_selections = 0
        self._generation = 0

        # Session tracking for fitness attribution
        self._active_sessions: Dict[str, str] = {}  # session_id -> prompt_id

        # Initialize with seed prompt
        self._initialize_seed_prompt()

    def _initialize_seed_prompt(self):
        """Initialize with a default seed prompt."""
        seed_regions = get_default_evolvable_regions(self.language)
        seed_prompt = PromptProgram(
            id=str(uuid.uuid4()),
            template_overrides=seed_regions,
            generation=0,
            metadata={"is_seed": True, "language": self.language},
        )
        self.add(seed_prompt)
        self._best_prompt_id = seed_prompt.id
        logging.info(f"Initialized seed prompt: {seed_prompt.id[:8]}...")

    def add(self, prompt: PromptProgram) -> bool:
        """
        Add a prompt to the database.

        Args:
            prompt: PromptProgram to add

        Returns:
            True if added, False if duplicate content
        """
        with self._lock:
            # Check for duplicate content
            content_hash = prompt.content_hash
            for existing in self._prompts.values():
                if existing.content_hash == content_hash:
                    logging.debug(f"Duplicate prompt content, skipping: {prompt.id[:8]}")
                    return False

            self._prompts[prompt.id] = prompt

            # Update best if this prompt has higher fitness
            if self._best_prompt_id is None:
                self._best_prompt_id = prompt.id
            elif prompt.fitness > self._prompts.get(self._best_prompt_id, PromptProgram(id="")).fitness:
                self._best_prompt_id = prompt.id

            logging.debug(f"Added prompt {prompt.id[:8]} (gen={prompt.generation})")
            return True

    def sample(self, rng=None) -> Tuple[PromptProgram, str]:
        """
        Sample a prompt for kernel generation.

        Uses softmax selection with exploration bonus for less-used prompts.

        Args:
            rng: Optional random number generator

        Returns:
            Tuple of (selected prompt, session_id for fitness tracking)
        """
        import random

        if rng is None:
            rng = random.Random()

        with self._lock:
            if not self._prompts:
                raise ValueError("No prompts in database")
            prompts = list(self._prompts.values())
            logging.info("Number of prompts currently in database: %d", len(prompts))
            self._total_selections += 1

            # Exploration: random selection with probability exploration_rate
            if rng.random() < self._exploration_rate:
                selected = rng.choice(prompts)
                logging.info("Sample prompt - exploration (random choice)")
            else:
                # Softmax selection based on fitness with usage penalty
                import math

                scores = []
                for p in prompts:
                    # Fitness score with exploration bonus for under-used prompts
                    usage_penalty = math.log1p(p.usage_count) * 0.1
                    exploration_bonus = 1.0 / (1 + p.usage_count) if p.usage_count < 5 else 0
                    score = p.fitness - usage_penalty + exploration_bonus
                    scores.append(score)
                logging.info("Sample prompt - exploitation (softmax selection)")

                # Softmax
                max_score = max(scores)
                exp_scores = [math.exp((s - max_score) / self._selection_temperature) for s in scores]
                sum_exp = sum(exp_scores)
                probabilities = [e / sum_exp for e in exp_scores]

                # Weighted random choice
                r = rng.random()
                cumulative = 0.0
                selected = prompts[-1]  # Fallback
                for prompt, prob in zip(prompts, probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        selected = prompt
                        break

            # Generate session ID for tracking
            session_id = f"meta_{uuid.uuid4().hex[:12]}"
            self._active_sessions[session_id] = selected.id

            return selected, session_id

    def update_fitness(
        self,
        session_id: str,
        kernel_score: float,
        kernel_id: Optional[str] = None,
        kernel_code: Optional[str] = None,
        kernel_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Update prompt fitness based on generated kernel performance.

        Args:
            session_id: Session ID from sample()
            kernel_score: Fitness score of the generated kernel
            kernel_id: Optional ID of the kernel
            kernel_code: Optional code of the kernel (for meta-prompting context)
            kernel_metrics: Optional detailed metrics
        """
        with self._lock:
            prompt_id = self._active_sessions.pop(session_id, None)
            if prompt_id is None:
                logging.warning(f"Unknown session ID: {session_id}")
                return

            prompt = self._prompts.get(prompt_id)
            if prompt is None:
                logging.warning(f"Prompt not found: {prompt_id}")
                return

            # Update prompt fitness
            prompt.update_fitness(
                child_score=kernel_score,
                child_id=kernel_id,
                child_code=kernel_code,
                child_metrics=kernel_metrics,
            )

            # Update global best
            if self._best_prompt_id is not None:
                best = self._prompts.get(self._best_prompt_id)
                if best is None or prompt.fitness > best.fitness:
                    self._best_prompt_id = prompt.id
            else:
                self._best_prompt_id = prompt.id

            logging.debug(f"Updated prompt {prompt_id[:8]} fitness: {prompt.fitness:.4f}")

    def get_best_prompt(self) -> Optional[PromptProgram]:
        """Get the prompt with highest fitness."""
        with self._lock:
            if self._best_prompt_id:
                return self._prompts.get(self._best_prompt_id)
            return None

    def get_prompt_for_evolution(self) -> Optional[PromptProgram]:
        """
        Get a prompt suitable for meta-prompting evolution.

        Selects a prompt that:
        1. Has been used enough to have reliable fitness estimates
        2. Has stored child code for context
        3. Balances exploitation (high fitness) with exploration (generation diversity)
        """
        with self._lock:
            candidates = [p for p in self._prompts.values() if p.usage_count >= 3 and p.best_child_code is not None]

            if not candidates:
                # Fall back to any prompt with child code
                candidates = [p for p in self._prompts.values() if p.best_child_code]

            if not candidates:
                # Fall back to best prompt
                return self.get_best_prompt()

            # Select based on fitness with diversity bonus
            import random

            weights = []
            for p in candidates:
                # Prefer high-fitness prompts but also explore different generations
                weight = p.fitness * (1 + 0.1 * p.generation)
                weights.append(max(weight, 0.01))

            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            return random.choices(candidates, weights=probabilities, k=1)[0]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            if not self._prompts:
                return {"size": 0}

            prompts = list(self._prompts.values())
            fitness_values = [p.fitness for p in prompts]
            generations = [p.generation for p in prompts]

            return {
                "size": len(prompts),
                "total_selections": self._total_selections,
                "mean_fitness": sum(fitness_values) / len(fitness_values),
                "max_fitness": max(fitness_values),
                "min_fitness": min(fitness_values),
                "max_generation": max(generations),
                "best_prompt_id": self._best_prompt_id,
                "active_sessions": len(self._active_sessions),
            }

    def save(self, filepath: Optional[str] = None):
        """Save database to file."""
        if filepath is None:
            if self.output_dir is None:
                logging.warning("No output directory specified, cannot save")
                return
            filepath = os.path.join(self.output_dir, "holistic_prompt_db.json")

        with self._lock:
            data = {
                "config": self.config,
                "language": self.language,
                "best_prompt_id": self._best_prompt_id,
                "total_selections": self._total_selections,
                "generation": self._generation,
                "prompts": [p.to_dict() for p in self._prompts.values()],
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved holistic prompt database to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "HolisticPromptDatabase":
        """Load database from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        db = cls(
            config=data.get("config", {}),
            output_dir=os.path.dirname(filepath),
            language=data.get("language", "SYCL"),
        )

        # Clear seed prompt and load saved prompts
        db._prompts.clear()
        for prompt_data in data.get("prompts", []):
            prompt = PromptProgram.from_dict(prompt_data)
            db._prompts[prompt.id] = prompt

        db._best_prompt_id = data.get("best_prompt_id")
        db._total_selections = data.get("total_selections", 0)
        db._generation = data.get("generation", 0)

        logging.info(f"Loaded holistic prompt database from {filepath}: {len(db._prompts)} prompts")
        return db


# =============================================================================
# META-PROMPTER
# =============================================================================


class MetaPrompter:
    """
    LLM-based prompt evolution via SEARCH/REPLACE diffs.

    Uses an auxiliary LLM to analyze prompt effectiveness and suggest improvements
    based on the kernels generated and their performance.
    """

    def __init__(
        self,
        llm_server: Callable,
        language: str = "SYCL",
        max_retries: int = 2,
        timeout: float = 60.0,
    ):
        """
        Initialize the meta-prompter.

        Args:
            llm_server: LLM inference function (same interface as Controller.llm_server)
            language: Target kernel language
            max_retries: Maximum retries on failure
            timeout: Timeout for LLM calls
        """
        self.llm_server = llm_server
        self.language = language
        self.max_retries = max_retries
        self.timeout = timeout

        env = Environment(loader=PackageLoader("kernelfoundry.algorithm.prompts"), autoescape=select_autoescape())
        self._system_template = env.get_template(META_PROMPT_SYSTEM_TEMPLATE_FILE)
        self._user_template = env.get_template(META_PROMPT_USER_TEMPLATE_FILE)

        logging.info("INITIALIZE META PROMPTER")

    def _format_prompt_sections(self, evolvable_content: Dict[str, str]) -> str:
        """
        Format evolvable content sections for clear presentation to the meta-prompter.

        Args:
            evolvable_content: Dict mapping section_name -> content

        Returns:
            Formatted string with each section clearly labeled
        """
        sections = []
        section_order = ["optimization_philosophy", "optimization_strategies", "common_pitfalls", "analysis_guidance"]

        for section_name in section_order:
            content = evolvable_content.get(section_name, "")
            if content:
                sections.append(f"[{section_name}]\n{content}")

        return "\n\n".join(sections)

    def evolve_prompt(
        self,
        prompt: PromptProgram,
        strategy: MetaPromptStrategy = MetaPromptStrategy.IMPROVE,
    ) -> Optional[PromptProgram]:
        """
        Evolve a prompt using meta-prompting.

        Args:
            prompt: The prompt to evolve
            strategy: Evolution strategy to apply

        Returns:
            New evolved PromptProgram, or None if evolution failed
        """
        assert prompt.best_child_code is not None

        # Get evolvable content from template_overrides
        evolvable_content = prompt.template_overrides
        if not evolvable_content:
            logging.warning("Prompt has no evolvable content")
            return None

        # Format evolvable sections for clear presentation
        prompt_sections = self._format_prompt_sections(evolvable_content)

        # Format metrics for display
        metrics = prompt.best_child_metrics or {}
        correctness = "✓ Correct" if metrics.get("correctness_success", False) else "✗ Incorrect"
        runtime = f"{metrics.get('runtime', 0):.4f}ms" if "runtime" in metrics else "N/A"
        speedup = f"{metrics.get('speedup', 0):.2f}x" if "speedup" in metrics else "N/A"
        compiled = "✓ Yes" if metrics.get("compiles", True) else "✗ No"
        error_msg = metrics.get("error", "None")

        user_content = self._user_template.render(
            prompt_sections=prompt_sections,
            language=self.language,
            kernel_code=prompt.best_child_code,
            correctness=correctness,
            runtime=runtime,
            speedup=speedup,
            compiled=compiled,
            error_msg=error_msg,
        )

        # Add strategy-specific guidance
        strategy_guidance = self._get_strategy_guidance(strategy)
        if strategy_guidance:
            user_content += f"\n\n{strategy_guidance}"

        messages = [
            {"role": "system", "content": self._system_template.render()},
            {"role": "user", "content": user_content},
        ]

        # Call LLM
        for attempt in range(self.max_retries + 1):
            try:
                logging.info("[MetaPrompter] CALLING META-PROMPTING LLM")
                response, _ = self.llm_server(messages=messages)
                # DEBUG: save messages
                # test = messages.copy()
                # test.append({"role": "assistant", "content": response})
                # with open("meta_prompting_debug.json", "w") as f:
                #     json.dump(test, f, indent=2)
                if response and len(response) > 0:
                    diff_text = response[0]
                    break
            except Exception as e:
                logging.exception(f"Meta-prompting attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    return None
                continue
        else:
            return None

        # Apply diff to create child prompt
        child_prompt = prompt.apply_diff(diff_text)

        if child_prompt:
            child_prompt.metadata["evolution_strategy"] = strategy.name
            logging.info(f"Evolved prompt {prompt.id[:8]} -> {child_prompt.id[:8]} (gen {child_prompt.generation})")

        return child_prompt

    def _get_strategy_guidance(self, strategy: MetaPromptStrategy) -> str:
        """Get additional guidance based on evolution strategy."""
        guidance = {
            MetaPromptStrategy.IMPROVE: "Focus on making the prompt more effective at generating correct and fast kernels.",
            MetaPromptStrategy.SPECIALIZE: "Specialize the prompt for the specific optimization patterns observed in the kernel.",
            MetaPromptStrategy.GENERALIZE: "Make the prompt more broadly applicable to different types of kernels.",
            MetaPromptStrategy.SIMPLIFY: "Simplify the prompt by removing redundant or ineffective guidance.",
            MetaPromptStrategy.ELABORATE: "Add more detailed and specific optimization guidance to the prompt.",
        }
        return guidance.get(strategy, "")


# =============================================================================
# META-PROMPTING MANAGER
# =============================================================================


class MetaPromptingManager:
    """
    High-level manager for meta-prompting integration with kernel generation.

    Coordinates:
    - Prompt selection for kernel generation
    - Fitness tracking and attribution
    - Periodic prompt evolution via meta-prompting
    - Persistence of prompt database

    This is the main integration point for the controller.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        llm_server: Optional[Callable] = None,
        language: str = "SYCL",
        load_database: bool = False,
    ):
        """
        Initialize the meta-prompting manager.

        Args:
            config: Configuration dictionary with meta-prompting settings
            output_dir: Directory for persistence
            llm_server: LLM server for meta-prompting (optional, can be set later)
            language: Target kernel language
        """
        self.config = config
        self.output_dir = output_dir
        self.language = language
        self._llm_server = llm_server
        self.load_database = load_database

        # Configuration
        self.enabled = config.get("enabled", False)
        self.evolution_interval = config.get("evolution_interval", 10)
        self.evolution_batch_size = config.get("evolution_batch_size", 2)
        self.min_samples_for_evolution = config.get("min_samples_for_evolution", 5)

        # Initialize database
        db_path = os.path.join(output_dir, "holistic_prompt_db.json")
        if self.load_database and os.path.exists(db_path):
            self._database = HolisticPromptDatabase.load(db_path)
            logging.info(f"Loaded existing prompt database from {db_path}")
        else:
            self._database = HolisticPromptDatabase(
                config=config,
                output_dir=output_dir,
                language=language,
            )

        # Initialize meta-prompter (deferred if no LLM server)
        self._meta_prompter: Optional[MetaPrompter] = None
        if llm_server:
            self._meta_prompter = MetaPrompter(
                llm_server=llm_server,
                language=language,
            )

        # Evolution tracking
        self._samples_since_evolution = 0
        self._evolution_lock = Lock()

    def set_llm_server(self, llm_server: Callable):
        """Set the LLM server for meta-prompting."""
        self._llm_server = llm_server
        self._meta_prompter = MetaPrompter(
            llm_server=llm_server,
            language=self.language,
        )

    @property
    def database(self) -> HolisticPromptDatabase:
        """Access the prompt database."""
        return self._database

    def sample_prompt(self, rng=None) -> Tuple[Optional[PromptProgram], str]:
        """
        Sample a prompt for kernel generation.

        Args:
            rng: Optional random number generator

        Returns:
            Tuple of (prompt, session_id)
        """
        if not self.enabled:
            # Return default prompt if disabled
            prompt = self._database.get_best_prompt()
            return prompt, ""

        prompt, session_id = self._database.sample(rng=rng)
        self._samples_since_evolution += 1

        return prompt, session_id

    def get_evolvable_content(self, prompt: PromptProgram) -> Dict[str, str]:
        """
        Get the evolvable content from a prompt.

        Returns dict that can be passed to Jinja2 template.
        """
        return prompt.template_overrides

    def report_fitness(
        self,
        session_id: str,
        kernel_score: float,
        kernel_id: Optional[str] = None,
        kernel_code: Optional[str] = None,
        kernel_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Report kernel fitness for prompt evolution.

        Args:
            session_id: Session ID from sample_prompt()
            kernel_score: Fitness score of the generated kernel
            kernel_id: Optional ID of the kernel
            kernel_code: Optional code of the kernel
            kernel_metrics: Optional detailed metrics
        """
        if not self.enabled or not session_id:
            return

        self._database.update_fitness(
            session_id=session_id,
            kernel_score=kernel_score,
            kernel_id=kernel_id,
            kernel_code=kernel_code,
            kernel_metrics=kernel_metrics,
        )

        # Check if we should trigger evolution
        self._maybe_evolve()

    def _maybe_evolve(self):
        """Check if we should evolve prompts and do so if needed."""
        if not self.enabled or self._meta_prompter is None:
            return

        if self._samples_since_evolution < self.evolution_interval:
            return

        with self._evolution_lock:
            # Double-check after acquiring lock
            if self._samples_since_evolution < self.evolution_interval:
                return

            self._samples_since_evolution = 0
            self._trigger_evolution()

    def _trigger_evolution(self):
        """Trigger meta-prompting evolution."""
        logging.info("Triggering meta-prompting evolution...")
        evolved_count = 0

        # Check that meta-prompter is available
        if self._meta_prompter is None:
            logging.warning("Cannot evolve prompts: no LLM server configured")
            return

        for _ in range(self.evolution_batch_size):
            # Select prompt for evolution
            parent = self._database.get_prompt_for_evolution()
            if parent is None:
                break

            if parent.usage_count < self.min_samples_for_evolution:
                logging.info("Skipping evolution for parent with low usage count")
                continue

            # Evolve the prompt
            strategy = self._select_evolution_strategy(parent)
            child = self._meta_prompter.evolve_prompt(parent, strategy=strategy)

            if child:
                self._database.add(child)
                evolved_count += 1

        if evolved_count > 0:
            logging.info(f"Evolved {evolved_count} prompts")
            self.save()

    def _select_evolution_strategy(self, prompt: PromptProgram) -> MetaPromptStrategy:
        """Select evolution strategy based on prompt characteristics."""
        import random

        # If prompt has low fitness, try to improve
        best_fitness = self._database.get_best_prompt()
        if best_fitness and prompt.fitness < best_fitness.fitness * 0.8:
            return MetaPromptStrategy.IMPROVE

        # If prompt is successful, try to elaborate or specialize
        if prompt.best_child_metrics and prompt.best_child_metrics.get("correctness_success"):
            return random.choice([MetaPromptStrategy.ELABORATE, MetaPromptStrategy.SPECIALIZE])

        # Otherwise, random strategy
        return random.choice(list(MetaPromptStrategy))

    def force_evolution(self, num_prompts: int = 1) -> List[PromptProgram]:
        """
        Force immediate evolution of prompts.

        Useful for manual triggering or testing.

        Args:
            num_prompts: Number of prompts to evolve

        Returns:
            List of newly created prompts
        """
        if self._meta_prompter is None:
            logging.warning("Cannot evolve: no LLM server configured")
            return []

        evolved = []
        for _ in range(num_prompts):
            parent = self._database.get_prompt_for_evolution()
            if parent is None:
                break

            strategy = self._select_evolution_strategy(parent)
            child = self._meta_prompter.evolve_prompt(parent, strategy=strategy)

            if child:
                self._database.add(child)
                evolved.append(child)

        if evolved:
            self.save()

        return evolved

    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-prompting statistics."""
        stats = self._database.get_statistics()
        stats["enabled"] = self.enabled
        stats["samples_since_evolution"] = self._samples_since_evolution
        stats["evolution_interval"] = self.evolution_interval
        return stats

    def save(self):
        """Save prompt database."""
        self._database.save()

    def load(self, filepath: Optional[str] = None):
        """Load prompt database from file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, "holistic_prompt_db.json")

        if os.path.exists(filepath):
            self._database = HolisticPromptDatabase.load(filepath)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_meta_prompting_manager(
    config: Dict[str, Any],
    output_dir: str,
    llm_server: Optional[Callable] = None,
    language: str = "SYCL",
) -> MetaPromptingManager:
    """
    Factory function to create a MetaPromptingManager.

    Args:
        config: Configuration dictionary
        output_dir: Directory for persistence
        llm_server: Optional LLM server for meta-prompting
        language: Target kernel language

    Returns:
        Configured MetaPromptingManager instance
    """
    os.makedirs(output_dir, exist_ok=True)

    return MetaPromptingManager(
        config=config,
        output_dir=output_dir,
        llm_server=llm_server,
        language=language,
    )
