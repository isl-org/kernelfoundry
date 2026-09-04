"""
Production MAP-Elites Database for Optimization-Aware Kernel Evolution

This implementation is specialized for optimization-based behavioral features:
- memory_opt (0-3): Memory hierarchy exploitation level
- compute_opt (0-3): Algorithmic efficiency level
- parallelism_opt (0-3): Parallelism granularity level

**Architectural Design**

- Feature coordinates are DISCRETE (0-3) extracted from code patterns
- No normalization or statistical scaling needed
- Deterministic coordinate assignment (same code → same coordinates)
- 4x4x4 grid = 64 possible behavioral niches
- Elite selection: Best performer in each occupied niche

Performance metrics (runtime, bandwidth, etc.) are used for:
- FITNESS SCORING within cells (which program is the elite)
- NOT for grid coordinates (which cell a program goes into)

**Thread Safety**

- All public methods acquire _population_lock before accessing shared state
- Internal methods (prefixed with _) assume lock is already held by caller
- Cache invalidation is atomic via _archive_cache_lock
"""

import json
import logging
import os
import random
import re
import pandas as pd
import time
import hashlib
import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from threading import RLock, Lock
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet, NamedTuple
import uuid
import numpy as np

from omegaconf import DictConfig

from kernelfoundry.algorithm.schemas import EvalResult, Program
from kernelfoundry.algorithm.utils.load_from_db import load_database
from kernelfoundry.eval_pipeline.utils.gpu_specs import ARCH_TO_NAME
from kernelfoundry.algorithm.utils.map_elites_patterns import (
    MEMORY_OPT_PATTERNS,
    COMPUTE_OPT_PATTERNS,
    PARALLELISM_OPT_PATTERNS,
    CUDA_MEMORY_OPT_PATTERNS,
    CUDA_COMPUTE_OPT_PATTERNS,
    CUDA_PARALLELISM_OPT_PATTERNS,
    OPENCL_MEMORY_OPT_PATTERNS,
    OPENCL_COMPUTE_OPT_PATTERNS,
    OPENCL_PARALLELISM_OPT_PATTERNS,
    ESIMD_OPT_PATTERNS,
    TRITON_MEMORY_OPT_PATTERNS,
    TRITON_COMPUTE_OPT_PATTERNS,
    TRITON_PARALLELISM_OPT_PATTERNS,
)

# Import QD gradient module for transition tracking (optional, gracefully degrades)
HAS_QD_GRADIENT = False
_TransitionTracker = None
_GradientEstimator = None
try:
    from kernelfoundry.algorithm import qd_gradient as _qd_gradient_module

    _TransitionTracker = _qd_gradient_module.TransitionTracker
    _GradientEstimator = _qd_gradient_module.GradientEstimator
    HAS_QD_GRADIENT = True
except ImportError:
    pass


# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================


class SamplingMethod(Enum):
    """Sampling strategies for program selection."""

    FITNESS_PROPORTIONAL = auto()
    UNIFORM = auto()
    ELITE = auto()
    ARCHIVE = auto()  # Alias for ELITE


class ExplorationStrategy(Enum):
    """Strategies for identifying underexplored regions."""

    EMPTY_FIRST = auto()
    LOW_QUALITY = auto()
    BALANCED = auto()


class ProgramScoreEntry(NamedTuple):
    """Entry for the score heap (min-heap, so we negate scores for max behavior)."""

    neg_score: float  # Negated score for min-heap to act as max-heap
    program_id: str


# ============================================================================
# DETERMINISTIC HASHING
# ============================================================================


def deterministic_code_hash(code: str) -> int:
    """
    Compute a deterministic hash of code that is stable across Python runs.

    Unlike Python's built-in hash() which is randomized by PYTHONHASHSEED,
    this function uses SHA256 to ensure identical code produces identical hashes
    across different Python processes. Critical for reproducible diversity selection.

    Args:
        code: Source code string to hash

    Returns:
        Integer hash value (stable across runs)
    """
    hash_bytes = hashlib.sha256(code.encode("utf-8")).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)
    return hash_int & 0x7FFFFFFFFFFFFFFF  # Clear sign bit for positive values


# ============================================================================
# SCORE CALCULATION
# ============================================================================

#: What the integer part of a combined score means, from EvalResult.perf_score. The score is that
#: ladder plus a runtime term, so anything below 2.0 describes a kernel that never ran.
_PERF_SCORE_MEANINGS = {
    0: "syntax error",
    1: "did not compile",
    2: "compiled but failed at runtime",
    3: "shape mismatch",
    4: "value mismatch",
}


def describe_score(score: float) -> str:
    """Spell out a score that encodes a failure, as a parenthetical for a log line.
    Returns an empty string for a passing kernel, where the number speaks for itself.
    """
    meaning = _PERF_SCORE_MEANINGS.get(int(score))
    return f" -- {meaning}" if meaning else ""


def get_main_score(metrics: Dict[str, Any]) -> float:
    """
    Get the main fitness score from metrics dictionary.

    This is used for elite selection WITHIN cells, not for coordinate assignment.
    Requires 'combined_score' to be present in metrics.

    Args:
        metrics: Dictionary of performance metrics

    Returns:
        Fitness score (higher is better)

    Raises:
        ValueError: If metrics is empty or combined_score is missing
    """
    if not metrics:
        raise ValueError("Metrics dictionary cannot be empty or None - indicates missing evaluation results")
    if "combined_score" not in metrics:
        raise ValueError(
            f"combined_score not found in metrics: {list(metrics.keys())}. Program may not have been evaluated."
        )
    return metrics["combined_score"]


# ============================================================================
# OPTIMIZATION FEATURE CLASSIFICATION
# ============================================================================


class OptimizationFeatureClassifier:
    """
    Classifies kernel optimization levels using static code pattern analysis.

    This is the ONLY source of feature coordinates - deterministic and execution-independent.
    Extracts discrete optimization levels (0-3) from code patterns.

    **Classification Philosophy**

    - Each level builds on the previous (Level 2 typically implies Level 1 patterns)
    - Classification uses weighted pattern matching with confidence scores
    - Patterns are grouped by category to avoid double-counting
    - Comments are handled separately to reduce false positives

    **Dimensions**

    1. memory_opt: Memory hierarchy exploitation
       0 = Naive global, 1 = Coalesced/vectorized, 2 = SLM tiling, 3 = Register blocking + async

    2. compute_opt: Algorithmic efficiency
       0 = Multi-pass, 1 = Fused operations, 2 = Single-pass/streaming, 3 = Tiled/blocked algorithms

    3. parallelism_opt: Parallelism granularity
       0 = Thread-only, 1 = Work-group barriers, 2 = Sub-group intrinsics, 3 = Hierarchical
    """

    # Pre-compiled regex patterns - initialized at class load time for thread safety
    # This eliminates lock contention during classification
    # Value of None indicates an invalid pattern that failed to compile
    _compiled_cache: Dict[str, Optional[re.Pattern]] = {}
    _cache_initialized = False
    _init_lock = Lock()  # Only used during one-time initialization

    # ========================================================================
    # CLASSIFICATION METHODS
    # ========================================================================

    @classmethod
    def _ensure_patterns_compiled(cls) -> None:
        """
        Pre-compile all regex patterns at first use.

        This eliminates lock contention during classification by doing all
        compilation upfront under a single lock acquisition.
        Thread-safe via double-checked locking pattern.
        """
        if cls._cache_initialized:
            return

        with cls._init_lock:
            if cls._cache_initialized:
                return

            # Collect all patterns from all pattern dictionaries
            all_pattern_dicts = [
                MEMORY_OPT_PATTERNS,
                COMPUTE_OPT_PATTERNS,
                PARALLELISM_OPT_PATTERNS,
                CUDA_MEMORY_OPT_PATTERNS,
                CUDA_COMPUTE_OPT_PATTERNS,
                CUDA_PARALLELISM_OPT_PATTERNS,
                OPENCL_MEMORY_OPT_PATTERNS,
                OPENCL_COMPUTE_OPT_PATTERNS,
                OPENCL_PARALLELISM_OPT_PATTERNS,
                ESIMD_OPT_PATTERNS,  # Include ESIMD patterns
            ]

            patterns_compiled = 0
            patterns_failed = 0

            for pattern_dict in all_pattern_dicts:
                for level, categories in pattern_dict.items():
                    for category, config in categories.items():
                        for pattern in config.get("patterns", []):
                            if pattern not in cls._compiled_cache:
                                try:
                                    compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                                    cls._compiled_cache[pattern] = compiled
                                    patterns_compiled += 1
                                except re.error as e:
                                    logging.warning(f"Invalid regex pattern '{pattern}': {e}")
                                    cls._compiled_cache[pattern] = None  # Mark as invalid
                                    patterns_failed += 1

            cls._cache_initialized = True
            logging.debug(f"Pre-compiled {patterns_compiled} patterns ({patterns_failed} failed)")

    @classmethod
    def _get_compiled_pattern(cls, pattern: str) -> Optional[re.Pattern]:
        """
        Get a compiled regex pattern from the pre-initialized cache.

        Lock-free after initial compilation. Returns None for invalid patterns.

        Args:
            pattern: The regex pattern string

        Returns:
            Compiled pattern or None if invalid/not found
        """
        # Ensure patterns are compiled (no-op after first call)
        cls._ensure_patterns_compiled()
        return cls._compiled_cache.get(pattern)

    @classmethod
    def _compute_level_score(cls, code: str, patterns_dict: dict) -> float:
        """
        Compute a weighted score for how well code matches patterns at a level.

        The scoring considers:
        - Weight of matched categories (higher weight = stronger indicator)
        - Number of categories matched
        - Bonus for matching high-weight (definitive) patterns

        Args:
            code: Source code to analyze
            patterns_dict: Dictionary of {category: {weight, patterns}}

        Returns:
            Weighted score (0.0 to 1.0+), can exceed 1.0 for exceptional matches
        """
        if not patterns_dict:
            return 0.0

        total_weight = 0.0
        matched_weight = 0.0
        matched_categories = set()
        max_single_weight = 0.0  # Track highest single category match

        for category, config in patterns_dict.items():
            weight = config.get("weight", 1.0)
            patterns = config.get("patterns", [])
            total_weight += weight

            # Check if any pattern in this category matches
            for pattern in patterns:
                compiled = cls._get_compiled_pattern(pattern)
                if compiled is not None and compiled.search(code):
                    if category not in matched_categories:
                        matched_weight += weight
                        matched_categories.add(category)
                        max_single_weight = max(max_single_weight, weight)
                    break  # One match per category is enough

        if total_weight == 0:
            return 0.0

        # Base score: fraction of total weight matched
        base_score = matched_weight / total_weight

        # Bonus: if we matched a high-weight (definitive) pattern, boost the score
        # This ensures that matching async_work_group_copy (weight=1.0) alone
        # is enough to classify as level 3, even if other patterns aren't present
        if max_single_weight >= 0.9 and matched_weight > 0:
            # Guarantee at least 0.25 score for matching a definitive pattern
            base_score = max(base_score, 0.25)

        return base_score

    @classmethod
    def _classify_dimension(
        cls,
        code: str,
        pattern_levels: dict,
        threshold_base: float = 0.15,  # Lower base threshold - matching any category is significant
        require_lower_levels: bool = False,  # Disabled by default - patterns are independent
    ) -> Tuple[int, float]:
        """
        Classify optimization level for a single dimension.

        The classification uses a "highest level matched" approach:
        - Start from level 3 and work down
        - If ANY pattern category at a level matches with sufficient weight, use that level
        - Thresholds are relatively low because matching even one specific pattern
          (e.g., async_work_group_copy) is a strong signal

        Args:
            code: Source code to analyze
            pattern_levels: Dict mapping level (0-3) to pattern definitions
            threshold_base: Base threshold for considering a level matched.
                           Set low (0.15) because matching any high-weight category
                           (weight=1.0) in a level with 5+ categories gives ~0.2 score.
            require_lower_levels: If True, higher levels need some lower-level support.
                                  Disabled by default since optimization patterns are
                                  often independent (e.g., async copy without vectorization).

        Returns:
            Tuple of (level, confidence)
        """
        scores = {}
        for level in [0, 1, 2, 3]:
            patterns = pattern_levels.get(level, {})
            scores[level] = cls._compute_level_score(code, patterns)

        # Determine level using scoring - highest matched level wins
        detected_level = 0
        confidence = 0.0

        for level in [3, 2, 1]:
            level_score = scores[level]

            # Threshold increases slightly with level to require more evidence for higher claims
            # Level 1: 0.15, Level 2: 0.18, Level 3: 0.21
            threshold = threshold_base + (level - 1) * 0.03

            if level_score >= threshold:
                # Optional: Check if lower levels have some support (for robustness)
                if require_lower_levels and level > 1:
                    lower_support = sum(scores[l] for l in range(1, level)) / (level - 1) if level > 1 else 1.0
                    if lower_support < threshold_base * 0.3:  # More lenient threshold
                        # Skip this level if no foundation
                        continue

                detected_level = level
                confidence = level_score
                break

        # If no patterns matched, level is 0
        if detected_level == 0:
            confidence = 1.0 - max(scores.values()) if scores else 1.0

        return detected_level, confidence

    @classmethod
    def classify_from_code(
        cls,
        code: str,
        language: str = "sycl",
        return_confidence: bool = False,
    ) -> Tuple[int, int, int, int]:
        """
        Classify optimization level using static code analysis.

        This is the source of truth for feature grid coordinates.
        Deterministic, stable across runs, immune to execution variations.

        Args:
            code: Kernel source code
            language: Programming language ("sycl", "cuda", "opencl", "triton")
            return_confidence: If True, also return confidence scores (deprecated, use classify_with_confidence)

        Returns:
            Tuple of (memory_opt, compute_opt, parallelism_opt, esimd_opt) each 0-3
        """
        # Select pattern sets based on language
        lang = language.lower()
        if lang in ["cuda", "cu"]:
            memory_patterns = cls._merge_patterns(MEMORY_OPT_PATTERNS, CUDA_MEMORY_OPT_PATTERNS)
            compute_patterns = cls._merge_patterns(COMPUTE_OPT_PATTERNS, CUDA_COMPUTE_OPT_PATTERNS)
            parallelism_patterns = cls._merge_patterns(PARALLELISM_OPT_PATTERNS, CUDA_PARALLELISM_OPT_PATTERNS)
            esimd_patterns = {}  # ESIMD not applicable to CUDA
        elif lang in ["opencl", "ocl", "cl"]:
            memory_patterns = cls._merge_patterns(MEMORY_OPT_PATTERNS, OPENCL_MEMORY_OPT_PATTERNS)
            compute_patterns = cls._merge_patterns(COMPUTE_OPT_PATTERNS, OPENCL_COMPUTE_OPT_PATTERNS)
            parallelism_patterns = cls._merge_patterns(PARALLELISM_OPT_PATTERNS, OPENCL_PARALLELISM_OPT_PATTERNS)
            esimd_patterns = {}  # ESIMD not applicable to OpenCL
        elif lang in ["triton"]:
            memory_patterns = TRITON_MEMORY_OPT_PATTERNS
            compute_patterns = TRITON_COMPUTE_OPT_PATTERNS
            parallelism_patterns = TRITON_PARALLELISM_OPT_PATTERNS
            esimd_patterns = {}  # ESIMD not applicable to Triton
        elif lang in ["sycl"]:
            memory_patterns = MEMORY_OPT_PATTERNS
            compute_patterns = COMPUTE_OPT_PATTERNS
            parallelism_patterns = PARALLELISM_OPT_PATTERNS
            esimd_patterns = ESIMD_OPT_PATTERNS  # ESIMD only for SYCL
        else:
            raise ValueError(
                f"Unsupported language '{language}' for classification. Supported: sycl, cuda, opencl, ocl, cl, cu, triton"
            )

        # Classify each dimension
        memory_opt, _ = cls._classify_dimension(code, memory_patterns)
        compute_opt, _ = cls._classify_dimension(code, compute_patterns)
        parallelism_opt, _ = cls._classify_dimension(code, parallelism_patterns)
        esimd_opt, _ = cls._classify_dimension(code, esimd_patterns) if esimd_patterns else (0, 1.0)

        return (memory_opt, compute_opt, parallelism_opt, esimd_opt)

    @classmethod
    def classify_with_confidence(
        cls,
        code: str,
        language: str = "sycl",
    ) -> Tuple[Tuple[int, int, int, int], Dict[str, float]]:
        """
        Classify optimization level and return confidence scores.

        Args:
            code: Kernel source code
            language: Programming language ("sycl", "cuda", "opencl", "triton")

        Returns:
            Tuple of (coords, confidence_dict) where:
            - coords: (memory_opt, compute_opt, parallelism_opt, esimd_opt) each 0-3
            - confidence_dict: {"memory": float, "compute": float, "parallelism": float, "esimd": float}
        """
        # Select pattern sets based on language
        lang = language.lower()
        if lang in ["cuda", "cu"]:
            memory_patterns = cls._merge_patterns(MEMORY_OPT_PATTERNS, CUDA_MEMORY_OPT_PATTERNS)
            compute_patterns = cls._merge_patterns(COMPUTE_OPT_PATTERNS, CUDA_COMPUTE_OPT_PATTERNS)
            parallelism_patterns = cls._merge_patterns(PARALLELISM_OPT_PATTERNS, CUDA_PARALLELISM_OPT_PATTERNS)
            esimd_patterns = {}  # ESIMD not applicable to CUDA
        elif lang in ["opencl", "ocl", "cl"]:
            logging.info("Classifying OpenCL code - using OpenCL-specific patterns")
            memory_patterns = cls._merge_patterns(MEMORY_OPT_PATTERNS, OPENCL_MEMORY_OPT_PATTERNS)
            compute_patterns = cls._merge_patterns(COMPUTE_OPT_PATTERNS, OPENCL_COMPUTE_OPT_PATTERNS)
            parallelism_patterns = cls._merge_patterns(PARALLELISM_OPT_PATTERNS, OPENCL_PARALLELISM_OPT_PATTERNS)
            esimd_patterns = {}  # ESIMD not applicable to OpenCL
        elif lang in ["triton"]:
            logging.info("Classifying Triton code - using Triton-specific patterns")
            memory_patterns = TRITON_MEMORY_OPT_PATTERNS
            compute_patterns = TRITON_COMPUTE_OPT_PATTERNS
            parallelism_patterns = TRITON_PARALLELISM_OPT_PATTERNS
            esimd_patterns = {}  # ESIMD not applicable to Triton
        elif lang in ["sycl"]:
            memory_patterns = MEMORY_OPT_PATTERNS
            compute_patterns = COMPUTE_OPT_PATTERNS
            parallelism_patterns = PARALLELISM_OPT_PATTERNS
            esimd_patterns = ESIMD_OPT_PATTERNS
        else:
            raise ValueError(f"Unsupported language '{language}' for classification")

        # Classify each dimension
        memory_opt, mem_conf = cls._classify_dimension(code, memory_patterns)
        compute_opt, comp_conf = cls._classify_dimension(code, compute_patterns)
        parallelism_opt, par_conf = cls._classify_dimension(code, parallelism_patterns)
        esimd_opt, esimd_conf = cls._classify_dimension(code, esimd_patterns) if esimd_patterns else (0, 1.0)

        coords = (memory_opt, compute_opt, parallelism_opt, esimd_opt)
        confidence = {
            "memory": mem_conf,
            "compute": comp_conf,
            "parallelism": par_conf,
            "esimd": esimd_conf,
        }

        return coords, confidence

    @classmethod
    def _merge_patterns(cls, base: dict, overlay: dict) -> dict:
        """Merge two pattern dictionaries, with overlay taking precedence."""
        merged = {}
        for level in [0, 1, 2, 3]:
            base_patterns = base.get(level, {})
            overlay_patterns = overlay.get(level, {})
            merged[level] = {**base_patterns, **overlay_patterns}
        return merged

    @classmethod
    def get_matched_patterns(cls, code: str, language: str = "sycl") -> Dict[str, Dict[int, List[str]]]:
        """
        Get detailed breakdown of which patterns matched for debugging.

        Args:
            code: Source code to analyze
            language: Programming language ("sycl", "cuda", "cu", "opencl", "ocl", "cl")

        Returns:
            Dict mapping dimension -> level -> list of matched pattern categories
        """
        result: Dict[str, Dict[int, List[str]]] = {
            "memory": {},
            "compute": {},
            "parallelism": {},
            "esimd": {},
        }

        # Select pattern sets based on language
        lang = language.lower()
        if lang in ["cuda", "cu"]:
            pattern_sets = {
                "memory": cls._merge_patterns(MEMORY_OPT_PATTERNS, CUDA_MEMORY_OPT_PATTERNS),
                "compute": cls._merge_patterns(COMPUTE_OPT_PATTERNS, CUDA_COMPUTE_OPT_PATTERNS),
                "parallelism": cls._merge_patterns(PARALLELISM_OPT_PATTERNS, CUDA_PARALLELISM_OPT_PATTERNS),
                "esimd": {},  # ESIMD not applicable to CUDA
            }
        elif lang in ["opencl", "ocl", "cl"]:
            pattern_sets = {
                "memory": cls._merge_patterns(MEMORY_OPT_PATTERNS, OPENCL_MEMORY_OPT_PATTERNS),
                "compute": cls._merge_patterns(COMPUTE_OPT_PATTERNS, OPENCL_COMPUTE_OPT_PATTERNS),
                "parallelism": cls._merge_patterns(PARALLELISM_OPT_PATTERNS, OPENCL_PARALLELISM_OPT_PATTERNS),
                "esimd": {},  # ESIMD not applicable to OpenCL
            }
        elif lang in ["triton"]:
            pattern_sets = {
                "memory": TRITON_MEMORY_OPT_PATTERNS,
                "compute": TRITON_COMPUTE_OPT_PATTERNS,
                "parallelism": TRITON_PARALLELISM_OPT_PATTERNS,
                "esimd": {},  # ESIMD not applicable to Triton
            }
        elif lang in ["sycl"]:
            pattern_sets = {
                "memory": MEMORY_OPT_PATTERNS,
                "compute": COMPUTE_OPT_PATTERNS,
                "parallelism": PARALLELISM_OPT_PATTERNS,
                "esimd": ESIMD_OPT_PATTERNS,
            }
        else:
            raise ValueError(f"Unsupported language '{language}' for pattern matching")

        for dim_name, patterns_by_level in pattern_sets.items():
            for level, categories in patterns_by_level.items():
                matched = []
                for category, config in categories.items():
                    for pattern in config.get("patterns", []):
                        compiled = cls._get_compiled_pattern(pattern)
                        if compiled is not None and compiled.search(code):
                            matched.append(category)
                            break
                if matched:
                    result[dim_name][level] = matched

        return result

    @classmethod
    def explain_classification(cls, code: str, language: str = "sycl") -> str:
        """
        Generate human-readable explanation of classification.

        Useful for debugging and understanding why code was classified a certain way.

        Args:
            code: Source code to analyze
            language: Programming language

        Returns:
            Formatted string explaining the classification
        """
        coords, confidence = cls.classify_with_confidence(code, language)
        matched = cls.get_matched_patterns(code, language)

        lines = [
            "=" * 60,
            "OPTIMIZATION CLASSIFICATION EXPLANATION",
            "=" * 60,
            f"\nClassification Result: memory={coords[0]}, compute={coords[1]}, "
            f"parallelism={coords[2]}, esimd={coords[3]}",
            f"Confidence: memory={confidence['memory']:.2f}, compute={confidence['compute']:.2f}, "
            f"parallelism={confidence['parallelism']:.2f}, esimd={confidence['esimd']:.2f}",
            "\n--- Memory Optimization ---",
        ]

        for level in [3, 2, 1]:
            if level in matched["memory"]:
                lines.append(f"  Level {level} patterns: {', '.join(matched['memory'][level])}")

        lines.append("\n--- Compute Optimization ---")
        for level in [3, 2, 1]:
            if level in matched["compute"]:
                lines.append(f"  Level {level} patterns: {', '.join(matched['compute'][level])}")

        lines.append("\n--- Parallelism Optimization ---")
        for level in [3, 2, 1]:
            if level in matched["parallelism"]:
                lines.append(f"  Level {level} patterns: {', '.join(matched['parallelism'][level])}")

        lines.append("\n--- ESIMD Optimization ---")
        for level in [3, 2, 1]:
            if "esimd" in matched and level in matched["esimd"]:
                lines.append(f"  Level {level} patterns: {', '.join(matched['esimd'][level])}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================================
# MAIN DATABASE CLASS
# ============================================================================


class OptimizationAwareDatabase:
    """
    Production MAP-Elites database specialized for optimization-aware features.

    **Grid Structure**

    - 3D grid: memory_opt × compute_opt × parallelism_opt
    - Each dimension has 4 discrete levels (0-3)
    - Total capacity: 4×4×4 = 64 behavioral niches
    - Each cell stores the best program (by fitness score) for that niche

    **Coordinate Assignment**

    - Coordinates extracted from code patterns (static analysis)
    - Same code always maps to same coordinates
    - No normalization, no statistical scaling
    - Deterministic and reproducible

    **Fitness Evaluation**

    - Performance metrics (runtime, bandwidth, etc.) used for elite selection
    - Within each cell, keep the program with highest combined_score
    - Metrics do NOT affect coordinates, only fitness ranking

    **Island Model**

    - Optional: Multiple sub-populations evolve independently
    - Periodic migration of top performers between islands
    - Prevents premature convergence
    """

    def __init__(self, config):
        """
        Initialize optimization-aware MAP-Elites database.

        Args:
            config: Configuration object with:
                - num_islands: Number of sub-populations
                - programs_per_island: Programs before island switch
                - migration_interval: Generations between migrations
                - migration_rate: Fraction of population to migrate
                - population_size: Maximum total programs
                - random_seed: For reproducibility
        """
        self.config = config

        # Thread Safety - main lock for all shared state modifications
        self._population_lock = RLock()
        # Separate lock for archive cache to allow concurrent reads
        self._archive_cache_lock = Lock()

        # Program Storage
        self.programs: Dict[str, Program] = {}
        self.num_inspirations: int = config.num_inspirations

        # Feature Grid (MAP-Elites) - Fixed 4×4×4×4 for optimization features including ESIMD
        self.feature_map: Dict[Tuple[int, int, int, int], str] = {}  # (x,y,z,w) -> program_id
        self.grid_shape = (4, 4, 4, 4)  # (memory_opt, compute_opt, parallelism_opt, esimd_opt)
        self.max_archive_size = 256  # 4×4×4×4

        # Cache for archive queries - invalidated atomically
        self._archive_cache: Optional[List[Program]] = None
        self._archive_cache_valid = False

        logging.info(f"MAP-Elites grid: 4×4×4×4 = {self.max_archive_size} cells")
        logging.info(f"Dimensions: memory_opt, compute_opt, parallelism_opt, esimd_opt")

        # Island Model
        self.num_islands: int = config.num_islands
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)
        self.programs_per_island: int = getattr(config, "programs_per_island", 5)
        self.current_island_counter: int = 0

        # Population Management with efficient score tracking
        self.population_size_limit: int = getattr(config, "population_size", 500)
        # Min-heap of (neg_score, program_id) for efficient worst-program removal
        # We use negative scores because heapq is a min-heap
        self._score_heap: List[ProgramScoreEntry] = []
        self._heap_valid = False  # Tracks if heap needs rebuilding

        # Prompt Configuration (for controller compatibility)
        num_top = getattr(config, "num_top_programs", 1)
        num_diverse = getattr(config, "num_diverse_programs", 0)
        self.num_top_programs: int = num_top if num_top is not None else 1
        self.num_diverse_programs: int = num_diverse if num_diverse is not None else 0

        # Best Program Tracking
        self.best_program_id: Optional[str] = None
        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands

        # Evolution Metadata
        self.last_iteration: int = 0
        self.output_dir: Optional[str] = None

        # Reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            logging.info(f"Set random seed to {config.random_seed}")

        # Diversity Infrastructure (for sampling, not coordinates)
        self.diversity_cache: OrderedDict[int, float] = OrderedDict()
        self.diversity_cache_size: int = 1000
        self.diversity_reference_codes: List[str] = []
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)

        # Exploration/Exploitation ratios (matching EvolveProgramDatabase)
        self.exploration_ratio: float = getattr(config, "exploration_ratio", 0.2)
        self.exploitation_ratio: float = getattr(config, "exploitation_ratio", 0.7)
        # Remaining probability (1.0 - exploration - exploitation) goes to random sampling

        # ====================================================================
        # QD GRADIENT TRACKING (Optional Enhancement)
        # ====================================================================
        # Enable gradient-based transition tracking for improved sampling
        # and mutation guidance. Gracefully degrades if not available.
        self._use_gradient_tracking: bool = getattr(config, "use_gradient_tracking", True)
        self._transition_tracker = None  # Will be TransitionTracker if enabled

        if self._use_gradient_tracking and HAS_QD_GRADIENT and _TransitionTracker is not None:
            gradient_config = getattr(config, "gradient_config", {})
            if isinstance(gradient_config, DictConfig):
                gradient_config = dict(gradient_config)
            # Create gradient estimator if available
            estimator = None
            if _GradientEstimator is not None:
                estimator = _GradientEstimator(
                    fitness_weight=gradient_config.get("fitness_weight", 0.4),
                    improvement_rate_weight=gradient_config.get("improvement_rate_weight", 0.4),
                    exploration_weight=gradient_config.get("exploration_weight", 0.2),
                )
            self._transition_tracker = _TransitionTracker(
                max_history=gradient_config.get("max_history", 10000),
                max_cell_cache=gradient_config.get("max_cell_cache", 256),
                gradient_estimator=estimator,
                checkpoint_interval=gradient_config.get("checkpoint_interval", 100),
            )
            logging.info("QD Gradient tracking ENABLED")
            logging.info(f"  Max history: {gradient_config.get('max_history', 10000)}")
        elif self._use_gradient_tracking and not HAS_QD_GRADIENT:
            logging.warning("QD Gradient tracking requested but qd_gradient module not available. ")
            self._use_gradient_tracking = False
        else:
            logging.info("QD Gradient tracking DISABLED")

        # Gradient-based sampling configuration
        self._gradient_sampling_weight: float = getattr(
            config, "gradient_sampling_weight", 0.3
        )  # Weight for gradient-informed sampling vs standard sampling

        logging.info("Optimization-aware database initialized")
        logging.info(f"  Exploration ratio: {self.exploration_ratio}")
        logging.info(f"  Exploitation ratio: {self.exploitation_ratio}")

    # ========================================================================
    # PROGRAM ADDITION
    # ========================================================================

    def add(
        self,
        program: Program,
        island_id: Optional[int] = None,
        force_add: bool = False,
        iteration: Optional[int] = None,
    ) -> bool:
        """
        Add a program to the database with coordinate calculation.

        Process:
        1. Extract optimization levels from code (0-3 for each dimension)
        2. Map to grid cell coordinates
        3. Compare with current elite in that cell
        4. Keep better program (by fitness score)

        Args:
            program: Program to add
            island_id: Island to assign program to (None = auto-assign)
            force_add: If True, bypass population limit enforcement
            iteration: Iteration number (for metadata)

        Returns:
            True if program became an elite (new or replaced existing)
        """
        with self._population_lock:
            # Check if program already exists
            if program.id in self.programs:
                logging.debug(f"Program {program.id} already exists - updating metrics")
                existing = self.programs[program.id]

                # Update metrics but preserve coordinates
                existing.code = program.code
                existing.metrics = program.metrics
                existing.metadata = {**existing.metadata, **program.metadata}

                # Check if improved enough to update feature map
                old_score = get_main_score(existing.metrics or {})
                new_score = get_main_score(program.metrics or {})

                if new_score > old_score:
                    stored_coords = existing.metadata.get("feature_coords")
                    if stored_coords:
                        # Handle legacy 3-tuple coords by adding esimd_opt=0
                        coords: Tuple[int, int, int, int]
                        if len(stored_coords) == 3:
                            coords = (stored_coords[0], stored_coords[1], stored_coords[2], 0)
                        else:
                            coords = tuple(stored_coords)  # type: ignore
                        coord_key = self._coords_to_key(coords)
                        current_elite_id = self.feature_map.get(coord_key)

                        if current_elite_id == program.id:
                            logging.debug(f"Updated elite in cell {coord_key}: {old_score:.4f} → {new_score:.4f}")

                return False  # Not a new program

            # Calculate coordinates from code patterns
            coords = self._calculate_feature_coords(program)

            # Store coordinates in metadata
            program.metadata["feature_coords"] = coords
            program.metadata["coord_timestamp"] = time.time()

            if iteration is not None:
                program.metadata["iteration"] = iteration

            # Add to programs dict
            self.programs[program.id] = program

            # Update feature map (MAP-Elites selection)
            coord_key = self._coords_to_key(coords)
            current_elite_id = self.feature_map.get(coord_key)

            should_replace = False
            if current_elite_id is None:
                # Empty cell - always add
                should_replace = True
                logging.debug(f"New program fills empty cell {coord_key}")
            else:
                # Check if better than current elite
                current_elite = self.programs.get(current_elite_id)
                if current_elite:
                    new_score = get_main_score(program.metrics or {})
                    current_score = get_main_score(current_elite.metrics or {})

                    if new_score > current_score:
                        should_replace = True
                        logging.info(
                            f"New elite in cell {coord_key}: "
                            f"{program.id} (score={new_score:.4f}) "
                            f"replaced {current_elite_id} (score={current_score:.4f})"
                        )
                else:
                    # Current elite was deleted - replace
                    should_replace = True
                    logging.debug(f"Replacing deleted elite in cell {coord_key}")

            if should_replace:
                self.feature_map[coord_key] = program.id
                self._invalidate_archive_cache()  # Cache invalidation on elite change

            # Track best program globally
            self._update_best_program(program)

            # Assign to island if using island model
            if self.use_islands:
                target_island = island_id if island_id is not None else self.current_island
                self.islands[target_island].add(program.id)
                program.metadata["island"] = target_island
                self._update_island_best(program, target_island)

            # Enforce population limit
            if not force_add:
                self._enforce_population_limit(exclude_program_id=program.id)

            return should_replace

    def _calculate_feature_coords(self, program: Program) -> Tuple[int, int, int, int]:
        """
        Calculate feature coordinates from code patterns.

        Extracts discrete optimization levels (0-3) using static analysis.
        Deterministic and execution-independent.

        Args:
            program: Program to calculate coordinates for

        Returns:
            Tuple of (memory_opt, compute_opt, parallelism_opt, esimd_opt)
        """
        return OptimizationFeatureClassifier.classify_from_code(program.code_as_str, program.language)

    def _coords_to_key(self, coords: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Convert coordinates to key for feature map lookup.

        Args:
            coords: Tuple of (memory_opt, compute_opt, parallelism_opt, esimd_opt)

        Returns:
            Tuple key (pass-through for type safety)
        """
        return coords

    # ========================================================================
    # GRADIENT-BASED TRANSITION TRACKING
    # ========================================================================

    def add_with_parent(
        self,
        child: Program,
        parent: Program,
        island_id: Optional[int] = None,
        force_add: bool = False,
        iteration: Optional[int] = None,
        mutation_hint: Optional[str] = None,
    ) -> bool:
        """
        Add a program with parent information for gradient tracking.

        This method extends the standard add() by recording the parent→child
        transition for gradient computation. Use this instead of add() when
        parent information is available.

        Args:
            child: The child program to add
            parent: The parent program that was mutated
            island_id: Island to assign the child to
            force_add: If True, bypass population limit enforcement
            iteration: Evolution iteration number
            mutation_hint: Optional description of the mutation applied

        Returns:
            True if child became an elite (new cell or replaced existing)
        """
        with self._population_lock:
            # Get parent info before adding child
            parent_coords_raw = parent.metadata.get("feature_coords")
            if parent_coords_raw is None:
                parent_coords_raw = self._calculate_feature_coords(parent)
            # Ensure 4-tuple with explicit indexing for type safety
            if len(parent_coords_raw) == 3:
                parent_coords: Tuple[int, int, int, int] = (
                    int(parent_coords_raw[0]),
                    int(parent_coords_raw[1]),
                    int(parent_coords_raw[2]),
                    0,
                )
            else:
                parent_coords = (
                    int(parent_coords_raw[0]),
                    int(parent_coords_raw[1]),
                    int(parent_coords_raw[2]),
                    int(parent_coords_raw[3]),
                )

            parent_fitness = get_main_score(parent.metrics or {})

            # Check if target cell was empty before
            child_coords = self._calculate_feature_coords(child)
            child_coord_key = self._coords_to_key(child_coords)
            was_empty = child_coord_key not in self.feature_map

            # Get current elite info if cell is occupied
            current_elite_id = self.feature_map.get(child_coord_key)
            current_elite_score = 0.0
            if current_elite_id and current_elite_id in self.programs:
                current_elite_score = get_main_score(self.programs[current_elite_id].metrics or {})

            # Perform the standard add
            became_elite = self.add(
                child,
                island_id=island_id,
                force_add=force_add,
                iteration=iteration,
            )

            # Record transition for gradient tracking
            if self._use_gradient_tracking and self._transition_tracker is not None:
                child_fitness = get_main_score(child.metrics or {})

                # Determine if this was an elite replacement
                is_elite_replacement = False
                if not was_empty and became_elite:
                    is_elite_replacement = child_fitness > current_elite_score

                self._transition_tracker.record_transition(
                    parent_id=parent.id,
                    child_id=child.id,
                    parent_coords=parent_coords,
                    child_coords=child_coords,
                    parent_fitness=parent_fitness,
                    child_fitness=child_fitness,
                    is_new_cell=was_empty and became_elite,
                    is_elite_replacement=is_elite_replacement,
                    iteration=iteration or self.last_iteration,
                    mutation_hint=mutation_hint,
                )

                logging.debug(
                    f"Recorded transition: {parent_coords} → {child_coords}, "
                    f"delta={child_fitness - parent_fitness:.4f}"
                )

            return became_elite

    def get_gradient_at_coords(
        self,
        coords: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[float, float, float, float], Dict[str, Any]]:
        """
        Get gradient estimate at specified behavioral coordinates.

        The gradient indicates which direction in optimization space
        is most likely to yield improvements based on historical transitions.

        Args:
            coords: Behavioral coordinates (memory_opt, compute_opt, parallelism_opt, esimd_opt)

        Returns:
            Tuple of (gradient_vector, metadata_dict):
            - gradient_vector: (d_memory, d_compute, d_parallelism, d_esimd)
              Positive values suggest increasing that dimension
            - metadata_dict: Contains confidence, sample_count, etc.
        """
        if not self._use_gradient_tracking or self._transition_tracker is None:
            # Return zero gradient with low confidence
            return (0.0, 0.0, 0.0, 0.0), {"source": "disabled", "confidence": 0.0}

        # Get empty and low-quality cells for exploration gradient
        underexplored = self._get_underexplored_cells_internal(n=20)
        empty_cells: List[Tuple[int, int, int, int]] = [
            (r["memory_opt"], r["compute_opt"], r["parallelism_opt"], r.get("esimd_opt", 0))
            for r in underexplored
            if "current_score" not in r
        ]
        low_quality_cells: List[Tuple[Tuple[int, int, int, int], float]] = [
            (
                (r["memory_opt"], r["compute_opt"], r["parallelism_opt"], r.get("esimd_opt", 0)),
                float(r["current_score"]),  # Explicit float conversion
            )
            for r in underexplored
            if "current_score" in r
        ]

        # Get max score for normalization
        max_score = 1.0
        if self.best_program_id and self.best_program_id in self.programs:
            max_score = max(1.0, get_main_score(self.programs[self.best_program_id].metrics or {}))

        return self._transition_tracker.get_gradient(
            coords=coords,
            empty_cells=empty_cells,
            low_quality_cells=low_quality_cells,
            max_score=max_score,
        )

    def get_mutation_hints_for_parent(
        self,
        parent: Program,
        max_hints: int = 3,
    ) -> List[str]:
        """
        Get mutation hints for a parent program based on gradient information.

        These hints can be injected into LLM prompts to guide the direction
        of optimization.

        Args:
            parent: The parent program to mutate
            max_hints: Maximum number of hints to return

        Returns:
            List of human-readable mutation hint strings
        """
        if not self._use_gradient_tracking or self._transition_tracker is None:
            return []

        parent_coords_raw = parent.metadata.get("feature_coords")
        if parent_coords_raw is None:
            parent_coords_raw = self._calculate_feature_coords(parent)
        # Construct explicit 4-tuple for type safety
        if len(parent_coords_raw) == 3:
            parent_coords: Tuple[int, int, int, int] = (
                int(parent_coords_raw[0]),
                int(parent_coords_raw[1]),
                int(parent_coords_raw[2]),
                0,
            )
        else:
            parent_coords = (
                int(parent_coords_raw[0]),
                int(parent_coords_raw[1]),
                int(parent_coords_raw[2]),
                int(parent_coords_raw[3]),
            )

        return self._transition_tracker.get_mutation_hints(
            coords=parent_coords,
            threshold=0.2,
            max_hints=max_hints,
        )

    def get_gradient_weighted_sampling_probabilities(
        self,
        candidate_ids: List[str],
        strategy: str = "combined",
    ) -> Dict[str, float]:
        """
        Compute sampling probabilities for candidates using gradient information.

        Combines standard fitness-based sampling with gradient-informed weights
        to prioritize parents that historically produce improvements.

        Args:
            candidate_ids: List of program IDs to consider
            strategy: Gradient weighting strategy:
                - "improvement_rate": Weight by historical improvement rate
                - "gradient_magnitude": Weight by gradient magnitude
                - "combined": Blend of both

        Returns:
            Dict mapping program_id to sampling probability
        """
        with self._population_lock:
            if not candidate_ids:
                return {}

            # Get base fitness scores
            base_scores = {}
            coords_by_id = {}
            for pid in candidate_ids:
                if pid in self.programs:
                    prog = self.programs[pid]
                    base_scores[pid] = get_main_score(prog.metrics or {})
                    prog_coords = prog.metadata.get("feature_coords")
                    if prog_coords:
                        if len(prog_coords) == 3:
                            coords_by_id[pid] = tuple(prog_coords) + (0,)
                        else:
                            coords_by_id[pid] = tuple(prog_coords)

            if not base_scores:
                return {}

            # Compute softmax of base scores
            scores_array = np.array(list(base_scores.values()))
            max_score = np.max(scores_array) if len(scores_array) > 0 else 0.0
            exp_scores = np.exp(scores_array - max_score)
            base_probs = exp_scores / np.sum(exp_scores)
            base_prob_dict = dict(zip(base_scores.keys(), base_probs))

            # If gradient tracking is disabled, return base probabilities
            if not self._use_gradient_tracking or self._transition_tracker is None:
                return base_prob_dict

            # Get gradient weights
            gradient_weights = {}
            coord_list = list(coords_by_id.values())
            if coord_list:
                weight_dict = self._transition_tracker.get_sampling_weights(
                    candidate_coords=coord_list,
                    strategy=strategy,
                )
                for pid, coords in coords_by_id.items():
                    gradient_weights[pid] = weight_dict.get(coords, 0.5)

            # Blend base probabilities with gradient weights
            alpha = self._gradient_sampling_weight  # Weight for gradient component
            combined_probs = {}
            for pid in base_scores.keys():
                base_p = base_prob_dict.get(pid, 0.0)
                grad_w = gradient_weights.get(pid, 0.5)
                combined_probs[pid] = (1 - alpha) * base_p + alpha * grad_w

            # Normalize to sum to 1
            total = sum(combined_probs.values())
            if total > 0:
                combined_probs = {k: v / total for k, v in combined_probs.items()}

            return combined_probs

    def get_best_transition_directions_from_parent(
        self,
        parent: Program,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get the most successful transition directions from a parent's cell.

        Returns information about which optimization changes historically
        led to improvements when starting from this parent's position.

        Args:
            parent: The parent program
            top_k: Number of directions to return

        Returns:
            List of dicts with keys:
                - direction: (d_mem, d_comp, d_par, d_esimd)
                - success_rate: Fraction of successful transitions
                - sample_count: Number of observations
                - description: Human-readable description
        """
        if not self._use_gradient_tracking or self._transition_tracker is None:
            return []

        parent_coords_raw = parent.metadata.get("feature_coords")
        if parent_coords_raw is None:
            parent_coords_raw = self._calculate_feature_coords(parent)
        # Construct explicit 4-tuple for type safety
        if len(parent_coords_raw) == 3:
            parent_coords: Tuple[int, int, int, int] = (
                int(parent_coords_raw[0]),
                int(parent_coords_raw[1]),
                int(parent_coords_raw[2]),
                0,
            )
        else:
            parent_coords = (
                int(parent_coords_raw[0]),
                int(parent_coords_raw[1]),
                int(parent_coords_raw[2]),
                int(parent_coords_raw[3]),
            )

        directions = self._transition_tracker.get_best_transition_directions(
            coords=parent_coords,
            top_k=top_k,
        )

        # Convert to more informative format
        dimension_names = ["memory_opt", "compute_opt", "parallelism_opt", "esimd_opt"]
        results = []
        for direction_vec, success_rate, sample_count in directions:
            # Build human-readable description
            changes = []
            for i, (dim_name, delta) in enumerate(zip(dimension_names, direction_vec)):
                if delta != 0:
                    direction_word = "increase" if delta > 0 else "decrease"
                    changes.append(f"{direction_word} {dim_name}")

            results.append(
                {
                    "direction": direction_vec,
                    "success_rate": success_rate,
                    "sample_count": sample_count,
                    "description": " and ".join(changes) if changes else "no change",
                }
            )

        return results

    def get_transition_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about evolutionary transitions.

        Returns metrics about improvement rates, discovery patterns,
        and gradient effectiveness.

        Returns:
            Dict with transition statistics
        """
        if not self._use_gradient_tracking or self._transition_tracker is None:
            return {"gradient_tracking": "disabled"}

        stats = self._transition_tracker.get_statistics()
        stats["gradient_tracking"] = "enabled"

        # Add database-specific context
        stats["archive_coverage"] = len(self.feature_map) / self.max_archive_size
        stats["total_programs"] = len(self.programs)

        return stats

    def get_cell_transition_info(
        self,
        coords: Tuple[int, int, int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed transition information for a specific cell.

        Args:
            coords: Cell coordinates

        Returns:
            Dict with cell-specific transition statistics, or None if no data
        """
        if not self._use_gradient_tracking or self._transition_tracker is None:
            return None

        return self._transition_tracker.get_cell_statistics(coords)

    def _update_best_program(self, program: Program) -> None:
        """Update global best program tracker"""
        score = get_main_score(program.metrics or {})

        if self.best_program_id is None:
            self.best_program_id = program.id
            logging.info(f"New best program: {program.id} (score={score:.4f}{describe_score(score)})")
        else:
            best = self.programs.get(self.best_program_id)
            if best:
                best_score = get_main_score(best.metrics or {})
                if score > best_score:
                    self.best_program_id = program.id
                    logging.info(
                        f"New best program: {program.id} (score={score:.4f}{describe_score(score)}) "
                        f"replaced {best.id} (score={best_score:.4f})"
                    )

    def _update_island_best(self, program: Program, island_id: int) -> None:
        """Update best program for a specific island"""
        score = get_main_score(program.metrics or {})

        if self.island_best_programs[island_id] is None:
            self.island_best_programs[island_id] = program.id
        else:
            best_id = self.island_best_programs[island_id]
            if best_id is not None:
                best = self.programs.get(best_id)
                if best:
                    best_score = get_main_score(best.metrics or {})
                    if score > best_score:
                        self.island_best_programs[island_id] = program.id

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:
        """
        Enforce population size limit by removing worst non-elite programs.

        Uses a heap-based approach for O(k log n) removal of k worst programs,
        instead of O(n log n) full sort.

        Never removes:
        - Elite programs (in feature_map)
        - Best program globally
        - The program being added (exclude_program_id)

        Args:
            exclude_program_id: Program ID to protect from deletion

        Note:
            This method assumes _population_lock is already held by caller.
            Called from add() which holds the lock.
        """
        # Check inside lock to avoid TOCTOU race
        if len(self.programs) <= self.population_size_limit:
            return

        num_to_remove = len(self.programs) - self.population_size_limit

        # Identify protected programs
        elite_ids = set(self.feature_map.values())
        protected_ids = elite_ids.copy()

        if self.best_program_id:
            protected_ids.add(self.best_program_id)
        if exclude_program_id:
            protected_ids.add(exclude_program_id)

        # Build a min-heap of candidates (by score) for efficient removal of worst
        # We want to remove the LOWEST scoring programs, so min-heap is perfect
        candidate_heap: List[Tuple[float, str]] = []

        for pid, prog in self.programs.items():
            if pid not in protected_ids:
                score = get_main_score(prog.metrics or {})
                heapq.heappush(candidate_heap, (score, pid))

        if not candidate_heap:
            logging.warning(
                f"Population limit exceeded ({len(self.programs)} > {self.population_size_limit}) "
                f"but all programs are protected"
            )
            return

        # Pop the k worst (lowest score) programs
        programs_to_remove: List[Tuple[str, float]] = []
        for _ in range(min(num_to_remove, len(candidate_heap))):
            score, program_id = heapq.heappop(candidate_heap)
            programs_to_remove.append((program_id, score))

        # Remove programs
        for program_id, score in programs_to_remove:
            del self.programs[program_id]

            # Remove from islands
            if self.use_islands:
                for island in self.islands:
                    island.discard(program_id)

            logging.debug(f"Removed non-elite program {program_id} (score={score:.4f})")

        if programs_to_remove:
            logging.info(
                f"Population limit enforced: removed {len(programs_to_remove)} programs "
                f"({len(self.programs)} remain)"
            )

    # ========================================================================
    # SETUP AND INITIALIZATION
    # ========================================================================

    def setup(
        self,
        language: Optional[str] = None,
        problem_name: Optional[str] = None,
        gpu_arch: Optional[str] = None,
        init_programs: Optional[List[Program]] = None,
        resume_from_archive: Optional[str] = None,
        resume_from_database: bool = False,
        max_speedup: float = 1.0,
        population_filter_fn=None,
        output_dir: Optional[str] = None,
        task_id: str | None = None,
    ) -> List[Program]:
        """
        Initialize the database with programs.
        1. Load initial programs or resume from checkpoint
        2. Add all programs to database
        3. Initialize diversity reference set (for sampling only)

        Args:
            language: Output language filter (for backward compatibility, unused)
            problem_name: Problem name filter (for backward compatibility, unused)
            gpu_arch: GPU architecture filter (for backward compatibility, unused)
            init_programs: Initial programs to seed population
            resume_from_archive: Path to checkpoint to resume from
            resume_from_database: Whether to load the database of programs
            population_filter_fn: Optional filter for programs
            output_dir: Output directory for visualizations
            max_speedup: Maximum speedup to consider when loading from database

        Returns:
            List of programs available for evolution

        Note:
            The language, problem_name, and gpu_arch arguments are accepted for
            backward compatibility with the controller but are not used in this
            simplified implementation. The optimization-aware database works with
            any language/architecture since coordinates come from code patterns.
        """
        logging.info("=" * 80)
        logging.info("OPTIMIZATION-AWARE DATABASE INITIALIZATION")
        logging.info("=" * 80)

        self.output_dir = output_dir

        # Set output directory for transition tracker (for checkpointing)
        if self._use_gradient_tracking and self._transition_tracker is not None and output_dir:
            self._transition_tracker.set_output_dir(output_dir)
            logging.info(f"  Transition tracker output dir: {output_dir}")

        # Collect initial programs
        programs = []

        if init_programs:
            programs.extend(init_programs)
            logging.info(f"Loaded {len(init_programs)} initial programs")

        if resume_from_archive:
            logging.info(f"Loading checkpoint from: {resume_from_archive}")
            checkpoint_programs = self._load_checkpoint(resume_from_archive)
            programs.extend(checkpoint_programs)
            logging.info(f"Loaded {len(checkpoint_programs)} checkpoint programs")

        # Apply filter if provided
        if population_filter_fn:
            logging.info("Applying population filter...")
            before_count = len(programs)
            programs = [p for p in programs if population_filter_fn(p)]
            logging.info(f"  Filter kept {len(programs)}/{before_count} programs")

        # LOAD KERNELS FROM DATABASE
        if resume_from_database:
            logging.info(f"Load programs from database")
            filter_dict = {"output_language": language, "task_name": problem_name}
            if task_id is not None:
                filter_dict["task_id"] = task_id
            if gpu_arch != "all":
                assert (
                    isinstance(gpu_arch, str) or len(gpu_arch) == 1
                ), f"Multiple gpu_arch currently not supported in evolve database"
                gpu_arch = gpu_arch if isinstance(gpu_arch, str) else gpu_arch[0]
                filter_dict["gpu_arch"] = gpu_arch
            # load all kernels generated for this specific problem
            db_kernels = load_database(filter_dict, max_improvement=max_speedup)

            if problem_name is not None:
                programs = self.load_all_programs_from_runs(
                    db_kernels, language, problem_name, restrict_to_correct=True
                )
                if len(programs) == 0:
                    print("No correct programs found for problem ID", problem_name, ", retry with all.")
                    programs = self.load_all_programs_from_runs(
                        db_kernels, language, problem_name, restrict_to_correct=False
                    )
                print(f"{len(programs)} programs found for problem ID {problem_name}")

        # Add all programs to database
        programs_added = 0
        for program in programs:
            if self.add(program, force_add=True):
                programs_added += 1

        logging.info(f"Added {programs_added} elites to feature map")
        logging.info(f"  Feature map coverage: {len(self.feature_map)}/{self.max_archive_size} cells")
        logging.info(f"  Total programs: {len(self.programs)}")

        # Initialize diversity reference set (for sampling diversity, not coordinates)
        if len(self.programs) > 0:
            self._initialize_diversity_reference_set()

        # Log island distribution
        if self.use_islands:
            for i, island in enumerate(self.islands):
                logging.info(f"  Island {i}: {len(island)} programs")

        available_programs = list(self.programs.values())

        logging.info("=" * 80)
        logging.info("INITIALIZATION COMPLETE")
        logging.info(f"  Archive coverage: {len(self.feature_map)}/{self.max_archive_size}")
        logging.info(f"  Total programs: {len(self.programs)}")
        logging.info(f"  Available for evolution: {len(available_programs)}")
        logging.info("=" * 80)

        return available_programs

    def load_all_programs_from_runs(
        self,
        db_kernels: pd.DataFrame,
        filter_language: str,
        filter_problem_name: str,
        restrict_to_correct: bool = False,
        verbose: bool = False,
    ) -> list[Program]:
        """
        Iterate over all runs we have and select the ones of <filter_problem_id> and <filter_language>.
        Returns a list of Program objects with their evaluation results.
        """

        program_list = []

        # problem-specific db
        db = db_kernels[db_kernels["task_name"] == filter_problem_name]
        print(f"Filtered for problem {filter_problem_name}, remaining entries:", len(db))
        if restrict_to_correct:
            db = db[db["status"] == "correct"]
            print("Restricted to correct solutions")
        if len(db) > 0:
            print(db[["job_name", "status", "trial", "Op_ID", "runtime", "improve_over_native"]].head(20))

        # iterate over database
        for _, row in db.iterrows():
            # recover exec result
            compiled = row["status"] != "error"
            correct = row["status"] == "correct"
            # TODO: from str and to str functions for exec results and save in db
            runtime_stats = row["runtime_stats"]
            runtime_stats["hardware"] = ARCH_TO_NAME[row["gpu_arch"]]
            kernel_exec_result = EvalResult(
                compiled=compiled,
                correctness=correct,
                runtime=row["runtime"],
                runtime_stats=runtime_stats,
                perf_score=row["score"],
                runtime_improvement=row["improve_over_native"],
            )
            # convert into problem
            program_id = str(uuid.uuid4())
            program = Program(id=program_id, code=row["output_code"], language=row["output_language"])
            if not row["eval_log"]:
                print("Eval log missing in database entry, skipping")
                continue
            program.add_eval_results(exec_result=kernel_exec_result, artifact_str=row["eval_log"])
            # add to list
            program_list.append(program)
            if verbose:
                print("Added program from ", row["job_name"], "trial", row["trial"], "version", row["version"])
                print(program.metrics)
                print()
        return program_list

    def _load_checkpoint(self, checkpoint_path: str) -> List[Program]:
        """Load programs from a checkpoint file"""
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)

            programs = []
            for prog_data in checkpoint_data.get("programs", []):
                program = Program(
                    id=prog_data["id"],
                    code=prog_data["code"],
                    language=prog_data.get("language", "unknown"),
                    metrics=prog_data.get("metrics"),
                    metadata=prog_data.get("metadata", {}),
                )
                programs.append(program)

            if "last_iteration" in checkpoint_data:
                self.last_iteration = checkpoint_data["last_iteration"]

            logging.info(f"Loaded {len(programs)} programs from checkpoint")
            return programs

        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return []

    def _initialize_diversity_reference_set(self) -> None:
        """
        Initialize diversity reference set for sampling.

        Note: This is for diversity-based SAMPLING, not for grid coordinates.
        Uses deterministic hash-based selection for reproducibility.
        """
        logging.info(f"Initializing diversity reference set from {len(self.programs)} programs")

        with self._population_lock:
            if len(self.programs) < self.diversity_reference_size:
                logging.warning(
                    f"Population ({len(self.programs)}) smaller than reference size "
                    f"({self.diversity_reference_size}) - using all programs"
                )
                selected_programs = list(self.programs.values())
            else:
                # Deterministic hash-based selection
                program_hashes = [(deterministic_code_hash(p.code), p) for p in self.programs.values()]

                # Sort by hash (deterministic across runs)
                program_hashes.sort(key=lambda x: x[0])

                # Select evenly spaced programs
                step = len(program_hashes) // self.diversity_reference_size
                selected_programs = [program_hashes[i * step][1] for i in range(self.diversity_reference_size)]

            # Store CODE snapshots (immune to deletion)
            self.diversity_reference_codes = [p.code for p in selected_programs]

            logging.info(
                f"Diversity reference set initialized with {len(self.diversity_reference_codes)} code snapshots"
            )

    # ========================================================================
    # SAMPLING AND RETRIEVAL
    # ========================================================================

    def is_empty(self) -> bool:
        """Check if the database has any programs"""
        return len(self.programs) == 0

    def sample(self) -> Tuple[Program, List[Program]]:
        """
        Sample one parent and multiple inspiration programs for evolution.

        Uses a three-strategy approach for parent selection:

        - EXPLORATION (exploration_ratio): Sample from underexplored regions
          to discover new optimization strategies
        - EXPLOITATION (exploitation_ratio): Sample from elite programs
          to refine and combine the best solutions
        - RANDOM (remaining): Completely random sampling for novelty

        Inspirations are sampled from DIFFERENT optimization niches to
        encourage cross-pollination of optimization strategies.

        Returns:
            Tuple of (parent_program, inspiration_programs)

        Raises:
            ValueError: If database is empty and no programs available
        """
        with self._population_lock:
            if not self.programs:
                raise ValueError("Cannot sample from empty database")

            # Sample parent using exploration/exploitation strategy
            parent = self._sample_parent()

            # Sample inspirations from diverse optimization niches
            inspirations = self._sample_diverse_inspirations(parent, n=self.num_inspirations)

            logging.debug(f"Sampled parent {parent.id} and {len(inspirations)} diverse inspirations")
            return parent, inspirations

    def _sample_parent(self) -> Program:
        """
        Sample a parent program using exploration/exploitation/random strategy.

        Returns:
            Selected parent program
        """
        rand_val = random.random()

        if rand_val < self.exploration_ratio:
            # EXPLORATION: Sample from underexplored regions or current island
            return self._sample_exploration_parent()
        elif rand_val < self.exploration_ratio + self.exploitation_ratio:
            # EXPLOITATION: Sample from elite programs (MAP-Elites archive)
            return self._sample_exploitation_parent()
        else:
            # RANDOM: Sample completely randomly for novelty
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """
        Sample a parent for exploration - prioritizing diversity and broad search.

        Unlike exploitation (which focuses on elites), exploration samples from the
        full population to discover new optimization patterns. This mirrors how the
        default database's exploration works with its island model.

        Prioritizes:
        1. Programs from current island (full population, not just elites)
        2. Programs from low-quality cells (to improve weak regions)
        3. Fallback to any program
        """
        # PRIMARY: Sample from current island (like default DB does)
        # This gives us diversity because the island contains ALL programs, not just elites
        if self.use_islands:
            island_programs = [pid for pid in self.islands[self.current_island] if pid in self.programs]
            if island_programs:
                selected_id = random.choice(island_programs)
                logging.debug(
                    f"Exploration: selected parent from island {self.current_island} (population size: {len(island_programs)})"
                )
                return self.programs[selected_id]

        # SECONDARY: If islands are empty or disabled, try low-quality cells to improve them
        # These are cells that HAVE programs but with below-median scores
        underexplored = self._get_underexplored_cells_internal(n=10)
        low_quality_cells = [r for r in underexplored if "current_score" in r]  # Has score = exists in feature_map

        if low_quality_cells:
            region = random.choice(low_quality_cells)
            coord_key = (
                region["memory_opt"],
                region["compute_opt"],
                region["parallelism_opt"],
                region.get("esimd_opt", 0),
            )
            if coord_key in self.feature_map:
                prog_id = self.feature_map[coord_key]
                if prog_id in self.programs:
                    logging.debug(
                        f"Exploration: selected parent from low-quality cell {coord_key} (score: {region['current_score']:.4f})"
                    )
                    return self.programs[prog_id]

        # FINAL FALLBACK: Sample from all programs for maximum diversity
        if self.programs:
            selected = random.choice(list(self.programs.values()))
            logging.debug(f"Exploration: selected random parent from full population")
            return selected

        # Emergency fallback
        raise ValueError("No programs available for exploration sampling")

    def _sample_exploitation_parent(self) -> Program:
        """
        Sample a parent for exploitation - targeting elite programs.

        Prioritizes:
        1. Elite programs from the MAP-Elites archive
        2. Best program from current island
        3. Global best program
        """
        # Get elite programs from feature map
        elite_ids = list(self.feature_map.values())
        valid_elites = [pid for pid in elite_ids if pid in self.programs]

        if valid_elites:
            # Fitness-proportional sampling among elites
            elites = [self.programs[pid] for pid in valid_elites]
            scores = np.array([get_main_score(p.metrics or {}) for p in elites])

            if np.sum(scores) > 0:
                # Softmax for stable probability computation
                max_score = np.max(scores)
                exp_scores = np.exp(scores - max_score)
                probabilities = exp_scores / np.sum(exp_scores)

                if not np.any(np.isnan(probabilities)):
                    idx = np.random.choice(len(elites), p=probabilities)
                    logging.debug(f"Exploitation: selected elite parent with score {scores[idx]:.4f}")
                    return elites[idx]

            # Fallback to random elite
            return random.choice(elites)

        # Fallback: best program from island or globally
        if self.use_islands and self.island_best_programs[self.current_island]:
            best_id = self.island_best_programs[self.current_island]
            if best_id in self.programs:
                return self.programs[best_id]

        if self.best_program_id and self.best_program_id in self.programs:
            return self.programs[self.best_program_id]

        return random.choice(list(self.programs.values()))

    def _sample_random_parent(self) -> Program:
        """
        Sample a completely random parent for novelty.
        """
        return random.choice(list(self.programs.values()))

    def _sample_diverse_inspirations(self, parent: Program, n: int) -> List[Program]:
        """
        Sample inspiration programs from DIVERSE optimization niches.

        The key insight of MAP-Elites is that innovations come from
        combining ideas across different behavioral niches. This method
        explicitly samples inspirations from different optimization
        levels to encourage cross-pollination.

        Sampling priority:
        1. Elite programs from HIGHER optimization levels than parent
           (to encourage upward evolution of optimization sophistication)
        2. Elite programs from different optimization cells than parent
        3. High-performing programs with different optimization patterns
        4. Random programs for additional diversity

        Args:
            parent: The parent program (to avoid sampling from same niche)
            n: Number of inspirations to sample

        Returns:
            List of inspiration programs from diverse niches
        """
        if n <= 0:
            return []

        inspirations: List[Program] = []
        used_ids: Set[str] = {parent.id}

        # Get parent's optimization coordinates
        parent_coords = parent.metadata.get("feature_coords")
        if parent_coords:
            if len(parent_coords) == 3:
                parent_coords = tuple(parent_coords) + (0,)
            else:
                parent_coords = tuple(parent_coords)
        parent_opt_sum = sum(parent_coords) if parent_coords else 0

        # Strategy 1: Sample elites from DIFFERENT optimization cells
        # CRITICAL: Prioritize programs with HIGHER optimization levels
        # This is the key insight for performance evolution - learn from more optimized code
        elite_slots = max(1, n // 2)  # Reserve half for diverse elites

        different_niche_elites = []
        for coord_key, prog_id in self.feature_map.items():
            if prog_id not in used_ids and prog_id in self.programs:
                # Check if this is a different niche than parent
                if parent_coords is None or coord_key != parent_coords:
                    program = self.programs[prog_id]
                    perf_score = get_main_score(program.metrics or {})

                    # Calculate "optimization level bonus" - programs at higher
                    # optimization levels get a boost in selection priority.
                    # This encourages the LLM to learn from more sophisticated
                    # optimization patterns (SLM tiling, sub-groups, async copy, etc.)
                    opt_level_sum = sum(coord_key[:4]) if len(coord_key) >= 4 else sum(coord_key)

                    # Bonus for being at higher optimization level than parent
                    level_bonus = max(0, opt_level_sum - parent_opt_sum) * 0.1

                    # Combined score: performance + optimization level bonus
                    combined_priority = perf_score + level_bonus

                    different_niche_elites.append((combined_priority, perf_score, opt_level_sum, program))

        # Sort by combined priority (performance + optimization level bonus)
        different_niche_elites.sort(key=lambda x: x[0], reverse=True)

        for priority, perf_score, opt_sum, program in different_niche_elites[:elite_slots]:
            if len(inspirations) >= n:
                break
            inspirations.append(program)
            used_ids.add(program.id)
            logging.debug(f"Added diverse elite: score={perf_score:.4f}, opt_level={opt_sum}")

        # Strategy 2: Add top programs (may overlap with strategy 1)
        if len(inspirations) < n:
            top_programs = self.get_top_programs(n=n - len(inspirations) + 3)
            for program in top_programs:
                if program.id not in used_ids:
                    inspirations.append(program)
                    used_ids.add(program.id)
                    if len(inspirations) >= n:
                        break

        # Strategy 3: Fill remaining slots with programs from other islands
        if len(inspirations) < n and self.use_islands:
            for island_idx in range(self.num_islands):
                if island_idx == self.current_island:
                    continue
                # Get best from other islands
                island_best_id = self.island_best_programs[island_idx]
                if island_best_id and island_best_id not in used_ids and island_best_id in self.programs:
                    inspirations.append(self.programs[island_best_id])
                    used_ids.add(island_best_id)
                    if len(inspirations) >= n:
                        break

        # Strategy 4: Fill any remaining slots with random correct programs
        if len(inspirations) < n:
            available = [
                p
                for p in self.programs.values()
                if p.id not in used_ids and p.metrics and p.metrics.get("correctness_success", False)
            ]
            if available:
                random.shuffle(available)
                for program in available:
                    inspirations.append(program)
                    used_ids.add(program.id)
                    if len(inspirations) >= n:
                        break

        logging.debug(f"Sampled {len(inspirations)} diverse inspirations from different optimization niches")
        return inspirations

    def _get_underexplored_cells_internal(self, n: int = 5) -> List[Dict[str, int]]:
        """
        Internal method to get underexplored cells (assumes lock is held).

        Returns cells that either:
        1. Are empty (no elite yet)
        2. Have low-quality elites (below median score)

        Args:
            n: Maximum number of cells to return

        Returns:
            List of cell coordinates as dicts
        """
        empty_cells = []
        low_quality_cells = []

        # Calculate median score for quality threshold
        all_scores = []
        for prog_id in self.feature_map.values():
            if prog_id in self.programs:
                score = get_main_score(self.programs[prog_id].metrics or {})
                all_scores.append(score)

        median_score = np.median(all_scores) if all_scores else 0.5

        # Scan grid
        for mem in range(4):
            for comp in range(4):
                for par in range(4):
                    for esimd in range(4):
                        coord_key = (mem, comp, par, esimd)
                        if coord_key not in self.feature_map:
                            empty_cells.append(
                                {"memory_opt": mem, "compute_opt": comp, "parallelism_opt": par, "esimd_opt": esimd}
                            )
                        else:
                            prog_id = self.feature_map[coord_key]
                            if prog_id in self.programs:
                                score = get_main_score(self.programs[prog_id].metrics or {})
                                if score < median_score:
                                    low_quality_cells.append(
                                        {
                                            "memory_opt": mem,
                                            "compute_opt": comp,
                                            "parallelism_opt": par,
                                            "esimd_opt": esimd,
                                            "current_score": score,
                                        }
                                    )

        # Prioritize empty cells, then low-quality
        candidates = empty_cells + low_quality_cells
        if len(candidates) > n:
            return random.sample(candidates, n)
        return candidates

    def _sample_programs(
        self,
        num_programs: int,
        temperature: float = 1.0,
        sampling_method: str = "fitness_proportional",
        island_id: Optional[int] = None,
    ) -> List[Program]:
        """
        Sample programs using specified strategy (string-based API for backward compatibility).

        Args:
            num_programs: Number of programs to sample
            temperature: Sampling temperature (higher = more random)
            sampling_method: Strategy to use (string):
                - "fitness_proportional": Sample based on fitness scores
                - "uniform": Uniform random sampling
                - "elite"/"archive": Sample only from feature map elites
            island_id: If provided, sample only from this island

        Returns:
            List of sampled programs
        """
        # Convert string to enum for internal processing
        method_map = {
            "fitness_proportional": SamplingMethod.FITNESS_PROPORTIONAL,
            "uniform": SamplingMethod.UNIFORM,
            "elite": SamplingMethod.ELITE,
            "archive": SamplingMethod.ARCHIVE,
        }
        method = method_map.get(sampling_method)
        if method is None:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        with self._population_lock:
            return self._sample_programs_internal(num_programs, temperature, method, island_id)

    def _sample_programs_internal(
        self,
        num_programs: int,
        temperature: float = 1.0,
        sampling_method: SamplingMethod = SamplingMethod.FITNESS_PROPORTIONAL,
        island_id: Optional[int] = None,
    ) -> List[Program]:
        """
        Internal sampling implementation using type-safe enums.

        Note: Assumes _population_lock is already held by caller.

        Args:
            num_programs: Number of programs to sample
            temperature: Sampling temperature (higher = more random)
            sampling_method: Strategy to use (enum)
            island_id: If provided, sample only from this island

        Returns:
            List of sampled programs
        """
        # Determine candidate pool
        if island_id is not None:
            if not (0 <= island_id < self.num_islands):
                raise ValueError(f"Invalid island_id: {island_id}")
            candidate_ids = list(self.islands[island_id])
        else:
            candidate_ids = list(self.programs.keys())

        if not candidate_ids:
            logging.warning("No programs available for sampling")
            return []

        sampled_ids: List[str] = []

        # Apply sampling method
        if sampling_method == SamplingMethod.UNIFORM:
            num_to_sample = min(num_programs, len(candidate_ids))
            sampled_ids = random.sample(candidate_ids, num_to_sample)

        elif sampling_method in (SamplingMethod.ELITE, SamplingMethod.ARCHIVE):
            elite_ids = list(set(self.feature_map.values()))
            if island_id is not None:
                elite_ids = [pid for pid in elite_ids if pid in self.islands[island_id]]

            if not elite_ids:
                logging.warning("No elites available - falling back to uniform")
                num_to_sample = min(num_programs, len(candidate_ids))
                sampled_ids = random.sample(candidate_ids, num_to_sample)
            else:
                num_to_sample = min(num_programs, len(elite_ids))
                sampled_ids = random.sample(elite_ids, num_to_sample)

        elif sampling_method == SamplingMethod.FITNESS_PROPORTIONAL:
            candidates = [(pid, self.programs[pid]) for pid in candidate_ids]
            scores = [get_main_score(prog.metrics or {}) for _, prog in candidates]

            if all(s == 0 for s in scores):
                num_to_sample = min(num_programs, len(candidate_ids))
                sampled_ids = random.sample(candidate_ids, num_to_sample)
            else:
                scores_array = np.array(scores, dtype=np.float64)

                # Softmax with numerical stability and correct temperature application
                # Temperature is applied AFTER max subtraction for numerical stability:
                # softmax(x/T) = exp((x - max(x))/T) / sum(exp((x - max(x))/T))
                max_score = np.max(scores_array)
                scaled_scores = (scores_array - max_score) / max(temperature, 1e-8)
                exp_scores = np.exp(scaled_scores)
                probabilities = exp_scores / np.sum(exp_scores)

                # Handle potential NaN from numerical issues
                if np.any(np.isnan(probabilities)):
                    logging.warning("NaN in sampling probabilities - falling back to uniform")
                    num_to_sample = min(num_programs, len(candidate_ids))
                    sampled_ids = random.sample(candidate_ids, num_to_sample)
                else:
                    num_to_sample = min(num_programs, len(candidate_ids))
                    sampled_indices = np.random.choice(
                        len(candidates), size=num_to_sample, replace=False, p=probabilities
                    )
                    sampled_ids = [candidates[i][0] for i in sampled_indices]
        else:
            # This should never happen with enum, but kept for safety
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        return [self.programs[pid] for pid in sampled_ids if pid in self.programs]

    def get_best_program(self, island_id: Optional[int] = None) -> Optional[Program]:
        """
        Get the best program globally or from a specific island.

        Args:
            island_id: If provided, return best from this island only

        Returns:
            Best program or None if database is empty
        """
        with self._population_lock:
            if island_id is not None:
                best_id = self.island_best_programs[island_id]
                return self.programs.get(best_id) if best_id else None
            else:
                best_id = self.best_program_id
                return self.programs.get(best_id) if best_id else None

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:
        """
        Get the top N programs based on a metric.

        This method is essential for the controller to get high-quality inspirations.
        It can filter by island for island-based evolution strategies.

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking (uses combined_score if None)
            island_idx: If specified, only return programs from this island

        Returns:
            List of top programs sorted by score (descending)
        """
        with self._population_lock:
            # Validate island_idx parameter
            if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands)):
                raise IndexError(f"Island index {island_idx} is out of range (0-{len(self.islands)-1})")

            if not self.programs:
                return []

            # Get candidate programs
            if island_idx is not None:
                # Island-specific query
                candidate_ids = [pid for pid in self.islands[island_idx] if pid in self.programs]
                candidates = [self.programs[pid] for pid in candidate_ids]
            else:
                # Global query
                candidates = list(self.programs.values())

            if not candidates:
                return []

            if metric:
                # Sort by specific metric
                sorted_programs = sorted(
                    [p for p in candidates if p.metrics and metric in p.metrics],
                    key=lambda p: p.metrics[metric],
                    reverse=True,
                )
            else:
                # Sort by combined_score (primary) or average of numeric metrics (fallback)
                sorted_programs = sorted(
                    candidates,
                    key=lambda p: get_main_score(p.metrics or {}),
                    reverse=True,
                )

            return sorted_programs[:n]

    def get_archive(self) -> List[Program]:
        """
        Get all elite programs from the feature map.

        Uses caching to avoid repeated list construction.
        Thread-safe with atomic cache operations.

        Returns:
            List of elite programs (one per occupied cell)
        """
        # Fast path: check cache validity without lock
        with self._archive_cache_lock:
            if self._archive_cache_valid and self._archive_cache is not None:
                return list(self._archive_cache)  # Return copy to prevent mutation

        # Slow path: rebuild cache under population lock
        with self._population_lock:
            with self._archive_cache_lock:
                # Double-check after acquiring locks
                if self._archive_cache_valid and self._archive_cache is not None:
                    return list(self._archive_cache)

                elite_ids = set(self.feature_map.values())
                self._archive_cache = [self.programs[pid] for pid in elite_ids if pid in self.programs]
                self._archive_cache_valid = True
                return list(self._archive_cache)

    def _invalidate_archive_cache(self) -> None:
        """
        Atomically invalidate the archive cache when feature map changes.

        Note: This method assumes _population_lock is held by caller.
        Uses _archive_cache_lock for atomicity of the invalidation itself.
        """
        with self._archive_cache_lock:
            self._archive_cache_valid = False
            self._archive_cache = None

    def get_all_programs(self) -> List[Program]:
        """Get all programs in the database"""
        with self._population_lock:
            return list(self.programs.values())

    # ========================================================================
    # ISLAND MODEL OPERATIONS
    # ========================================================================

    @property
    def use_islands(self) -> bool:
        """Check if island model is enabled"""
        return self.num_islands > 1

    def increase_island_counter_and_switch(self):
        """Increase counter and switch island if threshold reached"""
        if self.current_island_counter >= self.programs_per_island:
            self.next_island()
            self.current_island_counter = 0
        self.current_island_counter += 1
        self.increment_island_generation()

    def next_island(self) -> int:
        """Move to the next island in round-robin fashion"""
        self.current_island = (self.current_island + 1) % self.num_islands
        logging.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Increment generation counter for an island"""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1

    def should_migrate(self) -> bool:
        """Check if migration should occur"""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval

    def migrate_programs(self, copy_mode: bool = True) -> None:
        """
        Perform migration between islands.

        Migrates top programs from each island to the next island (ring topology).

        Args:
            copy_mode: If True (default), programs are COPIED to target island
                      (existing in both source and target). If False, programs
                      are MOVED (removed from source).

        Note:
            Copy mode is generally preferred as it maintains population diversity
            while still spreading good solutions. Move mode can cause population
            collapse in source islands.
        """
        if not self.use_islands or self.num_islands < 2:
            return

        with self._population_lock:
            logging.info(f"Performing island migration (iteration {self.last_iteration})")

            num_migrants = max(1, int(self.migration_rate * len(self.programs) / self.num_islands))
            total_migrated = 0

            for island_id in range(self.num_islands):
                island_programs = [(pid, self.programs[pid]) for pid in self.islands[island_id] if pid in self.programs]

                if not island_programs:
                    continue

                # Sort by score (best first)
                island_programs.sort(key=lambda x: get_main_score(x[1].metrics or {}), reverse=True)

                # Select migrants (top performers)
                migrants = island_programs[:num_migrants]

                # Migrate to next island (ring topology)
                target_island = (island_id + 1) % self.num_islands

                migrated_count = 0
                for prog_id, program in migrants:
                    # Add to target island
                    if prog_id not in self.islands[target_island]:
                        self.islands[target_island].add(prog_id)
                        self._update_island_best(program, target_island)
                        migrated_count += 1

                        # Optionally remove from source (move vs copy)
                        if not copy_mode:
                            self.islands[island_id].discard(prog_id)
                            # Update program's island metadata
                            program.metadata["island"] = target_island

                if migrated_count > 0:
                    logging.debug(
                        f"  {'Copied' if copy_mode else 'Moved'} {migrated_count} programs "
                        f"from island {island_id} to island {target_island}"
                    )
                    total_migrated += migrated_count

            self.last_migration_generation = max(self.island_generations)
            logging.info(f"Migration complete: {total_migrated} programs transferred")

    def log_island_status(self) -> None:
        """Log current status of all islands"""
        logging.info("Island Status:")
        for i in range(self.num_islands):
            is_current = i == self.current_island
            pop_size = len(self.islands[i])
            current_marker = " *" if is_current else "  "
            logging.info(f"{current_marker} Island {i}: {pop_size} programs, " f"gen={self.island_generations[i]}")

    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for monitoring.

        Returns:
            Dict with population stats, archive coverage, etc.
        """
        with self._population_lock:
            stats = {
                "total_programs": len(self.programs),
                "archive_size": len(self.feature_map),
                "max_archive_size": self.max_archive_size,
                "coverage_percent": 100.0 * len(self.feature_map) / self.max_archive_size,
                "grid_shape": self.grid_shape,
            }

            # Best program info
            if self.best_program_id:
                best = self.programs.get(self.best_program_id)
                if best:
                    stats["best_score"] = get_main_score(best.metrics or {})

            # Island statistics
            if self.use_islands:
                stats["islands"] = [
                    {
                        "id": i,
                        "size": len(island),
                        "best_score": (
                            get_main_score(self.programs[best_id].metrics or {})
                            if (best_id := self.island_best_programs[i]) is not None and best_id in self.programs
                            else None
                        ),
                    }
                    for i, island in enumerate(self.islands)
                ]

            return stats

    def get_grid_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the MAP-Elites grid.

        Returns:
            Dictionary with coverage, scores, and dimension-specific stats
        """
        with self._population_lock:
            # Get occupied cells and scores
            occupied = len(self.feature_map)
            scores = []
            coords_by_dim = {
                "memory_opt": [],
                "compute_opt": [],
                "parallelism_opt": [],
                "esimd_opt": [],
            }

            for coords, prog_id in self.feature_map.items():
                if prog_id in self.programs:
                    program = self.programs[prog_id]
                    scores.append(get_main_score(program.metrics))

                    coords_by_dim["memory_opt"].append(coords[0])
                    coords_by_dim["compute_opt"].append(coords[1])
                    coords_by_dim["parallelism_opt"].append(coords[2])
                    # Handle legacy 3-tuple coords
                    if len(coords) > 3:
                        coords_by_dim["esimd_opt"].append(coords[3])
                    else:
                        coords_by_dim["esimd_opt"].append(0)

            # Per-dimension coverage
            dim_coverage = {}
            for dim_name in ["memory_opt", "compute_opt", "parallelism_opt", "esimd_opt"]:
                if coords_by_dim[dim_name]:
                    unique_values = set(coords_by_dim[dim_name])
                    dim_coverage[dim_name] = {
                        "unique_levels_used": len(unique_values),
                        "total_levels": 4,
                        "coverage_percent": 100 * len(unique_values) / 4,
                    }

            # Count ESIMD usage
            esimd_counts = {level: coords_by_dim["esimd_opt"].count(level) for level in range(4)}
            esimd_usage = {
                "level_0_standard_sycl": esimd_counts[0],
                "level_1_basic_esimd": esimd_counts[1],
                "level_2_optimized_esimd": esimd_counts[2],
                "level_3_expert_esimd": esimd_counts[3],
                "total_esimd_kernels": sum(esimd_counts[k] for k in [1, 2, 3]),
            }

            stats = {
                "dimensions": ["memory_opt", "compute_opt", "parallelism_opt", "esimd_opt"],
                "grid_shape": self.grid_shape,
                "total_cells": self.max_archive_size,
                "occupied_cells": occupied,
                "coverage_percent": 100 * occupied / self.max_archive_size,
                "dimension_coverage": dim_coverage,
                "esimd_usage": esimd_usage,
            }

            if scores:
                stats["score_stats"] = {
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "median": float(np.median(scores)),
                }

            return stats

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save database state to checkpoint file.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        with self._population_lock:
            checkpoint_data = {
                "version": "1.1",  # Updated version for gradient support
                "timestamp": time.time(),
                "grid_type": "optimization_aware",
                "last_iteration": self.last_iteration,
                "gradient_tracking_enabled": self._use_gradient_tracking,
                "programs": [
                    {
                        "id": prog.id,
                        "code": prog.code,
                        "language": prog.language,
                        "metrics": prog.metrics,
                        "metadata": prog.metadata,
                    }
                    for prog in self.programs.values()
                ],
                "feature_map": {
                    f"{k[0]},{k[1]},{k[2]},{k[3] if len(k) > 3 else 0}": v for k, v in self.feature_map.items()
                },
                "best_program_id": self.best_program_id,
                "statistics": self.get_statistics(),
            }

            # Add transition statistics if available
            if self._use_gradient_tracking and self._transition_tracker is not None:
                checkpoint_data["transition_statistics"] = self._transition_tracker.get_statistics()

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            logging.info(f"Saved checkpoint to: {checkpoint_path}")
            logging.info(f"  Programs: {len(self.programs)}")
            logging.info(f"  Archive size: {len(self.feature_map)}")

            # Also save transition tracker checkpoint
            if self._use_gradient_tracking and self._transition_tracker is not None:
                transition_checkpoint_path = checkpoint_path.replace(".json", "_transitions.json")
                self._transition_tracker.save_checkpoint(transition_checkpoint_path)
                logging.info(f"  Saved transition tracker to: {transition_checkpoint_path}")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def visualize_grid(
        self, show_scores: bool = True, save_path: Optional[str] = None, include_esimd: bool = True
    ) -> None:
        """
        Visualize the MAP-Elites grid.

        When include_esimd=True (default):
            Creates a 4×4 grid of 2D slices (4×4×4×4 total):
            - Rows: ESIMD optimization levels (0-3)
            - Columns: Parallelism optimization levels (0-3)
            - Each cell: 4×4 heatmap of memory_opt × compute_opt

        When include_esimd=False:
            Creates a 1×4 grid of 2D slices (4×4×4 total):
            - Columns: Parallelism optimization levels (0-3)
            - Each cell: 4×4 heatmap of memory_opt × compute_opt

        Args:
            show_scores: If True, display performance scores in occupied cells
            save_path: Optional path to save the visualization
            include_esimd: If True, include ESIMD as a dimension (default True)
        """

        import matplotlib.pyplot as plt  # Import here since we know it's available

        with self._population_lock:
            if include_esimd:
                # 4D visualization: [esimd_opt, parallelism_opt, compute_opt, memory_opt]
                grid = np.full((4, 4, 4, 4), np.nan)

                # Fill grid with scores
                for coords, prog_id in self.feature_map.items():
                    if prog_id in self.programs:
                        program = self.programs[prog_id]
                        score = get_main_score(program.metrics)
                        # coords = (memory_opt, compute_opt, parallelism_opt, esimd_opt)
                        grid[coords[3], coords[2], coords[1], coords[0]] = score

                # Calculate statistics
                total_cells = 256  # 4×4×4×4
                occupied_cells = int(np.sum(~np.isnan(grid)))

                # Create 4×4 grid of subplots (rows=esimd_opt, cols=parallelism_opt)
                fig, axes = plt.subplots(4, 4, figsize=(20, 20))

                # Find global min/max for consistent color scale
                vmin = float(np.nanmin(grid)) if occupied_cells > 0 else 0.0
                vmax = float(np.nanmax(grid)) if occupied_cells > 0 else 1.0

                # ESIMD level labels for row titles
                esimd_labels = [
                    "ESIMD L0: Standard SYCL",
                    "ESIMD L1: Basic (simd<T,N>)",
                    "ESIMD L2: Optimized (LSC)",
                    "ESIMD L3: Expert (DPAS/XMX)",
                ]

                im = None  # Track last image for colorbar
                # Plot each 2D slice (memory_opt × compute_opt)
                for esimd_level in range(4):
                    for par_level in range(4):
                        ax = axes[esimd_level, par_level]
                        # slice_data[compute_opt, memory_opt]
                        slice_data = grid[esimd_level, par_level]
                        slice_occupied = int(np.sum(~np.isnan(slice_data)))

                        # Create heatmap
                        im = ax.imshow(slice_data, cmap="viridis", aspect="equal", origin="lower", vmin=vmin, vmax=vmax)

                        # Add cell values if requested
                        if show_scores:
                            for j in range(4):
                                for i in range(4):
                                    if not np.isnan(slice_data[j, i]):
                                        ax.text(
                                            i,
                                            j,
                                            f"{slice_data[j, i]:.2f}",
                                            ha="center",
                                            va="center",
                                            color="white",
                                            fontsize=8,
                                            fontweight="bold",
                                        )

                        # Labels and ticks
                        ax.set_xticks(range(4))
                        ax.set_yticks(range(4))

                        # Only show axis labels on edges
                        if esimd_level == 3:
                            ax.set_xlabel("memory_opt", fontsize=10)
                        else:
                            ax.set_xticklabels([])

                        if par_level == 0:
                            ax.set_ylabel("compute_opt", fontsize=10)
                        else:
                            ax.set_yticklabels([])

                        # Title for each subplot
                        ax.set_title(f"par={par_level} ({slice_occupied}/16)", fontsize=9)

                # Add row labels for ESIMD levels (on the right side)
                for esimd_level in range(4):
                    axes[esimd_level, 3].annotate(
                        esimd_labels[esimd_level],
                        xy=(1.15, 0.5),
                        xycoords="axes fraction",
                        fontsize=10,
                        fontweight="bold",
                        rotation=270,
                        va="center",
                        ha="left",
                    )

                # Add column header for parallelism levels
                for par_level in range(4):
                    axes[0, par_level].annotate(
                        f"parallelism_opt = {par_level}",
                        xy=(0.5, 1.15),
                        xycoords="axes fraction",
                        fontsize=10,
                        fontweight="bold",
                        ha="center",
                        va="bottom",
                    )

                # Add colorbar
                if im is not None:
                    fig.subplots_adjust(right=0.88, top=0.92, hspace=0.15, wspace=0.1)
                    cbar_ax = fig.add_axes((0.90, 0.15, 0.02, 0.7))
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Performance Score", rotation=270, labelpad=20, fontsize=12)

                # Overall title
                fig.suptitle(
                    f"Optimization-Aware MAP-Elites Grid (4×4×4×4)\n"
                    f"Dimensions: memory_opt × compute_opt × parallelism_opt × esimd_opt\n"
                    f"Coverage: {occupied_cells}/{total_cells} cells ({100 * occupied_cells / total_cells:.1f}%)",
                    fontsize=14,
                    fontweight="bold",
                    y=0.98,
                )
            else:
                # 3D visualization (no ESIMD): [parallelism_opt, compute_opt, memory_opt]
                grid = np.full((4, 4, 4), np.nan)

                # Fill grid with scores (aggregate across esimd_opt dimension)
                for coords, prog_id in self.feature_map.items():
                    if prog_id in self.programs:
                        program = self.programs[prog_id]
                        score = get_main_score(program.metrics)
                        # coords = (memory_opt, compute_opt, parallelism_opt, esimd_opt)
                        # Only use first 3 dimensions, take max score if multiple esimd levels
                        coord_3d = (coords[2], coords[1], coords[0])  # (parallelism_opt, compute_opt, memory_opt)
                        if np.isnan(grid[coord_3d]) or score > grid[coord_3d]:
                            grid[coord_3d] = score

                # Calculate statistics
                total_cells = 64  # 4×4×4
                occupied_cells = int(np.sum(~np.isnan(grid)))

                # Create 1×4 grid of subplots (cols=parallelism_opt)
                fig, axes = plt.subplots(1, 4, figsize=(20, 6))

                # Find global min/max for consistent color scale
                vmin = float(np.nanmin(grid)) if occupied_cells > 0 else 0.0
                vmax = float(np.nanmax(grid)) if occupied_cells > 0 else 1.0

                im = None  # Track last image for colorbar
                # Plot each 2D slice (memory_opt × compute_opt)
                for par_level in range(4):
                    ax = axes[par_level]
                    # slice_data[compute_opt, memory_opt]
                    slice_data = grid[par_level]
                    slice_occupied = int(np.sum(~np.isnan(slice_data)))

                    # Create heatmap
                    im = ax.imshow(slice_data, cmap="viridis", aspect="equal", origin="lower", vmin=vmin, vmax=vmax)

                    # Add cell values if requested
                    if show_scores:
                        for j in range(4):
                            for i in range(4):
                                if not np.isnan(slice_data[j, i]):
                                    ax.text(
                                        i,
                                        j,
                                        f"{slice_data[j, i]:.2f}",
                                        ha="center",
                                        va="center",
                                        color="white",
                                        fontsize=8,
                                        fontweight="bold",
                                    )

                    # Labels and ticks
                    ax.set_xticks(range(4))
                    ax.set_yticks(range(4))
                    ax.set_xlabel("memory_opt", fontsize=10)

                    if par_level == 0:
                        ax.set_ylabel("compute_opt", fontsize=10)
                    else:
                        ax.set_yticklabels([])

                    # Title for each subplot
                    ax.set_title(f"parallelism_opt = {par_level} ({slice_occupied}/16)", fontsize=10)

                # Add colorbar
                if im is not None:
                    fig.subplots_adjust(right=0.88, top=0.85, wspace=0.15)
                    cbar_ax = fig.add_axes((0.90, 0.15, 0.02, 0.6))
                    cbar = fig.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Performance Score", rotation=270, labelpad=20, fontsize=12)

                # Overall title
                fig.suptitle(
                    f"Optimization-Aware MAP-Elites Grid (4×4×4)\n"
                    f"Dimensions: memory_opt × compute_opt × parallelism_opt\n"
                    f"Coverage: {occupied_cells}/{total_cells} cells ({100 * occupied_cells / total_cells:.1f}%)",
                    fontsize=14,
                    fontweight="bold",
                    y=0.98,
                )

            # Save or show
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logging.info(f"Saved grid visualization to: {save_path}")
                plt.close()
            else:
                plt.show()

    def save_grid_visualization(self, iteration: Optional[int] = None, include_esimd: bool = True) -> None:
        """
        Save MAP-Elites grid visualization to the output directory.

        Args:
            iteration: Iteration number for filename
            include_esimd: If True, include ESIMD as a dimension (default True)
        """
        if self.output_dir is None:
            logging.warning("Cannot save visualization: output_dir not set")
            return

        if iteration is None:
            iteration = self.last_iteration

        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        filename = f"mapelites_grid_iter_{iteration:04d}.png"
        save_path = os.path.join(viz_dir, filename)

        self.visualize_grid(show_scores=True, save_path=save_path, include_esimd=include_esimd)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def update_program_metadata(self, program_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a program"""
        with self._population_lock:
            if program_id in self.programs:
                self.programs[program_id].metadata.update(metadata)

    # ========================================================================
    # OPTIMIZATION-AWARE EXPLORATION
    # ========================================================================

    def get_underexplored_regions(self, n: int = 5, strategy: str = "empty_first") -> List[Dict[str, int]]:
        """
        Identify underexplored regions of the optimization space.

        Args:
            n: Maximum number of regions to return
            strategy: Selection strategy:
                - "empty_first": Prioritize completely empty cells
                - "low_quality": Include cells with low-performing elites
                - "balanced": Mix of empty and low-quality cells

        Returns:
            List of target optimization profiles for underexplored regions
        """
        with self._population_lock:
            empty_cells: List[Dict[str, int]] = []
            low_quality_cells: List[Dict[str, Any]] = []

            # Scan all possible coordinates (4D grid including ESIMD)
            for mem in range(4):
                for comp in range(4):
                    for par in range(4):
                        for esimd in range(4):
                            coord_key = (mem, comp, par, esimd)

                            if coord_key not in self.feature_map:
                                # Empty cell
                                empty_cells.append(
                                    {
                                        "memory_opt": mem,
                                        "compute_opt": comp,
                                        "parallelism_opt": par,
                                        "esimd_opt": esimd,
                                    }
                                )
                            elif strategy in ["low_quality", "balanced"]:
                                # Check quality of current elite
                                prog_id = self.feature_map[coord_key]
                                if prog_id in self.programs:
                                    program = self.programs[prog_id]
                                    score = get_main_score(program.metrics or {})

                                    # Consider low-quality if below median
                                    if score < 0.5:  # Threshold can be adjusted
                                        low_quality_cells.append(
                                            {
                                                "memory_opt": mem,
                                                "compute_opt": comp,
                                                "parallelism_opt": par,
                                                "esimd_opt": esimd,
                                                "current_score": score,
                                            }
                                        )

            # Select regions based on strategy
            if strategy == "empty_first":
                candidates = empty_cells + low_quality_cells
            elif strategy == "low_quality":
                candidates = low_quality_cells + empty_cells
            elif strategy == "balanced":
                # Mix empty and low-quality
                half_n = n // 2
                candidates = random.sample(empty_cells, min(half_n, len(empty_cells))) + random.sample(
                    low_quality_cells, min(n - half_n, len(low_quality_cells))
                )
                return candidates
            else:
                candidates = empty_cells

            # Return random sample
            if len(candidates) > n:
                return random.sample(candidates, n)
            return candidates

    def get_parent_optimization_profile(self, parent: Program) -> Optional[Dict[str, int]]:
        """
        Extract optimization profile from parent program.

        Args:
            parent: Parent program

        Returns:
            Dict with memory_opt, compute_opt, parallelism_opt, esimd_opt or None
        """
        coords = parent.metadata.get("feature_coords")
        if coords:
            # Handle legacy 3-tuple coords
            if len(coords) == 3:
                return {
                    "memory_opt": coords[0],
                    "compute_opt": coords[1],
                    "parallelism_opt": coords[2],
                    "esimd_opt": 0,
                }
            return {
                "memory_opt": coords[0],
                "compute_opt": coords[1],
                "parallelism_opt": coords[2],
                "esimd_opt": coords[3],
            }
        return None

    def get_higher_optimization_programs(
        self, parent: Program, n: int = 3, dimension: Optional[str] = None
    ) -> List[Program]:
        """
        Get programs with higher optimization levels than the parent.

        This is a KEY method for unlocking performance improvements.
        It finds programs that are more optimized than the current parent
        and returns them as inspirations/exemplars for the LLM.

        Args:
            parent: Current parent program
            n: Maximum number of programs to return
            dimension: If specified, only consider this dimension for comparison.
                       Options: "memory_opt", "compute_opt", "parallelism_opt", "esimd_opt"
                       If None, uses the sum of all optimization levels.

        Returns:
            List of programs at higher optimization levels, sorted by score
        """
        with self._population_lock:
            parent_coords = parent.metadata.get("feature_coords")
            if not parent_coords:
                # Calculate coordinates if not cached
                parent_coords = self._calculate_feature_coords(parent)

            # Handle legacy 3-tuple coords
            if len(parent_coords) == 3:
                parent_coords = tuple(parent_coords) + (0,)
            else:
                parent_coords = tuple(parent_coords)

            dim_to_idx = {
                "memory_opt": 0,
                "compute_opt": 1,
                "parallelism_opt": 2,
                "esimd_opt": 3,
            }

            higher_level_programs = []

            for coord_key, prog_id in self.feature_map.items():
                if prog_id == parent.id or prog_id not in self.programs:
                    continue

                program = self.programs[prog_id]

                # Ensure coord_key is a 4-tuple
                if len(coord_key) == 3:
                    coord_key = coord_key + (0,)

                # Check if this program is at a higher optimization level
                is_higher = False
                if dimension:
                    # Compare specific dimension
                    idx = dim_to_idx.get(dimension, 0)
                    if coord_key[idx] > parent_coords[idx]:
                        is_higher = True
                else:
                    # Compare sum of all optimization levels
                    if sum(coord_key) > sum(parent_coords):
                        is_higher = True

                if is_higher:
                    score = get_main_score(program.metrics or {})
                    opt_sum = sum(coord_key)
                    higher_level_programs.append((score, opt_sum, program))

            # Sort by score (primary) and optimization level (secondary)
            higher_level_programs.sort(key=lambda x: (x[0], x[1]), reverse=True)

            result = [prog for _, _, prog in higher_level_programs[:n]]

            if result:
                logging.info(
                    f"Found {len(result)} programs at higher optimization levels than parent "
                    f"(parent level: {sum(parent_coords)})"
                )
            else:
                logging.debug(f"No programs found at higher optimization levels than parent")

            return result

    def get_best_program_at_level(
        self, memory_opt: int, compute_opt: int, parallelism_opt: int, esimd_opt: int = 0
    ) -> Optional[Program]:
        """
        Get the best program at a specific optimization level.

        Args:
            memory_opt: Target memory optimization level (0-3)
            compute_opt: Target compute optimization level (0-3)
            parallelism_opt: Target parallelism optimization level (0-3)
            esimd_opt: Target ESIMD optimization level (0-3)

        Returns:
            Elite program at that level, or None if cell is empty
        """
        with self._population_lock:
            coord_key = (memory_opt, compute_opt, parallelism_opt, esimd_opt)
            prog_id = self.feature_map.get(coord_key)
            if prog_id and prog_id in self.programs:
                return self.programs[prog_id]
            return None

    def refresh_diversity_reference(self, force: bool = False) -> None:
        """
        Refresh the diversity reference set if population has changed significantly.

        The diversity reference set is used for sampling diversity calculations.
        It should be periodically refreshed as the population evolves to remain
        representative.

        Args:
            force: If True, always refresh. If False, only refresh if population
                   has changed by more than 20% since last refresh.
        """
        with self._population_lock:
            if not self.programs:
                return

            current_size = len(self.programs)
            reference_size = len(self.diversity_reference_codes)

            # Check if refresh is needed
            if not force and reference_size > 0:
                # Only refresh if population changed significantly (>20%)
                change_ratio = abs(current_size - reference_size) / max(reference_size, 1)
                if change_ratio < 0.2:
                    return

            logging.info(f"Refreshing diversity reference set (population: {current_size})")
            self._initialize_diversity_reference_set()

    def validate_coordinates_determinism(self, sample_size: int = 10) -> bool:
        """
        Validate that coordinate classification is deterministic.

        Runs classification twice on a sample of programs and verifies
        identical results. Useful for debugging and testing.

        Args:
            sample_size: Number of programs to validate

        Returns:
            True if all sampled programs produce consistent coordinates
        """
        with self._population_lock:
            sample = list(self.programs.values())[:sample_size]

            for program in sample:
                coords1 = OptimizationFeatureClassifier.classify_from_code(program.code_as_str, program.language)
                coords2 = OptimizationFeatureClassifier.classify_from_code(program.code_as_str, program.language)

                if coords1 != coords2:
                    logging.error(f"Non-deterministic classification for {program.id}: " f"{coords1} != {coords2}")
                    return False

                # Also check against stored coordinates
                stored_coords = program.metadata.get("feature_coords")
                if stored_coords and tuple(stored_coords) != coords1:
                    logging.error(f"Coordinate drift for {program.id}: " f"stored={stored_coords}, computed={coords1}")
                    return False

            logging.info(f"Coordinate determinism validated for {len(sample)} programs")
            return True


# ============================================================================
# MODULE-LEVEL EXPORTS
# ============================================================================

__all__ = [
    "OptimizationAwareDatabase",
    "OptimizationFeatureClassifier",
    "SamplingMethod",
    "ExplorationStrategy",
    "deterministic_code_hash",
    "get_main_score",
    # QD Gradient exports (if available)
    "HAS_QD_GRADIENT",
]
