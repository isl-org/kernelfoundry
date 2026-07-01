"""
Quality-Diversity Gradient Module for MAP-Elites Kernel Evolution

This module implements gradient-based enhancements for MAP-Elites, inspired by:
- CMA-ME (Covariance Matrix Adaptation MAP-Elites) - Fontaine et al., 2020
- PGA-MAP-Elites (Policy Gradient Assisted) - Nilsson & Cully, 2021
- DQD (Differentiable Quality-Diversity) - Fontaine & Nikolaidis, 2022

**Design Philosophy**

In standard MAP-Elites, we only track WHERE solutions are (behavioral coordinates)
and HOW GOOD they are (fitness). We lose information about:
- Which parent→child transitions yield improvements
- Which "directions" in behavior space are promising
- Historical success rates for different mutation strategies

This module captures "gradient-like" signals from evolutionary transitions to:
1. Guide parent selection toward productive regions
2. Inform mutation direction hints for the LLM
3. Identify high-potential transition pathways
4. Enable adaptive exploration based on historical success

**Architecture**

- TransitionRecord: Immutable record of a single evolution step
- TransitionStatistics: Aggregated statistics for a behavioral cell
- GradientEstimator: Computes gradient approximations from transition history
- TransitionTracker: Main class integrating all components

**Thread Safety**

All public methods are thread-safe via internal locking.
The tracker can be safely used from multiple worker processes.

**Memory Efficiency**

- Circular buffer limits memory usage to configurable max_history
- LRU eviction for per-cell statistics
- Compact data structures using __slots__ where applicable
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)
import hashlib
import json
import logging
import math
import os
import time

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TYPE ALIASES
# ============================================================================

# Behavioral coordinates in the 4D optimization space
# (memory_opt, compute_opt, parallelism_opt, esimd_opt)
BehaviorCoords = Tuple[int, int, int, int]

# Transition direction vector (delta for each dimension)
TransitionVector = Tuple[int, int, int, int]

# Gradient vector (continuous values for each dimension)
GradientVector = Tuple[float, float, float, float]


# ============================================================================
# ENUMS
# ============================================================================


class TransitionOutcome(Enum):
    """Classification of transition outcomes."""

    IMPROVEMENT = auto()  # Child has better fitness than parent
    NEUTRAL = auto()  # Child has similar fitness (within epsilon)
    REGRESSION = auto()  # Child has worse fitness than parent
    CELL_DISCOVERY = auto()  # Child discovered a new behavioral cell
    ELITE_REPLACEMENT = auto()  # Child replaced an elite in its cell


class GradientType(Enum):
    """Types of gradient estimates."""

    FITNESS = auto()  # Gradient of fitness values
    IMPROVEMENT_RATE = auto()  # Gradient of improvement success rate
    EXPLORATION = auto()  # Gradient toward unexplored regions
    COMBINED = auto()  # Weighted combination of all gradients


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================


class TransitionRecord(NamedTuple):
    """
    Immutable record of a single parent→child evolutionary transition.

    Uses NamedTuple for memory efficiency and immutability guarantees.
    This is the atomic unit of gradient information storage.
    """

    parent_id: str
    """Unique identifier of the parent program."""
    child_id: str
    """Unique identifier of the child program."""
    parent_coords: BehaviorCoords
    """Behavioral coordinates of parent (4D tuple)."""
    child_coords: BehaviorCoords
    """Behavioral coordinates of child (4D tuple)."""
    parent_fitness: float
    """Fitness score of parent at time of transition."""
    child_fitness: float
    """Fitness score of child after evaluation."""
    fitness_delta: float
    """child_fitness - parent_fitness."""
    outcome: TransitionOutcome
    """Classification of the transition result."""
    timestamp: float
    """Unix timestamp when transition occurred."""
    iteration: int
    """Evolution iteration number."""
    mutation_hint: Optional[str] = None
    """Optional string describing the mutation applied."""

    @property
    def transition_vector(self) -> TransitionVector:
        """Compute the direction of movement in behavior space."""
        return (
            self.child_coords[0] - self.parent_coords[0],
            self.child_coords[1] - self.parent_coords[1],
            self.child_coords[2] - self.parent_coords[2],
            self.child_coords[3] - self.parent_coords[3],
        )

    @property
    def is_improvement(self) -> bool:
        """Check if this transition improved fitness."""
        return self.outcome in (
            TransitionOutcome.IMPROVEMENT,
            TransitionOutcome.CELL_DISCOVERY,
            TransitionOutcome.ELITE_REPLACEMENT,
        )

    @property
    def behavioral_distance(self) -> int:
        """Manhattan distance in behavior space."""
        return sum(abs(d) for d in self.transition_vector)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "parent_coords": list(self.parent_coords),
            "child_coords": list(self.child_coords),
            "parent_fitness": self.parent_fitness,
            "child_fitness": self.child_fitness,
            "fitness_delta": self.fitness_delta,
            "outcome": self.outcome.name,
            "timestamp": self.timestamp,
            "iteration": self.iteration,
            "mutation_hint": self.mutation_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionRecord":
        """Deserialize from dictionary."""
        return cls(
            parent_id=data["parent_id"],
            child_id=data["child_id"],
            parent_coords=tuple(data["parent_coords"]),
            child_coords=tuple(data["child_coords"]),
            parent_fitness=data["parent_fitness"],
            child_fitness=data["child_fitness"],
            fitness_delta=data["fitness_delta"],
            outcome=TransitionOutcome[data["outcome"]],
            timestamp=data["timestamp"],
            iteration=data["iteration"],
            mutation_hint=data.get("mutation_hint"),
        )


@dataclass
class CellStatistics:
    """
    Aggregated statistics for a single behavioral cell.

    Tracks both incoming and outgoing transition patterns to understand
    the "flow" of evolution through this cell.
    """

    coords: BehaviorCoords
    """The behavioral coordinates of this cell."""
    total_arrivals: int = 0
    """Number of children that landed in this cell."""
    total_departures: int = 0
    """Number of parents sampled from this cell."""
    improvements_from: int = 0
    """Count of improvements originating from this cell."""
    improvements_to: int = 0
    """Count of improvements arriving at this cell."""
    sum_fitness_delta_out: float = 0.0
    """Sum of fitness deltas for transitions leaving this cell."""
    sum_fitness_delta_in: float = 0.0
    """Sum of fitness deltas for transitions entering this cell."""
    discovery_count: int = 0
    """How many times this cell was discovered (first filled)."""
    elite_replacements: int = 0
    """How many times the elite was replaced here."""
    last_update: float = field(default_factory=time.time)
    """Timestamp of last update."""

    # Track per-direction statistics for gradient computation
    # Key: transition_vector, Value: (success_count, total_count, sum_delta)
    direction_stats: Dict[TransitionVector, Tuple[int, int, float]] = field(default_factory=dict)

    @property
    def avg_fitness_delta_out(self) -> float:
        """Average fitness change for transitions leaving this cell."""
        if self.total_departures == 0:
            return 0.0
        return self.sum_fitness_delta_out / self.total_departures

    @property
    def avg_fitness_delta_in(self) -> float:
        """Average fitness change for transitions entering this cell."""
        if self.total_arrivals == 0:
            return 0.0
        return self.sum_fitness_delta_in / self.total_arrivals

    @property
    def improvement_rate_out(self) -> float:
        """Fraction of departures that led to improvements."""
        if self.total_departures == 0:
            return 0.5  # Prior: unknown potential
        return self.improvements_from / self.total_departures

    @property
    def improvement_rate_in(self) -> float:
        """Fraction of arrivals that were improvements."""
        if self.total_arrivals == 0:
            return 0.5
        return self.improvements_to / self.total_arrivals

    def update_departure(self, record: TransitionRecord) -> None:
        """Update statistics for a transition leaving this cell."""
        self.total_departures += 1
        self.sum_fitness_delta_out += record.fitness_delta

        if record.is_improvement:
            self.improvements_from += 1

        # Update direction-specific stats
        vec = record.transition_vector
        if vec in self.direction_stats:
            succ, total, sum_delta = self.direction_stats[vec]
            self.direction_stats[vec] = (
                succ + (1 if record.is_improvement else 0),
                total + 1,
                sum_delta + record.fitness_delta,
            )
        else:
            self.direction_stats[vec] = (
                1 if record.is_improvement else 0,
                1,
                record.fitness_delta,
            )

        self.last_update = time.time()

    def update_arrival(self, record: TransitionRecord) -> None:
        """Update statistics for a transition entering this cell."""
        self.total_arrivals += 1
        self.sum_fitness_delta_in += record.fitness_delta

        if record.is_improvement:
            self.improvements_to += 1

        if record.outcome == TransitionOutcome.CELL_DISCOVERY:
            self.discovery_count += 1
        elif record.outcome == TransitionOutcome.ELITE_REPLACEMENT:
            self.elite_replacements += 1

        self.last_update = time.time()

    def get_best_directions(self, top_k: int = 3) -> List[Tuple[TransitionVector, float]]:
        """
        Get the most successful transition directions from this cell.

        Returns list of (direction_vector, success_rate) tuples.
        """
        if not self.direction_stats:
            return []

        direction_rates = []
        for vec, (succ, total, sum_delta) in self.direction_stats.items():
            if total >= 2:  # Require at least 2 samples for reliability
                rate = succ / total
                # Bonus for positive average delta
                avg_delta = sum_delta / total
                combined_score = rate + 0.1 * max(0, avg_delta)
                direction_rates.append((vec, combined_score))

        direction_rates.sort(key=lambda x: x[1], reverse=True)
        return direction_rates[:top_k]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "coords": list(self.coords),
            "total_arrivals": self.total_arrivals,
            "total_departures": self.total_departures,
            "improvements_from": self.improvements_from,
            "improvements_to": self.improvements_to,
            "sum_fitness_delta_out": self.sum_fitness_delta_out,
            "sum_fitness_delta_in": self.sum_fitness_delta_in,
            "discovery_count": self.discovery_count,
            "elite_replacements": self.elite_replacements,
            "last_update": self.last_update,
            "direction_stats": {
                f"{v[0]},{v[1]},{v[2]},{v[3]}": list(stats) for v, stats in self.direction_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellStatistics":
        """Deserialize from dictionary."""
        stats = cls(
            coords=tuple(data["coords"]),
            total_arrivals=data["total_arrivals"],
            total_departures=data["total_departures"],
            improvements_from=data["improvements_from"],
            improvements_to=data["improvements_to"],
            sum_fitness_delta_out=data["sum_fitness_delta_out"],
            sum_fitness_delta_in=data["sum_fitness_delta_in"],
            discovery_count=data["discovery_count"],
            elite_replacements=data["elite_replacements"],
            last_update=data["last_update"],
        )

        # Reconstruct direction_stats
        for vec_str, values in data.get("direction_stats", {}).items():
            vec = tuple(int(x) for x in vec_str.split(","))
            stats.direction_stats[vec] = tuple(values)

        return stats


# ============================================================================
# GRADIENT ESTIMATOR
# ============================================================================


class GradientEstimator:
    """
    Estimates gradients in behavior/fitness space from transition history.

    This implements ideas from QD gradient literature:
    - Natural gradient estimation via finite differences
    - Importance-weighted gradient averaging
    - Multi-objective gradient balancing (fitness + novelty)

    **Mathematical Foundation**

    For each dimension d ∈ {memory, compute, parallelism, esimd}:

    1. FITNESS GRADIENT at cell c:
       ∂F/∂d ≈ (1/N) Σ (child_fitness - parent_fitness) * sign(child_d - parent_d)
       where sum is over transitions from c with movement in dimension d

    2. IMPROVEMENT RATE GRADIENT:
       ∂R/∂d ≈ P(improvement | move in +d direction) - P(improvement | move in -d direction)
       Indicates whether moving in +d or -d is more likely to improve

    3. EXPLORATION GRADIENT:
       Points toward empty/low-quality cells weighted by reachability
       ∂E/∂d ≈ Σ (max_score - cell_score) * (cell_d - current_d) / distance

    The combined gradient is: α*∂F/∂d + β*∂R/∂d + γ*∂E/∂d
    with α, β, γ as tunable hyperparameters.
    """

    # Dimension names for interpretability
    DIMENSIONS = ["memory_opt", "compute_opt", "parallelism_opt", "esimd_opt"]

    def __init__(
        self,
        fitness_weight: float = 0.4,
        improvement_rate_weight: float = 0.4,
        exploration_weight: float = 0.2,
        min_samples_for_gradient: int = 3,
        decay_factor: float = 0.95,  # Exponential decay for older transitions
    ):
        """
        Initialize gradient estimator.

        Args:
            fitness_weight: Weight for fitness gradient component
            improvement_rate_weight: Weight for improvement rate gradient
            exploration_weight: Weight for exploration gradient
            min_samples_for_gradient: Minimum transitions needed for reliable estimate
            decay_factor: How much to down-weight older transitions
        """
        self.fitness_weight = fitness_weight
        self.improvement_rate_weight = improvement_rate_weight
        self.exploration_weight = exploration_weight
        self.min_samples = min_samples_for_gradient
        self.decay_factor = decay_factor

        # Validate weights sum to 1.0
        total = fitness_weight + improvement_rate_weight + exploration_weight
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"Gradient weights sum to {total}, normalizing to 1.0")
            self.fitness_weight /= total
            self.improvement_rate_weight /= total
            self.exploration_weight /= total

    def estimate_fitness_gradient(
        self,
        cell_stats: CellStatistics,
        recent_transitions: List[TransitionRecord],
    ) -> GradientVector:
        """
        Estimate fitness gradient at a cell using recent transitions.

        Returns a vector indicating which direction improves fitness.
        """
        gradient = [0.0, 0.0, 0.0, 0.0]
        weights = [0.0, 0.0, 0.0, 0.0]

        current_time = time.time()

        for record in recent_transitions:
            if record.parent_coords != cell_stats.coords:
                continue

            # Time-based decay
            age = current_time - record.timestamp
            decay = self.decay_factor ** (age / 3600)  # Decay per hour

            vec = record.transition_vector

            for d in range(4):
                if vec[d] != 0:
                    # Contribution: sign(direction) * fitness_delta * decay
                    contribution = math.copysign(1, vec[d]) * record.fitness_delta * decay
                    gradient[d] += contribution
                    weights[d] += decay

        # Normalize by total weight
        for d in range(4):
            if weights[d] > 0:
                gradient[d] /= weights[d]

        return tuple(gradient)

    def estimate_improvement_rate_gradient(
        self,
        cell_stats: CellStatistics,
    ) -> GradientVector:
        """
        Estimate gradient based on improvement success rates per direction.

        Positive gradient means moving in +d direction improves more often.
        """
        gradient = [0.0, 0.0, 0.0, 0.0]

        for vec, (succ, total, _) in cell_stats.direction_stats.items():
            if total < self.min_samples:
                continue

            success_rate = succ / total

            for d in range(4):
                if vec[d] != 0:
                    # If moving in +d succeeded often, gradient is positive
                    # If moving in -d succeeded often, gradient is negative
                    direction_contribution = math.copysign(success_rate, vec[d])
                    gradient[d] += direction_contribution

        # Normalize to [-1, 1] range
        max_abs = max(abs(g) for g in gradient) or 1.0
        if max_abs > 1.0:
            gradient = [g / max_abs for g in gradient]

        return tuple(gradient)

    def estimate_exploration_gradient(
        self,
        current_coords: BehaviorCoords,
        empty_cells: List[BehaviorCoords],
        low_quality_cells: List[Tuple[BehaviorCoords, float]],  # (coords, score)
        max_score: float = 1.0,
    ) -> GradientVector:
        """
        Estimate gradient pointing toward unexplored/underexplored regions.

        This encourages exploration of empty cells and improvement of weak cells.
        """
        gradient = [0.0, 0.0, 0.0, 0.0]

        # Contribution from empty cells
        for target in empty_cells:
            distance = sum(abs(target[d] - current_coords[d]) for d in range(4))
            if distance == 0:
                continue

            # Inverse distance weighting
            weight = 1.0 / distance

            for d in range(4):
                diff = target[d] - current_coords[d]
                if diff != 0:
                    gradient[d] += math.copysign(weight, diff)

        # Contribution from low-quality cells (with potential for improvement)
        for target, score in low_quality_cells:
            distance = sum(abs(target[d] - current_coords[d]) for d in range(4))
            if distance == 0:
                continue

            # Weight by improvement potential and inverse distance
            improvement_potential = max_score - score
            weight = improvement_potential / distance

            for d in range(4):
                diff = target[d] - current_coords[d]
                if diff != 0:
                    gradient[d] += math.copysign(weight, diff)

        # Normalize
        norm = math.sqrt(sum(g**2 for g in gradient))
        if norm > 0:
            gradient = [g / norm for g in gradient]

        return tuple(gradient)

    def estimate_combined_gradient(
        self,
        cell_stats: CellStatistics,
        recent_transitions: List[TransitionRecord],
        empty_cells: List[BehaviorCoords],
        low_quality_cells: List[Tuple[BehaviorCoords, float]],
        max_score: float = 1.0,
    ) -> Tuple[GradientVector, Dict[str, GradientVector]]:
        """
        Compute combined gradient from all components.

        Returns:
            Tuple of (combined_gradient, component_gradients_dict)
        """
        # Compute individual gradients
        fitness_grad = self.estimate_fitness_gradient(cell_stats, recent_transitions)
        improvement_grad = self.estimate_improvement_rate_gradient(cell_stats)
        exploration_grad = self.estimate_exploration_gradient(
            cell_stats.coords, empty_cells, low_quality_cells, max_score
        )

        # Combine with weights
        combined = [0.0, 0.0, 0.0, 0.0]
        for d in range(4):
            combined[d] = (
                self.fitness_weight * fitness_grad[d]
                + self.improvement_rate_weight * improvement_grad[d]
                + self.exploration_weight * exploration_grad[d]
            )

        components = {
            "fitness": fitness_grad,
            "improvement_rate": improvement_grad,
            "exploration": exploration_grad,
        }

        return tuple(combined), components

    def gradient_to_mutation_hints(
        self,
        gradient: GradientVector,
        threshold: float = 0.2,
    ) -> List[str]:
        """
        Convert a gradient vector to human-readable mutation hints for the LLM.

        Args:
            gradient: The gradient vector
            threshold: Minimum absolute value to generate a hint

        Returns:
            List of mutation hint strings
        """
        hints = []

        dimension_hints = {
            0: {  # memory_opt
                "positive": [
                    "Consider adding shared/local memory (SLM) usage",
                    "Try implementing register blocking or tiling",
                    "Add asynchronous memory operations or prefetching",
                ],
                "negative": [
                    "Simplify memory access patterns",
                    "Remove unnecessary memory hierarchy optimizations",
                ],
            },
            1: {  # compute_opt
                "positive": [
                    "Fuse multiple operations into a single kernel pass",
                    "Use FMA (fused multiply-add) operations",
                    "Implement tiled or blocked algorithms",
                ],
                "negative": [
                    "Split complex computations into simpler stages",
                    "Reduce algorithmic complexity",
                ],
            },
            2: {  # parallelism_opt
                "positive": [
                    "Add work-group level synchronization and cooperation",
                    "Use sub-group intrinsics for SIMD parallelism",
                    "Implement hierarchical parallelism patterns",
                ],
                "negative": [
                    "Simplify parallel structure",
                    "Reduce synchronization overhead",
                ],
            },
            3: {  # esimd_opt
                "positive": [
                    "Consider using ESIMD (Explicit SIMD) extensions",
                    "Use LSC (Load/Store Cache) operations",
                    "Apply Intel-specific DPAS/XMX matrix operations",
                ],
                "negative": [
                    "Use standard SYCL instead of ESIMD",
                    "Reduce hardware-specific optimizations for portability",
                ],
            },
        }

        for d in range(4):
            if abs(gradient[d]) >= threshold:
                direction = "positive" if gradient[d] > 0 else "negative"
                dim_hints = dimension_hints[d][direction]
                # Select hint based on gradient magnitude
                idx = min(int(abs(gradient[d]) * len(dim_hints)), len(dim_hints) - 1)
                hints.append(dim_hints[idx])

        return hints


# ============================================================================
# TRANSITION TRACKER (Main Class)
# ============================================================================


class TransitionTracker:
    """
    Main class for tracking evolutionary transitions and computing gradients.

    This is the primary interface for the gradient-enhanced MAP-Elites system.
    It maintains a history of transitions, computes per-cell statistics, and
    provides gradient-based sampling and mutation guidance.

    **Usage**

    1. Create tracker with configuration
    2. Call record_transition() after each parent→child evolution
    3. Use get_gradient() to get mutation direction hints
    4. Use get_sampling_weights() to bias parent selection
    5. Periodically call save_checkpoint() for persistence

    **Integration with MAP-Elites**

    The tracker integrates with OptimizationAwareDatabase via:

    - record_transition(): Called after add() with parent/child info
    - get_sampling_weights(): Called during sample() to weight parent selection
    - get_mutation_hints(): Called during prompt construction
    """

    def __init__(
        self,
        max_history: int = 10000,
        max_cell_cache: int = 256,  # 4×4×4×4 = 256 cells
        gradient_estimator: Optional[GradientEstimator] = None,
        checkpoint_interval: int = 100,  # Save every N transitions
    ):
        """
        Initialize transition tracker.

        Args:
            max_history: Maximum number of transitions to keep in memory
            max_cell_cache: Maximum number of cells to track (should be ≥ grid size)
            gradient_estimator: Custom gradient estimator (uses default if None)
            checkpoint_interval: How often to auto-checkpoint (0 = disabled)
        """
        self._lock = RLock()

        # Transition history (circular buffer via deque)
        self._history: Deque[TransitionRecord] = deque(maxlen=max_history)

        # Per-cell statistics
        self._cell_stats: Dict[BehaviorCoords, CellStatistics] = {}
        self._max_cell_cache = max_cell_cache

        # Gradient estimator
        self._gradient_estimator = gradient_estimator or GradientEstimator()

        # Global statistics
        self._total_transitions: int = 0
        self._total_improvements: int = 0
        self._total_discoveries: int = 0
        self._total_elite_replacements: int = 0

        # Checkpoint management
        self._checkpoint_interval = checkpoint_interval
        self._transitions_since_checkpoint: int = 0
        self._output_dir: Optional[str] = None

        # Cache for expensive computations
        self._gradient_cache: Dict[BehaviorCoords, Tuple[GradientVector, float]] = {}
        self._gradient_cache_ttl: float = 60.0  # Cache valid for 60 seconds

        logger.info(f"TransitionTracker initialized: " f"max_history={max_history}, max_cell_cache={max_cell_cache}")

    def set_output_dir(self, output_dir: str) -> None:
        """Set output directory for checkpoints."""
        self._output_dir = output_dir

    def record_transition(
        self,
        parent_id: str,
        child_id: str,
        parent_coords: BehaviorCoords,
        child_coords: BehaviorCoords,
        parent_fitness: float,
        child_fitness: float,
        is_new_cell: bool = False,
        is_elite_replacement: bool = False,
        iteration: int = 0,
        mutation_hint: Optional[str] = None,
    ) -> TransitionRecord:
        """
        Record a single evolutionary transition.

        This is the main entry point for tracking. Call this after each
        parent→child evolution step.

        Args:
            parent_id: Unique ID of parent program
            child_id: Unique ID of child program
            parent_coords: Parent's behavioral coordinates
            child_coords: Child's behavioral coordinates
            parent_fitness: Parent's fitness at time of evolution
            child_fitness: Child's fitness after evaluation
            is_new_cell: True if child discovered a new cell
            is_elite_replacement: True if child replaced the elite
            iteration: Current evolution iteration number
            mutation_hint: Optional description of mutation applied

        Returns:
            The created TransitionRecord
        """
        with self._lock:
            # Compute fitness delta
            fitness_delta = child_fitness - parent_fitness

            # Determine outcome
            if is_new_cell:
                outcome = TransitionOutcome.CELL_DISCOVERY
            elif is_elite_replacement:
                outcome = TransitionOutcome.ELITE_REPLACEMENT
            elif fitness_delta > 0.01:  # Small epsilon for numerical stability
                outcome = TransitionOutcome.IMPROVEMENT
            elif fitness_delta < -0.01:
                outcome = TransitionOutcome.REGRESSION
            else:
                outcome = TransitionOutcome.NEUTRAL

            # Create record
            record = TransitionRecord(
                parent_id=parent_id,
                child_id=child_id,
                parent_coords=parent_coords,
                child_coords=child_coords,
                parent_fitness=parent_fitness,
                child_fitness=child_fitness,
                fitness_delta=fitness_delta,
                outcome=outcome,
                timestamp=time.time(),
                iteration=iteration,
                mutation_hint=mutation_hint,
            )

            # Add to history
            self._history.append(record)

            # Update cell statistics
            self._update_cell_stats(record)

            # Update global counters
            self._total_transitions += 1
            if record.is_improvement:
                self._total_improvements += 1
            if outcome == TransitionOutcome.CELL_DISCOVERY:
                self._total_discoveries += 1
            if outcome == TransitionOutcome.ELITE_REPLACEMENT:
                self._total_elite_replacements += 1

            # Invalidate gradient cache for affected cells
            self._invalidate_gradient_cache(parent_coords)
            self._invalidate_gradient_cache(child_coords)

            # Auto-checkpoint if configured
            self._transitions_since_checkpoint += 1
            if (
                self._checkpoint_interval > 0
                and self._transitions_since_checkpoint >= self._checkpoint_interval
                and self._output_dir
            ):
                self._auto_checkpoint()

            logger.debug(
                f"Recorded transition: {parent_coords} → {child_coords}, "
                f"delta={fitness_delta:.4f}, outcome={outcome.name}"
            )

            return record

    def _update_cell_stats(self, record: TransitionRecord) -> None:
        """Update statistics for cells involved in a transition."""
        # Get or create stats for parent cell
        if record.parent_coords not in self._cell_stats:
            self._cell_stats[record.parent_coords] = CellStatistics(coords=record.parent_coords)
        self._cell_stats[record.parent_coords].update_departure(record)

        # Get or create stats for child cell
        if record.child_coords not in self._cell_stats:
            self._cell_stats[record.child_coords] = CellStatistics(coords=record.child_coords)
        self._cell_stats[record.child_coords].update_arrival(record)

        # LRU eviction if cache is full
        if len(self._cell_stats) > self._max_cell_cache:
            # Remove least recently updated cell
            oldest = min(self._cell_stats.items(), key=lambda x: x[1].last_update)
            del self._cell_stats[oldest[0]]

    def _invalidate_gradient_cache(self, coords: BehaviorCoords) -> None:
        """Invalidate cached gradient for a cell."""
        if coords in self._gradient_cache:
            del self._gradient_cache[coords]

    def get_gradient(
        self,
        coords: BehaviorCoords,
        empty_cells: Optional[List[BehaviorCoords]] = None,
        low_quality_cells: Optional[List[Tuple[BehaviorCoords, float]]] = None,
        max_score: float = 1.0,
        use_cache: bool = True,
    ) -> Tuple[GradientVector, Dict[str, Any]]:
        """
        Get gradient estimate for a cell.

        Args:
            coords: The behavioral coordinates to compute gradient for
            empty_cells: List of empty cells (for exploration gradient)
            low_quality_cells: List of (coords, score) for low-quality cells
            max_score: Maximum possible score (for normalization)
            use_cache: Whether to use cached gradients

        Returns:
            Tuple of (gradient_vector, metadata_dict)
        """
        with self._lock:
            # Check cache
            if use_cache and coords in self._gradient_cache:
                cached_grad, cache_time = self._gradient_cache[coords]
                if time.time() - cache_time < self._gradient_cache_ttl:
                    return cached_grad, {"source": "cache"}

            # Get cell stats
            cell_stats = self._cell_stats.get(coords)
            if cell_stats is None:
                # No data for this cell - return zero gradient
                zero_grad = (0.0, 0.0, 0.0, 0.0)
                return zero_grad, {"source": "no_data", "confidence": 0.0}

            # Get recent transitions for this cell
            recent = [r for r in self._history if r.parent_coords == coords]

            # Compute gradient
            combined_grad, components = self._gradient_estimator.estimate_combined_gradient(
                cell_stats=cell_stats,
                recent_transitions=recent,
                empty_cells=empty_cells or [],
                low_quality_cells=low_quality_cells or [],
                max_score=max_score,
            )

            # Compute confidence based on sample count
            sample_count = cell_stats.total_departures
            confidence = min(1.0, sample_count / 20)  # Full confidence at 20 samples

            # Cache result
            self._gradient_cache[coords] = (combined_grad, time.time())

            metadata = {
                "source": "computed",
                "confidence": confidence,
                "sample_count": sample_count,
                "components": components,
            }

            return combined_grad, metadata

    def get_mutation_hints(
        self,
        coords: BehaviorCoords,
        threshold: float = 0.2,
        max_hints: int = 3,
    ) -> List[str]:
        """
        Get human-readable mutation hints based on gradient at a cell.

        These hints can be injected into LLM prompts to guide optimization
        direction.

        Args:
            coords: Current behavioral coordinates
            threshold: Minimum gradient magnitude to generate hint
            max_hints: Maximum number of hints to return

        Returns:
            List of mutation hint strings
        """
        with self._lock:
            gradient, _ = self.get_gradient(coords)
            hints = self._gradient_estimator.gradient_to_mutation_hints(gradient, threshold)
            return hints[:max_hints]

    def get_sampling_weights(
        self,
        candidate_coords: List[BehaviorCoords],
        strategy: str = "improvement_rate",
    ) -> Dict[BehaviorCoords, float]:
        """
        Compute sampling weights for parent selection.

        Higher weights indicate cells that are more likely to produce
        improvements when used as parents.

        Args:
            candidate_coords: List of candidate cell coordinates
            strategy: Weighting strategy:
                - "improvement_rate": Weight by historical improvement rate
                - "gradient_magnitude": Weight by gradient magnitude
                - "combined": Combination of both

        Returns:
            Dictionary mapping coordinates to sampling weights
        """
        with self._lock:
            weights = {}

            for coords in candidate_coords:
                stats = self._cell_stats.get(coords)

                if stats is None:
                    # No history - use prior of 0.5
                    weights[coords] = 0.5
                    continue

                if strategy == "improvement_rate":
                    # Weight by historical success rate
                    weights[coords] = stats.improvement_rate_out

                elif strategy == "gradient_magnitude":
                    # Weight by gradient magnitude (explores high-gradient areas)
                    gradient, _ = self.get_gradient(coords)
                    magnitude = math.sqrt(sum(g**2 for g in gradient))
                    weights[coords] = magnitude

                elif strategy == "combined":
                    # Combination: high improvement rate OR high gradient
                    rate = stats.improvement_rate_out
                    gradient, _ = self.get_gradient(coords)
                    magnitude = math.sqrt(sum(g**2 for g in gradient))
                    weights[coords] = 0.5 * rate + 0.5 * min(1.0, magnitude)

                else:
                    raise ValueError(f"Unknown sampling strategy: {strategy}")

            # Normalize weights to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            return weights

    def get_best_transition_directions(
        self,
        coords: BehaviorCoords,
        top_k: int = 3,
    ) -> List[Tuple[TransitionVector, float, int]]:
        """
        Get the most successful transition directions from a cell.

        Args:
            coords: Source cell coordinates
            top_k: Number of directions to return

        Returns:
            List of (direction_vector, success_rate, sample_count) tuples
        """
        with self._lock:
            stats = self._cell_stats.get(coords)
            if stats is None:
                return []

            results = []
            for vec, (succ, total, _) in stats.direction_stats.items():
                if total >= 2:  # Minimum samples
                    rate = succ / total
                    results.append((vec, rate, total))

            results.sort(key=lambda x: (x[1], x[2]), reverse=True)
            return results[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracker statistics."""
        with self._lock:
            # Compute aggregate metrics
            improvement_rate = self._total_improvements / max(1, self._total_transitions)

            # Per-cell stats summary
            active_cells = len(self._cell_stats)
            avg_departures = (
                np.mean([s.total_departures for s in self._cell_stats.values()]) if self._cell_stats else 0.0
            )

            # Direction diversity
            all_directions: Set[TransitionVector] = set()
            for stats in self._cell_stats.values():
                all_directions.update(stats.direction_stats.keys())

            return {
                "total_transitions": self._total_transitions,
                "total_improvements": self._total_improvements,
                "total_discoveries": self._total_discoveries,
                "total_elite_replacements": self._total_elite_replacements,
                "overall_improvement_rate": improvement_rate,
                "active_cells_tracked": active_cells,
                "avg_departures_per_cell": avg_departures,
                "unique_directions_observed": len(all_directions),
                "history_size": len(self._history),
                "history_capacity": self._history.maxlen,
            }

    def get_cell_statistics(
        self,
        coords: BehaviorCoords,
    ) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific cell."""
        with self._lock:
            stats = self._cell_stats.get(coords)
            if stats is None:
                return None

            return {
                "coords": coords,
                "total_arrivals": stats.total_arrivals,
                "total_departures": stats.total_departures,
                "improvement_rate_out": stats.improvement_rate_out,
                "improvement_rate_in": stats.improvement_rate_in,
                "avg_fitness_delta_out": stats.avg_fitness_delta_out,
                "avg_fitness_delta_in": stats.avg_fitness_delta_in,
                "discovery_count": stats.discovery_count,
                "elite_replacements": stats.elite_replacements,
                "best_directions": stats.get_best_directions(3),
            }

    def get_recent_transitions(
        self,
        n: int = 100,
        filter_improvements: bool = False,
    ) -> List[TransitionRecord]:
        """Get recent transitions from history."""
        with self._lock:
            recent = list(self._history)[-n:]

            if filter_improvements:
                recent = [r for r in recent if r.is_improvement]

            return recent

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def save_checkpoint(self, path: str) -> None:
        """Save tracker state to file."""
        with self._lock:
            checkpoint = {
                "version": "1.0",
                "timestamp": time.time(),
                "statistics": {
                    "total_transitions": self._total_transitions,
                    "total_improvements": self._total_improvements,
                    "total_discoveries": self._total_discoveries,
                    "total_elite_replacements": self._total_elite_replacements,
                },
                "history": [r.to_dict() for r in self._history],
                "cell_stats": {f"{c[0]},{c[1]},{c[2]},{c[3]}": s.to_dict() for c, s in self._cell_stats.items()},
            }

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(checkpoint, f, indent=2)

            logger.info(f"Saved transition tracker checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load tracker state from file."""
        with self._lock:
            with open(path, "r") as f:
                checkpoint = json.load(f)

            # Restore statistics
            stats = checkpoint.get("statistics", {})
            self._total_transitions = stats.get("total_transitions", 0)
            self._total_improvements = stats.get("total_improvements", 0)
            self._total_discoveries = stats.get("total_discoveries", 0)
            self._total_elite_replacements = stats.get("total_elite_replacements", 0)

            # Restore history
            self._history.clear()
            for r_dict in checkpoint.get("history", []):
                self._history.append(TransitionRecord.from_dict(r_dict))

            # Restore cell stats
            self._cell_stats.clear()
            for coord_str, s_dict in checkpoint.get("cell_stats", {}).items():
                coords = tuple(int(x) for x in coord_str.split(","))
                self._cell_stats[coords] = CellStatistics.from_dict(s_dict)

            # Clear caches
            self._gradient_cache.clear()

            logger.info(
                f"Loaded transition tracker: "
                f"{self._total_transitions} transitions, "
                f"{len(self._cell_stats)} cells"
            )

    def _auto_checkpoint(self) -> None:
        """Perform automatic checkpoint if output_dir is set."""
        if self._output_dir:
            path = os.path.join(self._output_dir, "transition_tracker_checkpoint.json")
            self.save_checkpoint(path)
            self._transitions_since_checkpoint = 0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_transition_matrix(
    tracker: TransitionTracker,
    dimension: int = 0,
) -> np.ndarray:
    """
    Compute a transition probability matrix for a single dimension.

    Returns a 4×4 matrix where entry [i,j] is P(move to level j | at level i).
    Useful for visualizing transition patterns.

    Args:
        tracker: The transition tracker
        dimension: Which dimension to analyze (0-3)

    Returns:
        4×4 numpy array of transition probabilities
    """
    matrix = np.zeros((4, 4))
    counts = np.zeros(4)

    for record in tracker.get_recent_transitions(n=1000):
        from_level = record.parent_coords[dimension]
        to_level = record.child_coords[dimension]
        matrix[from_level, to_level] += 1
        counts[from_level] += 1

    # Normalize rows to probabilities
    for i in range(4):
        if counts[i] > 0:
            matrix[i] /= counts[i]
        else:
            matrix[i, i] = 1.0  # Self-loop if no data

    return matrix


def compute_improvement_heatmap(
    tracker: TransitionTracker,
    dim1: int = 0,
    dim2: int = 1,
) -> np.ndarray:
    """
    Compute a 4×4 heatmap of improvement rates for two dimensions.

    Entry [i,j] is the improvement rate for transitions from cells
    with dim1=i and dim2=j (marginalizing over other dimensions).

    Args:
        tracker: The transition tracker
        dim1: First dimension index (0-3)
        dim2: Second dimension index (0-3)

    Returns:
        4×4 numpy array of improvement rates
    """
    improvements = np.zeros((4, 4))
    totals = np.zeros((4, 4))

    for record in tracker.get_recent_transitions(n=1000):
        i = record.parent_coords[dim1]
        j = record.parent_coords[dim2]
        totals[i, j] += 1
        if record.is_improvement:
            improvements[i, j] += 1

    # Compute rates (with smoothing to avoid division by zero)
    rates = (improvements + 1) / (totals + 2)  # Laplace smoothing

    return rates


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core classes
    "TransitionTracker",
    "TransitionRecord",
    "CellStatistics",
    "GradientEstimator",
    # Enums
    "TransitionOutcome",
    "GradientType",
    # Type aliases
    "BehaviorCoords",
    "TransitionVector",
    "GradientVector",
    # Utility functions
    "compute_transition_matrix",
    "compute_improvement_heatmap",
]
