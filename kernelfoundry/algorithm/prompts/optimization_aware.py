"""
Optimization-aware prompting that adds direct, actionable guidance for LLMs.

This module provides direct, actionable guidance for LLMs to generate
high-performance GPU kernels. The prompts focus on WHAT TO DO rather
than abstract taxonomy.

Design Philosophy:
- Give specific, actionable instructions
- Lead with concrete code patterns
- Minimize classification overhead
- Focus the LLM on the transformation task

Backend Support:
- SYCL: Intel oneAPI DPC++ with ESIMD extensions
- CUDA: NVIDIA CUDA with Tensor Cores and CUB/CUTLASS
- OpenCL: Khronos OpenCL C with local memory/sub-group optimizations
- Triton: OpenAI Triton (future support)

The backend parameter controls which set of optimization instructions to use.
When backend is None, it defaults to SYCL for backward compatibility.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum, Enum, auto
import random
import re
import json
from pathlib import Path
import logging
from functools import lru_cache

# =============================================================================
# BACKEND DEFINITIONS
# =============================================================================


class Backend(Enum):
    """Supported GPU backends for kernel generation."""

    SYCL = auto()  # Intel oneAPI DPC++ (includes ESIMD)
    CUDA = auto()  # NVIDIA CUDA
    OPENCL = auto()  # Khronos OpenCL
    TRITON = auto()  # OpenAI Triton (future)

    @classmethod
    def from_string(cls, s: Optional[str]) -> "Backend":
        """
        Convert a string to Backend enum.

        Args:
            s: Backend name string (case-insensitive). None defaults to SYCL.

        Returns:
            Backend enum value

        Raises:
            ValueError: If string doesn't match any known backend
        """
        if s is None:
            return cls.SYCL

        s_upper = s.upper().strip()
        mapping = {
            "SYCL": cls.SYCL,
            "DPC++": cls.SYCL,
            "DPCPP": cls.SYCL,
            "ONEAPI": cls.SYCL,
            "CUDA": cls.CUDA,
            "NVIDIA": cls.CUDA,
            "OPENCL": cls.OPENCL,
            "OCL": cls.OPENCL,
            "KHRONOS": cls.OPENCL,
            "TRITON": cls.TRITON,
        }
        if s_upper in mapping:
            return mapping[s_upper]
        raise ValueError(f"Unknown backend: {s}. Supported: SYCL, CUDA, OCL, TRITON")

    @property
    def supports_explicit_simd(self) -> bool:
        """Whether this backend supports explicit SIMD extensions (like ESIMD)."""
        return self == Backend.SYCL

    @property
    def has_tensor_cores(self) -> bool:
        """Whether this backend supports tensor core operations."""
        return self == Backend.CUDA

    @property
    def display_name(self) -> str:
        """Human-readable backend name for prompts."""
        names = {
            Backend.SYCL: "SYCL",
            Backend.CUDA: "CUDA",
            Backend.OPENCL: "OCL",
            Backend.TRITON: "Triton",
        }
        return names.get(self, self.name)


# =============================================================================
# OPTIMIZATION LEVELS (for tracking, not prompting)
# =============================================================================


class MemoryOptLevel(IntEnum):
    NAIVE = 0
    COALESCED = 1
    SLM_CACHED = 2
    MULTI_LEVEL = 3


class ComputeOptLevel(IntEnum):
    MULTI_PASS = 0
    FUSED = 1
    STREAMING = 2
    ADVANCED = 3


class ParallelismOptLevel(IntEnum):
    THREAD_ONLY = 0
    WORKGROUP = 1
    SUBGROUP = 2
    HIERARCHICAL = 3


class EsimdOptLevel(IntEnum):
    DISABLED = 0
    BASIC = 1
    OPTIMIZED = 2
    EXPERT = 3


@dataclass(frozen=True)
class OptimizationProfile:
    """Optimization profile for tracking kernel characteristics."""

    memory_opt: int
    compute_opt: int
    parallelism_opt: int
    esimd_opt: int = 0

    def __post_init__(self):
        for field, val in [
            ("memory_opt", self.memory_opt),
            ("compute_opt", self.compute_opt),
            ("parallelism_opt", self.parallelism_opt),
            ("esimd_opt", self.esimd_opt),
        ]:
            if not 0 <= val <= 3:
                raise ValueError(f"{field} must be in range [0, 3], got {val}")

    def to_dict(self) -> Dict[str, int]:
        return {
            "memory_opt": self.memory_opt,
            "compute_opt": self.compute_opt,
            "parallelism_opt": self.parallelism_opt,
            "esimd_opt": self.esimd_opt,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "OptimizationProfile":
        return cls(
            memory_opt=d.get("memory_opt", 0),
            compute_opt=d.get("compute_opt", 0),
            parallelism_opt=d.get("parallelism_opt", 0),
            esimd_opt=d.get("esimd_opt", 0),
        )

    @property
    def uses_esimd(self) -> bool:
        return self.esimd_opt > 0

    @property
    def is_high_opt(self) -> bool:
        return (self.memory_opt >= 2 and self.parallelism_opt >= 2) or self.esimd_opt >= 2

    @property
    def total_level(self) -> int:
        return self.memory_opt + self.compute_opt + self.parallelism_opt + self.esimd_opt


# =============================================================================
# ACTIONABLE OPTIMIZATION INSTRUCTIONS
# =============================================================================


@lru_cache(maxsize=1)
def _get_prompt_strings() -> Dict[str, Any]:

    strings_path = Path(__file__).with_name("optimization_aware_prompts.json")
    return json.loads(strings_path.read_text(encoding="utf-8"))


def _prompt_text(*keys: str, default: str = "") -> str:
    current: Any = _get_prompt_strings()
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current if isinstance(current, str) else default


def _get_memory_optimization_instructions(level: int) -> str:
    """Get specific instructions for memory optimization at given level."""
    key = str(level)
    return _prompt_text("memory", "sycl", key, default=_prompt_text("memory", "sycl", "3"))


def _get_cuda_memory_optimization_instructions(level: int) -> str:
    """Get CUDA-specific instructions for memory optimization at given level."""
    key = str(level)
    return _prompt_text("memory", "cuda", key, default=_prompt_text("memory", "cuda", "3"))


def _get_opencl_memory_optimization_instructions(level: int) -> str:
    """Get OpenCL-specific instructions for memory optimization at given level."""
    key = str(level)
    return _prompt_text("memory", "opencl", key, default=_prompt_text("memory", "opencl", "3"))


def _get_triton_memory_optimization_instructions(level: int) -> str:
    """Get Triton-specific instructions for memory optimization at given level."""
    key = str(level)
    return _prompt_text("memory", "triton", key, default=_prompt_text("memory", "triton", "3"))


def _get_compute_optimization_instructions(level: int) -> str:
    """Get specific instructions for compute/algorithmic optimization."""
    key = str(level)
    return _prompt_text("compute", "sycl", key, default=_prompt_text("compute", "sycl", "3"))


def _get_cuda_compute_optimization_instructions(level: int) -> str:
    """Get CUDA-specific instructions for compute/algorithmic optimization."""
    key = str(level)
    return _prompt_text("compute", "cuda", key, default=_prompt_text("compute", "cuda", "3"))


def _get_opencl_compute_optimization_instructions(level: int) -> str:
    """Get OpenCL-specific instructions for compute/algorithmic optimization."""
    key = str(level)
    return _prompt_text("compute", "opencl", key, default=_prompt_text("compute", "opencl", "3"))


def _get_triton_compute_optimization_instructions(level: int) -> str:
    """Get Triton-specific instructions for compute/algorithmic optimization."""
    key = str(level)
    return _prompt_text("compute", "triton", key, default=_prompt_text("compute", "triton", "3"))


def _get_parallelism_optimization_instructions(level: int) -> str:
    """Get specific instructions for parallelism optimization."""
    key = str(level)
    return _prompt_text("parallelism", "sycl", key, default=_prompt_text("parallelism", "sycl", "3"))


def _get_cuda_parallelism_optimization_instructions(level: int) -> str:
    """Get CUDA-specific instructions for parallelism optimization."""
    key = str(level)
    return _prompt_text("parallelism", "cuda", key, default=_prompt_text("parallelism", "cuda", "3"))


def _get_opencl_parallelism_optimization_instructions(level: int) -> str:
    """Get OpenCL-specific instructions for parallelism optimization."""
    key = str(level)
    return _prompt_text("parallelism", "opencl", key, default=_prompt_text("parallelism", "opencl", "3"))


def _get_triton_parallelism_optimization_instructions(level: int) -> str:
    """Get Triton-specific instructions for parallelism optimization."""
    key = str(level)
    return _prompt_text("parallelism", "triton", key, default=_prompt_text("parallelism", "triton", "3"))


def _get_esimd_instructions() -> str:
    """Get ESIMD-specific instructions."""
    return _prompt_text("explicit_simd", "sycl_esimd")


def _get_tensor_core_instructions() -> str:
    """Get NVIDIA Tensor Core instructions (CUDA equivalent of ESIMD for matrix operations)."""
    return _prompt_text("explicit_simd", "cuda_tensor_cores")


def _get_opencl_explicit_simd_instructions() -> str:
    """Get OpenCL SIMD/sub-group instructions."""
    return _prompt_text("explicit_simd", "opencl")


def get_memory_optimization_instructions(level: int, backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get memory optimization instructions for the specified backend.

    Args:
        level: Optimization level (0-3)
        backend: Target backend (SYCL, CUDA, or string). Defaults to SYCL.

    Returns:
        Formatted instruction string for the specified backend and level
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_cuda_memory_optimization_instructions(level)
    if backend == Backend.OPENCL:
        return _get_opencl_memory_optimization_instructions(level)
    if backend == Backend.TRITON:
        return _get_triton_memory_optimization_instructions(level)
    if backend == Backend.SYCL:
        return _get_memory_optimization_instructions(level)
    raise ValueError(f"Unsupported backend '{backend}' for memory optimization instructions")


def get_compute_optimization_instructions(level: int, backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get compute optimization instructions for the specified backend.

    Args:
        level: Optimization level (0-3)
        backend: Target backend (SYCL, CUDA, or string). Defaults to SYCL.

    Returns:
        Formatted instruction string for the specified backend and level
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_cuda_compute_optimization_instructions(level)
    if backend == Backend.OPENCL:
        return _get_opencl_compute_optimization_instructions(level)
    if backend == Backend.TRITON:
        return _get_triton_compute_optimization_instructions(level)
    if backend == Backend.SYCL:
        return _get_compute_optimization_instructions(level)
    raise ValueError(f"Unsupported backend '{backend}' for compute optimization instructions")


def get_parallelism_optimization_instructions(level: int, backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get parallelism optimization instructions for the specified backend.

    Args:
        level: Optimization level (0-3)
        backend: Target backend (SYCL, CUDA, or string). Defaults to SYCL.

    Returns:
        Formatted instruction string for the specified backend and level
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_cuda_parallelism_optimization_instructions(level)
    if backend == Backend.OPENCL:
        return _get_opencl_parallelism_optimization_instructions(level)
    if backend == Backend.TRITON:
        return _get_triton_parallelism_optimization_instructions(level)
    if backend == Backend.SYCL:
        return _get_parallelism_optimization_instructions(level)
    raise ValueError(f"Unsupported backend '{backend}' for parallelism optimization instructions")


def get_explicit_simd_instructions(backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get explicit SIMD/Tensor Core instructions for the specified backend.

    Args:
        backend: Target backend (SYCL, CUDA, or string). Defaults to SYCL.

    Returns:
        Formatted instruction string:
        - SYCL: Intel ESIMD instructions
        - CUDA: NVIDIA Tensor Core instructions
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_tensor_core_instructions()
    if backend == Backend.OPENCL:
        return _get_opencl_explicit_simd_instructions()
    if backend == Backend.TRITON:
        return ""  # Triton has no explicit SIMD; use tl.dot for tensor core dispatch
    if backend == Backend.SYCL:
        return _get_esimd_instructions()
    raise ValueError(f"Unsupported backend '{backend}' for explicit SIMD instructions")


# =============================================================================
# PROMPT BUILDERS
# =============================================================================


def get_optimization_taxonomy_prompt(
    include_antipatterns: bool = False,
    include_performance_hints: bool = True,
    dimensions: Optional[List[str]] = None,
    include_esimd: bool = True,
    backend: Optional[Union[Backend, str]] = None,
) -> str:
    """
    Get optimization taxonomy/instructions for the requested dimensions.

    Args:
        include_antipatterns: Whether to include common antipatterns section
        include_performance_hints: Whether to include performance hints section
        dimensions: List of dimensions to include (memory, compute, parallelism, esimd/tensor_core)
        include_esimd: Whether to include ESIMD/Tensor Core section (for backward compatibility)
        backend: Target backend (SYCL, CUDA). Defaults to SYCL.

    Returns:
        Formatted optimization taxonomy prompt for the specified backend
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    parts = []

    # Add dimension-specific instructions
    valid_dimensions = dimensions or ["memory", "compute", "parallelism", "esimd"]

    if "memory" in valid_dimensions:
        parts.append(get_memory_optimization_instructions(2, backend))  # SLM/SMEM level as reference

    if "compute" in valid_dimensions:
        parts.append(get_compute_optimization_instructions(2, backend))  # Online algorithms as reference

    if "parallelism" in valid_dimensions:
        parts.append(get_parallelism_optimization_instructions(2, backend))  # Sub-group/warp as reference

    # Handle explicit SIMD / tensor core instructions
    if ("esimd" in valid_dimensions or "tensor_core" in valid_dimensions) and include_esimd:
        parts.append(get_explicit_simd_instructions(backend))

    if include_antipatterns:
        parts.append(_get_antipatterns_section(backend))

    if include_performance_hints:
        parts.append(_get_performance_hints_section(backend))

    return "\n".join(parts)


def _get_antipatterns_section(backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get common antipatterns to avoid in GPU kernel development.

    Args:
        backend: Target backend (SYCL, CUDA). Defaults to SYCL.

    Returns:
        Backend-specific antipatterns section
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_cuda_antipatterns_section()
    if backend == Backend.OPENCL:
        return _get_opencl_antipatterns_section()
    if backend == Backend.TRITON:
        return _get_triton_antipatterns_section()
    if backend == Backend.SYCL:
        return _get_sycl_antipatterns_section()
    raise ValueError(f"Unsupported backend '{backend}' for antipatterns section")


def _get_sycl_antipatterns_section() -> str:
    """Get SYCL-specific antipatterns."""
    return _prompt_text("antipatterns", "sycl")


def _get_cuda_antipatterns_section() -> str:
    """Get CUDA-specific antipatterns."""
    return _prompt_text("antipatterns", "cuda")


def _get_opencl_antipatterns_section() -> str:
    """Get OpenCL-specific antipatterns."""
    return _prompt_text("antipatterns", "opencl")


def _get_triton_antipatterns_section() -> str:
    """Get Triton-specific antipatterns."""
    return _prompt_text("antipatterns", "triton")


def _get_performance_hints_section(backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get performance optimization hints and best practices.

    Args:
        backend: Target backend (SYCL, CUDA). Defaults to SYCL.

    Returns:
        Backend-specific performance hints section
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_cuda_performance_hints_section()
    if backend == Backend.OPENCL:
        return _get_opencl_performance_hints_section()
    if backend == Backend.TRITON:
        return _get_triton_performance_hints_section()
    if backend == Backend.SYCL:
        return _get_sycl_performance_hints_section()
    raise ValueError(f"Unsupported backend '{backend}' for performance hints section")


def _get_sycl_performance_hints_section() -> str:
    """Get SYCL-specific performance hints."""
    return _prompt_text("performance_hints", "sycl")


def _get_cuda_performance_hints_section() -> str:
    """Get CUDA-specific performance hints."""
    return _prompt_text("performance_hints", "cuda")


def _get_opencl_performance_hints_section() -> str:
    """Get OpenCL-specific performance hints."""
    return _prompt_text("performance_hints", "opencl")


def _get_triton_performance_hints_section() -> str:
    """Get Triton-specific performance hints."""
    return _prompt_text("performance_hints", "triton")


def build_exploration_prompt(
    base_prompt: str,
    underexplored_regions: List[Dict[str, int]],
    include_taxonomy: bool = True,
    exploration_temperature: float = 0.7,
    random_seed: Optional[int] = None,
    include_esimd: bool = True,
    backend: Optional[Union[Backend, str]] = None,
) -> str:
    """
    Build a prompt that explores underexplored optimization strategies.

    Generates concise, actionable optimization directives that integrate
    cleanly with the main prompt structure.

    Args:
        base_prompt: The base prompt to augment
        underexplored_regions: List of optimization profiles to explore
        include_taxonomy: Whether to include full taxonomy (unused, kept for compatibility)
        exploration_temperature: Temperature for sampling (unused, kept for compatibility)
        random_seed: Random seed for reproducibility
        include_esimd: Whether to include ESIMD/Tensor Core techniques
        backend: Target backend (SYCL, CUDA). Defaults to SYCL.

    Returns:
        Augmented prompt with exploration directives
    """
    # Kept for API compatibility; currently not used in generation logic.
    _ = include_taxonomy
    _ = exploration_temperature

    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if not underexplored_regions:
        return base_prompt

    rng = random.Random(random_seed) if random_seed is not None else random
    target = rng.choice(underexplored_regions)

    # Build specific technique recommendations based on target profile
    techniques = _collect_techniques_for_profile(target, include_esimd, backend)

    if not techniques:
        return base_prompt

    # Format as a clean, integrated section
    exploration_section = _format_optimization_section(
        techniques,
        header="Required Optimizations",
        intro="Apply the following optimization techniques in your implementation:",
    )

    return base_prompt + "\n\n" + exploration_section


def _collect_techniques_for_profile(
    profile: Dict[str, int],
    include_esimd: bool = True,
    backend: Optional[Union[Backend, str]] = None,
) -> List[str]:
    """
    Collect specific optimization techniques based on a target profile.

    Args:
        profile: Dictionary with memory_opt, compute_opt, parallelism_opt, esimd_opt levels
        include_esimd: Whether to include ESIMD/Tensor Core techniques
        backend: Target backend (SYCL, CUDA). Defaults to SYCL.

    Returns:
        List of technique description strings
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    techniques = []
    is_cuda = backend == Backend.CUDA
    is_triton = backend == Backend.TRITON

    memory_level = profile.get("memory_opt", 0)
    if memory_level == 1:
        if is_triton:
            techniques.append(
                "**Blocked Coalesced Loads**: Use `tl.load(ptr + offsets, mask=offsets < N, other=0.0)` "
                "with `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`. "
                "BLOCK_SIZE must be a power of 2 declared as `tl.constexpr`."
            )
        elif is_cuda:
            techniques.append(
                "**Vectorized Memory Access**: Replace scalar loads with CUDA vectors (`float4` for general use, "
                "`float2` for tight register pressure). Use `__ldg()` for read-only data. "
                "Ensure adjacent threads access adjacent memory addresses for coalesced access."
            )
        else:
            techniques.append(
                "**Vectorized Memory Access**: Replace scalar loads with SYCL vectors (`float4` for general use, "
                "`float8`/`float16` for bulk transfers, `float2` for tight register pressure). "
                "Ensure adjacent work-items access adjacent memory addresses for coalesced access."
            )
    elif memory_level == 2:
        if is_triton:
            techniques.append(
                "**Cache Hints & 2D Tiling**: Use `eviction_policy='evict_last'` for reused weights and "
                "`eviction_policy='evict_first'` for streaming activations. For matrix ops, use 2D offset "
                "arrays: `a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak`."
            )
        elif is_cuda:
            techniques.append(
                "**Shared Memory (SMEM) Tiling**: Cache frequently accessed data in `__shared__` memory. "
                "Synchronize with `__syncthreads()` after writes and before reads. "
                "Use 16×16 or 32×32 tiles for float data. Add padding to avoid bank conflicts."
            )
        else:
            techniques.append(
                "**Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using "
                "`group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes "
                "and before reads. Use 16×16 or 32×32 tiles for float data."
            )
    elif memory_level == 3:
        if is_triton:
            techniques.append(
                "**Block Pointer API + Software Pipelining**: Use `tl.make_block_ptr` + `tl.advance` for "
                "clean 2D tile iteration. Set `num_stages=3` (or higher) in the kernel launch to enable "
                "automatic memory latency hiding via software pipelining."
            )
        elif is_cuda:
            techniques.append(
                "**Register Blocking**: Each thread computes a THREAD_M×THREAD_N output block in "
                "private register arrays. Use `#pragma unroll` on inner loops. Combine with SMEM tiling "
                "and consider async copy for double buffering."
            )
        else:
            techniques.append(
                "**Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in "
                "private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling "
                "for multi-level memory hierarchy optimization."
            )

    compute_level = profile.get("compute_opt", 0)
    if compute_level == 1:
        if is_triton:
            techniques.append(
                "**Fused Operations with tl.dot**: Fuse element-wise operations in a single kernel to eliminate "
                "intermediate global memory writes. Use `tl.dot(a, b)` for matrix multiply — it automatically "
                "uses Tensor Cores for float16 inputs. Accumulate in float32 for numerical stability."
            )
        elif is_cuda:
            techniques.append(
                "**Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into "
                "a single kernel. Use `fmaf()` for fused multiply-add. Eliminate intermediate buffers."
            )
        else:
            techniques.append(
                "**Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into "
                "a single kernel. Eliminate intermediate buffers by computing in registers."
            )
    elif compute_level == 2:
        if is_triton:
            techniques.append(
                "**Block-Level Reductions**: Use `tl.max(x, axis=0)` and `tl.sum(x, axis=0)` for block-level "
                "reductions. Implement numerically stable softmax in one pass: subtract max before exp. "
                "For N > BLOCK_N, use online rescaling with running max/sum accumulators."
            )
        elif is_cuda:
            techniques.append(
                "**Online Algorithms**: Use single-pass algorithms with running statistics. For softmax: "
                "maintain running_max and running_sum with warp shuffle reductions. For variance: use "
                "Welford's online algorithm."
            )
        else:
            techniques.append(
                "**Online Algorithms**: Use single-pass algorithms with running statistics. For softmax: "
                "maintain running_max and running_sum, rescaling sum when max changes. For variance: use "
                "Welford's online algorithm."
            )
    elif compute_level == 3:
        if is_triton:
            techniques.append(
                "**Flash-Attention Style Blocked Algorithm**: Process large sequences in tiles: outer loop over "
                "Q blocks, inner loop over K/V blocks. Maintain (m_i, l_i, acc) registers and rescale on each "
                "inner iteration. Use `tl.dot` for QK^T and P*V — requires BLOCK_K >= 16."
            )
        elif is_cuda:
            techniques.append(
                "**Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Use Tensor Cores "
                "for matrix operations (WMMA API). Consider Flash-Attention style with rescaling."
            )
        else:
            techniques.append(
                "**Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade "
                "recomputation for memory savings (e.g., Flash-Attention style). Maintain running "
                "accumulators across blocks with proper rescaling."
            )

    parallelism_level = profile.get("parallelism_opt", 0)
    if parallelism_level == 1:
        if is_triton:
            techniques.append(
                "**1D Blocked Programs**: Use `pid = tl.program_id(0)` and `offsets = pid * BLOCK_SIZE + "
                "tl.arange(0, BLOCK_SIZE)`. Set grid with `(triton.cdiv(N, BLOCK_SIZE),)`. "
                "BLOCK_SIZE must be a `tl.constexpr` power of 2."
            )
        elif is_cuda:
            techniques.append(
                "**Block-Level Reductions**: Replace atomic operations with O(log N) tree-based "
                "reductions in shared memory. Synchronize with `__syncthreads()` between iterations."
            )
        else:
            techniques.append(
                "**Work-Group Reductions**: Replace atomic operations with O(log N) tree-based "
                "reductions in local memory. Synchronize with `group_barrier()` between iterations."
            )
    elif parallelism_level == 2:
        if is_triton:
            techniques.append(
                "**2D Program Grid with num_warps/num_stages**: Use a 2D grid for matrix kernels: "
                "`pid_m = tl.program_id(0), pid_n = tl.program_id(1)`. Set `num_warps=4` (or 8 for "
                "compute-bound) and `num_stages=3` for memory latency hiding. Apply L2 swizzling "
                "(GROUP_SIZE_M pattern) to improve cache reuse."
            )
        elif is_cuda:
            techniques.append(
                "**Warp Shuffle Operations**: Use `__shfl_xor_sync()` for hardware-accelerated "
                "warp reductions. Use `__shfl_sync()` for broadcasts. Always use `_sync` versions with mask."
            )
        else:
            techniques.append(
                "**Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated "
                "SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing."
            )
    elif parallelism_level == 3:
        if is_triton:
            techniques.append(
                "**Auto-Tuning with @triton.autotune**: Add multiple `triton.Config` entries covering "
                "different BLOCK_M/N/K sizes and num_warps/num_stages values. Specify `key=['M', 'N', 'K']` "
                "so the tuner retunes when problem dimensions change. Or use persistent kernels with "
                "`tl.num_programs(0)` for maximum SM utilization."
            )
        elif is_cuda:
            techniques.append(
                "**Hierarchical Parallelism**: Structure work at three levels - blocks (tile assignment), "
                "warps (sub-tile processing), threads (register tile). Use warp shuffles to "
                "share data within warp without shared memory."
            )
        else:
            techniques.append(
                "**Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), "
                "sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to "
                "share data within sub-group without SLM."
            )

    esimd_level = profile.get("esimd_opt", 0) if include_esimd else 0
    if esimd_level > 0 and not is_triton:
        if is_cuda:
            techniques.append(
                "**NVIDIA Tensor Cores**: Use WMMA API with `fragment<>` types, `load_matrix_sync`, "
                "`mma_sync`, and `store_matrix_sync`. Use half precision inputs for best performance. "
                "Stage data through shared memory, not global."
            )
        else:
            techniques.append(
                "**Intel ESIMD**: Use explicit SIMD with `simd<T, N>` vector types, `block_load/store` "
                "for memory, and `[[intel::sycl_explicit_simd]]` kernel attribute. Replace all SYCL "
                "work-group primitives with ESIMD equivalents (e.g., `barrier()` not `group_barrier`)."
            )

    return techniques


def _format_optimization_section(
    techniques: List[str],
    header: str = "Optimization Directives",
    intro: str = "Implement the following optimizations:",
) -> str:
    """Format a list of techniques into a clean markdown section."""
    if not techniques:
        return ""

    lines = [f"## {header}", "", intro, ""]
    for technique in techniques:
        lines.append(f"- {technique}")

    return "\n".join(lines)


def get_optimization_guidance_for_parent(
    parent_profile: Dict[str, int],
    strategy: str = "mutate",
    parent_performance: Optional[Dict[str, float]] = None,
    target_dimension: Optional[str] = None,
    random_seed: Optional[int] = None,
    include_esimd: bool = True,
    backend: Optional[Union[Backend, str]] = None,
) -> Dict[str, int]:
    """
    Compute the target optimization coordinates to explore next.

    Args:
        parent_profile: Current optimization profile of the parent kernel
        strategy: Evolution strategy (mutate, intensify, diversify, specialize, balance, esimd_upgrade)
        parent_performance: Performance metrics of parent kernel (unused, kept for compatibility)
        target_dimension: Specific dimension to target for mutation
        random_seed: Random seed for reproducibility
        include_esimd: Whether to include ESIMD/Tensor Core upgrades
        backend: Target backend (SYCL, CUDA). Defaults to SYCL.

    Returns:
        Target optimization profile coordinates
    """
    _ = parent_performance
    _ = backend

    rng = random.Random(random_seed) if random_seed is not None else random
    target = {
        "memory_opt": min(3, max(0, int(parent_profile.get("memory_opt", 0)))),
        "compute_opt": min(3, max(0, int(parent_profile.get("compute_opt", 0)))),
        "parallelism_opt": min(3, max(0, int(parent_profile.get("parallelism_opt", 0)))),
        "esimd_opt": min(3, max(0, int(parent_profile.get("esimd_opt", 0)))) if include_esimd else 0,
    }

    if strategy == "esimd_upgrade":
        if include_esimd:
            target["esimd_opt"] = max(1, min(3, target["esimd_opt"] + 1))
        return target

    if strategy == "intensify":
        for key in ["memory_opt", "compute_opt", "parallelism_opt"]:
            target[key] = min(3, target[key] + 1)
        return target

    if strategy == "specialize":
        weakest = min(["memory_opt", "compute_opt", "parallelism_opt"], key=lambda k: target[k])
        target[weakest] = min(3, target[weakest] + 1)
        return target

    if strategy == "balance":
        target_level = max(target["memory_opt"], target["compute_opt"], target["parallelism_opt"])
        for key in ["memory_opt", "compute_opt", "parallelism_opt"]:
            if target[key] < target_level:
                target[key] = min(3, target[key] + 1)
        return target

    if strategy == "mutate":
        candidates = [k for k in ["memory_opt", "compute_opt", "parallelism_opt"] if target[k] < 3]
        if include_esimd and target["esimd_opt"] == 0:
            candidates.append("esimd_opt")
        if not candidates:
            return target
        dim_map = {
            "memory": "memory_opt",
            "compute": "compute_opt",
            "parallelism": "parallelism_opt",
            "esimd": "esimd_opt",
        }
        forced = dim_map.get(target_dimension) if target_dimension else None
        selected = forced if forced in candidates else rng.choice(candidates)
        target[selected] = min(3, target[selected] + 1)
        return target

    return target


def _get_mutation_guidance(
    memory: int,
    compute: int,
    parallelism: int,
    esimd: int,
    include_esimd: bool,
    target_dimension: Optional[str],
    rng: Any,
    backend: Optional[Union[Backend, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Get guidance for mutation strategy."""
    candidates = []
    if memory < 3:
        candidates.append(("memory", memory + 1))
    if compute < 3:
        candidates.append(("compute", compute + 1))
    if parallelism < 3:
        candidates.append(("parallelism", parallelism + 1))
    if include_esimd and esimd == 0:
        candidates.append(("esimd", 1))

    if not candidates:
        return (["Fine-tune tile sizes and loop unrolling factors for the target hardware"], [])

    if target_dimension:
        selected = next((c for c in candidates if c[0] == target_dimension), candidates[0])
    else:
        selected = rng.choice(candidates)

    dim, level = selected
    return _get_dimension_guidance(dim, level, backend)


def _get_intensify_guidance(
    memory: int,
    compute: int,
    parallelism: int,
    backend: Optional[Union[Backend, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Get guidance for intensify strategy."""
    techniques: List[str] = []
    code_examples: List[str] = []

    for dim, current in [("memory", memory), ("compute", compute), ("parallelism", parallelism)]:
        if current < 3:
            t, c = _get_dimension_guidance(dim, current + 1, backend)
            techniques.extend(t)
            code_examples.extend(c)

    return techniques, code_examples


def _get_specialize_guidance(
    memory: int,
    compute: int,
    parallelism: int,
    backend: Optional[Union[Backend, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Get guidance for specialize strategy - focus on weakest dimension."""
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    dims = {"memory": memory, "compute": compute, "parallelism": parallelism}
    weakest = min(dims.keys(), key=lambda k: dims[k])
    current = dims[weakest]

    if current >= 3:
        return (["All dimensions are at maximum level - focus on micro-optimizations"], [])

    techniques, code_examples = _get_dimension_guidance(weakest, current + 1, backend)

    # Add emphasis (backend-aware terminology)
    if backend == Backend.CUDA:
        focus_hints = {
            "memory": "Focus on memory optimization - this is your biggest opportunity for improvement.",
            "compute": "Focus on algorithmic optimization - reducing passes will have the highest impact.",
            "parallelism": "Focus on parallelism - better warp utilization will significantly improve performance.",
        }
    else:
        focus_hints = {
            "memory": "Focus on memory optimization - this is your biggest opportunity for improvement.",
            "compute": "Focus on algorithmic optimization - reducing passes will have the highest impact.",
            "parallelism": "Focus on parallelism - better sub-group utilization will significantly improve performance.",
        }
    techniques.insert(0, focus_hints.get(weakest, ""))

    return techniques, code_examples


def _get_balance_guidance(
    memory: int,
    compute: int,
    parallelism: int,
    backend: Optional[Union[Backend, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Get guidance for balance strategy - bring all dimensions to similar levels."""
    target_level = max(memory, compute, parallelism)
    techniques: List[str] = []
    code_examples: List[str] = []

    for dim, current in [("memory", memory), ("compute", compute), ("parallelism", parallelism)]:
        if current < target_level:
            t, c = _get_dimension_guidance(dim, min(current + 1, 3), backend)
            techniques.extend(t)
            code_examples.extend(c)

    return techniques, code_examples


def _get_dimension_guidance(
    dimension: str,
    level: int,
    backend: Optional[Union[Backend, str]] = None,
) -> Tuple[List[str], List[str]]:
    """Get techniques and code examples for a specific dimension and level."""
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_cuda_dimension_guidance(dimension, level)
    if backend == Backend.OPENCL:
        return _get_ocl_dimension_guidance(dimension, level)
    if backend == Backend.SYCL:
        return _get_sycl_dimension_guidance(dimension, level)
    raise ValueError(f"Unsupported backend '{backend}' for dimension guidance")


def _load_dimension_guidance_json(backend_key: str, dimension: str, level: int) -> Tuple[List[str], List[str]]:
    """Load dimension guidance techniques and code from the prompts JSON file."""
    entry = (
        _get_prompt_strings().get("dimension_guidance", {}).get(backend_key, {}).get(dimension, {}).get(str(level), {})
    )
    return entry.get("techniques", []), entry.get("code", [])


def _get_sycl_dimension_guidance(dimension: str, level: int) -> Tuple[List[str], List[str]]:
    """Get SYCL-specific techniques and code examples for a dimension and level."""
    return _load_dimension_guidance_json("sycl", dimension, level)


def _get_ocl_dimension_guidance(dimension: str, level: int) -> Tuple[List[str], List[str]]:
    """Get OpenCL-specific techniques and code examples for a dimension and level."""
    return _load_dimension_guidance_json("opencl", dimension, level)


def _get_cuda_dimension_guidance(dimension: str, level: int) -> Tuple[List[str], List[str]]:
    """Get CUDA-specific techniques and code examples for a dimension and level."""
    # CUDA has tensor_cores instead of ESIMD
    cuda_dim = "tensor_cores" if dimension == "esimd" else dimension
    return _load_dimension_guidance_json("cuda", cuda_dim, level)


def _get_explicit_simd_upgrade_section(backend: Optional[Union[Backend, str]] = None) -> str:
    """
    Get explicit SIMD/Tensor Core upgrade guidance based on backend.

    For SYCL, returns Intel ESIMD guidance.
    For CUDA, returns NVIDIA Tensor Core (WMMA) guidance.

    Args:
        backend: Target backend. Defaults to SYCL.

    Returns:
        Formatted guidance string for explicit SIMD/accelerator features.
    """
    if isinstance(backend, str):
        backend = Backend.from_string(backend)
    elif backend is None:
        backend = Backend.SYCL

    if backend == Backend.CUDA:
        return _get_tensor_core_upgrade_section()
    if backend == Backend.OPENCL:
        return _get_opencl_upgrade_section()
    if backend == Backend.SYCL:
        return _get_esimd_upgrade_section()
    raise ValueError(f"Unsupported backend '{backend}' for explicit SIMD upgrade section")


def _get_tensor_core_upgrade_section() -> str:
    """Get complete Tensor Core upgrade guidance section for CUDA."""
    return _prompt_text("upgrades", "cuda_tensor_cores")


def _get_esimd_upgrade_section() -> str:
    """Get complete ESIMD upgrade guidance section."""
    return _prompt_text("upgrades", "sycl_esimd")


def _get_opencl_upgrade_section() -> str:
    """Get complete OpenCL upgrade guidance section."""
    return _prompt_text("upgrades", "opencl")


def _format_guidance_section(
    techniques: List[str],
    code_examples: List[str],
    strategy: str,
) -> str:
    """Format guidance into a clean markdown section."""
    strategy_headers = {
        "mutate": "Optimization Focus",
        "intensify": "Intensification Targets",
        "diversify": "Alternative Approaches",
        "specialize": "Specialization Focus",
        "balance": "Balancing Optimizations",
    }

    header = strategy_headers.get(strategy, "Optimization Guidance")
    lines = [f"\n## {header}\n"]

    # Add techniques as bullet points (not numbered - avoids conflicts with main prompt)
    for technique in techniques:
        if technique:  # Skip empty strings
            lines.append(f"- {technique}")

    # Add code examples if present
    if code_examples:
        lines.append("")
        for example in code_examples:
            lines.append(example)

    return "\n".join(lines)


# =============================================================================
# RESPONSE PARSING
# =============================================================================


def parse_llm_optimization_response(
    llm_output: str,
    strict: bool = False,
) -> Optional[Dict[str, int]]:
    """
    Parse optimization profile from LLM response.

    Looks for JSON with optimization levels.
    """
    required_keys = ["memory_opt", "compute_opt", "parallelism_opt"]

    patterns = [
        r"```json\s*(\{[\s\S]*?\})\s*```",
        r'(\{[^{}]*"memory_opt"[^{}]*\})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, llm_output, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
                data = json.loads(cleaned)

                profile = data.get("optimization_profile", data)

                result = {}
                for key in required_keys:
                    if key in profile:
                        val = profile[key]
                        if isinstance(val, str):
                            val = int(val.split("-")[0].strip())
                        result[key] = max(0, min(3, int(val)))
                    elif strict:
                        break
                    else:
                        result[key] = 0

                if len(result) == len(required_keys):
                    if "esimd_opt" in profile:
                        result["esimd_opt"] = max(0, min(3, int(profile["esimd_opt"])))
                    return result

            except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                continue

    return None


def parse_llm_full_response(llm_output: str) -> Dict:
    """Parse complete structured response from LLM."""
    result = {
        "optimization_profile": None,
        "techniques_used": [],
        "rationale": "",
        "potential_improvements": "",
        "raw_json": None,
        "parse_success": False,
    }

    profile = parse_llm_optimization_response(llm_output)
    if profile:
        result["optimization_profile"] = profile
        result["parse_success"] = True

    return result


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Backend enum
    "Backend",
    # Optimization profiles and levels
    "OptimizationProfile",
    "MemoryOptLevel",
    "ComputeOptLevel",
    "ParallelismOptLevel",
    "EsimdOptLevel",
    # Main prompt builders
    "build_exploration_prompt",
    "get_optimization_taxonomy_prompt",
    "get_optimization_guidance_for_parent",
    # Backend-aware instruction getters
    "get_memory_optimization_instructions",
    "get_compute_optimization_instructions",
    "get_parallelism_optimization_instructions",
    "get_explicit_simd_instructions",
    # Response parsers
    "parse_llm_optimization_response",
    "parse_llm_full_response",
]
