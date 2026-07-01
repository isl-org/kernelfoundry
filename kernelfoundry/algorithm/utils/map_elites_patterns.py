# ========================================================================
# PATTERN DEFINITIONS
# ========================================================================
#
# Pattern structure: Each level contains categorized patterns with weights.
# Weight indicates how strong an indicator the pattern is (1.0 = definitive).
# Categories prevent double-counting related patterns.


MEMORY_OPT_PATTERNS = {
    # Level 0: No patterns - absence of optimization
    0: {},
    # Level 1: Vectorized/coalesced global memory access
    1: {
        "vector_types": {
            "weight": 1.0,
            "patterns": [
                r"sycl::vec<[^>]+,\s*[2348]>",
                r"sycl::(?:float|double|int|uint|short|ushort|char|uchar|long|ulong|half)[2348]\b",
                r"\bfloat[2348]\b",
                r"\bint[2348]\b",
                r"\bdouble[2348]\b",
                r"\bhalf[2348]\b",
            ],
        },
        "vector_load_store": {
            "weight": 1.0,
            "patterns": [
                r"\.load\s*\(",
                r"\.store\s*\(",
                r"reinterpret_cast<[^>]*(?:float|int|double)[2348]\s*\*>",
                r"as_(?:float|int|uint|double)[2348]\s*\(",
            ],
        },
        "aligned_access": {
            "weight": 0.8,
            "patterns": [
                r"alignas\s*\(\s*\d+\s*\)",
                r"\[\[intel::aligned(?:\s*\(\s*\d+\s*\))?\]\]",
                r"aligned_alloc_(?:device|shared|host)",
                r"__attribute__\s*\(\s*\(\s*aligned",
            ],
        },
        "global_ptr": {
            "weight": 0.6,
            "patterns": [
                r"global_ptr<",
                r"multi_ptr<[^>]*global_space",
                r"address_space::global",
            ],
        },
        "esimd_global_memory_ops": {
            "weight": 0.9,
            "patterns": [
                r"\bblock_load\s*<",
                r"\bblock_store\s*<",
                r"\blsc_block_load\s*<",
                r"\blsc_block_store\s*<",
                r"\bgather\s*<",
                r"\bscatter\s*<",
                r"\blsc_gather\s*<",
                r"\blsc_scatter\s*<",
            ],
        },
    },
    # Level 2: Shared/Local memory (SLM) usage
    2: {
        "local_memory_accessor": {
            "weight": 1.0,
            "patterns": [
                r"local_accessor<",
                r"accessor<[^>]*target::local",
                r"group_local_memory<",
                r"group_local_memory_for_overwrite<",
                r"ext::oneapi::group_local_memory",
                r"__local\s+\w+",  # OpenCL-style
            ],
        },
        "local_ptr": {
            "weight": 0.9,
            "patterns": [
                r"local_ptr<",
                r"multi_ptr<[^>]*local_space",
                r"address_space::local",
            ],
        },
        "slm_allocation_pattern": {
            "weight": 0.8,
            "patterns": [
                # Common SLM tile declarations
                r"(?:float|double|int|half)\s*(?:tile|smem|slm|shared|local)_?\w*\s*\[",
                r"\[\s*TILE_?\w*\s*\]\s*\[\s*TILE_?\w*\s*\]",
                r"\[\s*BLOCK_?\w*\s*\]\s*\[\s*BLOCK_?\w*\s*\]",
            ],
        },
        "esimd_slm_memory_ops": {
            "weight": 1.0,
            "patterns": [
                r"\bslm_block_load\s*<",
                r"\bslm_block_store\s*<",
                r"\bslm_gather\s*<",
                r"\bslm_scatter\s*<",
                r"\bslm_init\s*<",
                r"\bslm_allocator\s*<",
            ],
        },
        "barrier_for_slm": {
            "weight": 0.7,
            "patterns": [
                r"group_barrier\s*\(",
                r"sycl::group_barrier",
                r"item\.barrier\s*\(",
                r"work_group_barrier",
            ],
        },
        "bank_conflict_avoidance": {
            "weight": 0.6,
            "patterns": [
                # Padding for bank conflict avoidance
                r"\[\s*\w+\s*\+\s*[1-4]\s*\]",  # [SIZE + 1] padding
                r"\[\s*\w+\s*\]\s*\[\s*\w+\s*\+\s*[1-4]\s*\]",
            ],
        },
    },
    # Level 3: Multi-level hierarchy (registers + SLM + prefetch/async)
    3: {
        "register_blocking": {
            "weight": 1.0,
            "patterns": [
                # Register tile arrays
                r"(?:float|double|half)\s+reg_?\w*\s*\[\s*(?:THREAD_)?(?:TILE_)?[A-Z_]*\s*\]",
                r"(?:float|double|half)\s+reg_?\w*\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]",
                r"(?:float|double|half)\s+(?:frag|fragment|acc)_?\w*\s*\[",
                # Explicit register hints
                r"\[\[intel::private_copies\]\]",
                r"__private\s+\w+\s*\[",
            ],
        },
        "async_copy": {
            "weight": 1.0,
            "patterns": [
                r"async_work_group_copy\s*\(",
                r"group::async_work_group_copy",
                r"group\.async_work_group_copy",
                r"sycl::ext::oneapi::experimental::async",
            ],
        },
        "prefetch": {
            "weight": 0.9,
            "patterns": [
                r"\.prefetch\s*\(",
                r"handler\.prefetch",
                r"prefetch<",
                r"__builtin_prefetch",
            ],
        },
        "double_buffering": {
            "weight": 0.9,
            "patterns": [
                r"(?:buffer|buf|tile|smem)_?[aAbB]?\s*\[\s*2\s*\]",
                r"(?:ping|pong)_?(?:buffer|buf)",
                r"(?:buffer|tile)_?(?:idx|index)\s*\^\s*1",
                r"(?:curr|next|prev)_?(?:buffer|tile|buf)",
            ],
        },
        "unroll_directives": {
            "weight": 0.7,
            "patterns": [
                r"#\s*pragma\s+unroll",
                r"\[\[intel::loop_coalesce",
                r"\[\[intel::ivdep\]\]",
                r"\[\[intel::initiation_interval",
                r"\[\[intel::max_concurrency",
            ],
        },
        "memory_scope_control": {
            "weight": 0.6,
            "patterns": [
                r"memory_order::",
                r"memory_scope::",
                r"sycl::memory_order",
                r"sycl::memory_scope",
            ],
        },
        "esimd_advanced_memory_pipeline": {
            "weight": 0.8,
            "patterns": [
                r"\blsc_prefetch\s*<",
                r"\blsc_fence\s*<",
                r"\blsc_atomic_update\s*<",
                r"\braw_send\s*<",
                r"\braw_sends\s*<",
                r"cache_hint::",
                r"alignment<\s*\d+\s*>",
                r"cache_hint_L1<",
                r"cache_hint_L2<",
            ],
        },
    },
}

COMPUTE_OPT_PATTERNS = {
    # Level 0: No patterns - multi-pass naive
    0: {},
    # Level 1: Operation fusion
    1: {
        "fma_operations": {
            "weight": 1.0,
            "patterns": [
                r"sycl::fma\s*\(",
                r"sycl::mad\s*\(",
                r"__fmaf?\s*\(",
                r"fma\s*\([^)]+,[^)]+,[^)]+\)",
            ],
        },
        "native_math": {
            "weight": 0.8,
            "patterns": [
                r"sycl::native::",
                r"sycl::half_precision::",
                r"__(?:exp|log|sin|cos|sqrt)f\s*\(",
            ],
        },
        "fused_accumulation": {
            "weight": 0.6,
            "patterns": [
                # Accumulation patterns (more specific than before)
                r"\w+\s*\+=\s*\w+\s*\*\s*\w+",  # acc += a * b
                r"\w+\s*=\s*sycl::fma",
            ],
        },
        "inline_fusion_comments": {
            "weight": 0.5,  # Lower weight for comments
            "patterns": [
                r"//\s*[Ff]used?\b",
                r"//\s*[Cc]ombined?\b",
                r"/\*\s*[Ff]usion\b",
            ],
        },
    },
    # Level 2: Single-pass / streaming algorithms
    2: {
        "group_reduce": {
            "weight": 1.0,
            "patterns": [
                r"reduce_over_group\s*\(",
                r"sycl::reduce_over_group",
                r"joint_reduce\s*\(",
                r"sycl::joint_reduce",
            ],
        },
        "atomic_reduce": {
            "weight": 0.9,
            "patterns": [
                r"atomic_ref<",
                r"sycl::atomic_ref",
                r"atomic_fetch_(?:add|max|min)\s*\(",
            ],
        },
        "online_algorithm_indicators": {
            "weight": 0.8,
            "patterns": [
                # Variable names suggesting online algorithms
                r"running_(?:max|min|sum|mean|var)",
                r"(?:online|streaming)_(?:max|min|sum|softmax)",
                r"(?:local|partial)_(?:max|min|sum)\s*=",
            ],
        },
        "single_pass_comments": {
            "weight": 0.5,
            "patterns": [
                r"//\s*[Ss]ingle[- ]?pass",
                r"//\s*[Ss]treaming",
                r"//\s*[Oo]nline\s+(?:softmax|algorithm|reduction)",
                r"//\s*[Ii]n[- ]?place",
            ],
        },
        "welford_pattern": {
            "weight": 1.0,
            "patterns": [
                # Welford's online algorithm for variance
                r"delta\s*=.*-\s*mean",
                r"mean\s*\+=.*delta.*(?:count|n)",
                r"M2\s*\+=",
            ],
        },
    },
    # Level 3: Advanced algorithmic transformations (tiled, blocked, flash-style)
    3: {
        "tiled_algorithm": {
            "weight": 1.0,
            "patterns": [
                # Explicit tiling loops
                r"for\s*\([^)]*(?:TILE|BLOCK|tile|block)_?\w*\s*[+<]",
                r"(?:t_?|k_?)(?:tile|block)\s*[<+=]",
                r"(?:tile|block)_?(?:idx|id|start)",
            ],
        },
        "nd_range_2d_3d": {
            "weight": 0.9,
            "patterns": [
                r"nd_range<\s*[23]\s*>",
                r"sycl::nd_range<\s*[23]\s*>",
            ],
        },
        "joint_matrix": {
            "weight": 1.0,
            "patterns": [
                r"joint_matrix<",
                r"joint_matrix_load\s*\(",
                r"joint_matrix_store\s*\(",
                r"joint_matrix_mad\s*\(",
                r"sycl::ext::oneapi::experimental::matrix",
            ],
        },
        "scan_algorithms": {
            "weight": 1.0,
            "patterns": [
                r"inclusive_scan_over_group\s*\(",
                r"exclusive_scan_over_group\s*\(",
                r"joint_inclusive_scan\s*\(",
                r"joint_exclusive_scan\s*\(",
            ],
        },
        "recomputation_tradeoff": {
            "weight": 0.7,
            "patterns": [
                # Flash-attention style recomputation
                r"//\s*[Rr]ecompute",
                r"//\s*[Ff]lash",
                r"//\s*[Bb]lock[- ]?wise",
                r"scale\s*=.*exp\s*\(.*-.*max",
            ],
        },
        "tile_size_constants": {
            "weight": 0.6,
            "patterns": [
                r"(?:constexpr|const)\s+int\s+(?:TILE|BLOCK|BM|BN|BK|TM|TN)_?\w*\s*=\s*\d+",
                r"#\s*define\s+(?:TILE|BLOCK|BM|BN|BK)_?\w*\s+\d+",
            ],
        },
    },
}

PARALLELISM_OPT_PATTERNS = {
    # Level 0: No patterns - thread-level only
    0: {},
    # Level 1: Work-group synchronization
    1: {
        "nd_item_usage": {
            "weight": 1.0,
            "patterns": [
                r"nd_item<",
                r"sycl::nd_item",
                r"item\.get_local_id\s*\(",
                r"item\.get_group\s*\(",
            ],
        },
        "work_group_barrier": {
            "weight": 1.0,
            "patterns": [
                r"group_barrier\s*\(",
                r"sycl::group_barrier",
                r"item\.barrier\s*\(",
                r"work_group_barrier",
            ],
        },
        "work_group_id": {
            "weight": 0.8,
            "patterns": [
                r"get_group_id\s*\(",
                r"get_local_id\s*\(",
                r"get_group_range\s*\(",
                r"get_local_range\s*\(",
            ],
        },
        "group_broadcast": {
            "weight": 0.9,
            "patterns": [
                r"group_broadcast\s*\(",
                r"sycl::group_broadcast",
            ],
        },
    },
    # Level 2: Sub-group (SIMD) intrinsics
    2: {
        "sub_group_object": {
            "weight": 1.0,
            "patterns": [
                r"get_sub_group\s*\(",
                r"item\.get_sub_group\s*\(",
                r"sycl::sub_group\b",
                r"\bsub_group\s+sg\b",
            ],
        },
        "sub_group_collectives": {
            "weight": 1.0,
            "patterns": [
                r"reduce_over_group\s*\(\s*sg",
                r"sycl::reduce_over_group\s*\(\s*(?:sub_group|sg)",
                r"group_broadcast\s*\(\s*sg",
                r"any_of_group\s*\(\s*sg",
                r"all_of_group\s*\(\s*sg",
            ],
        },
        "sub_group_shuffle": {
            "weight": 1.0,
            "patterns": [
                r"shift_group_left\s*\(",
                r"shift_group_right\s*\(",
                r"permute_group_by_xor\s*\(",
                r"select_from_group\s*\(",
                r"shuffle\s*\(\s*sg",
                r"shuffle_xor\s*\(",
                r"shuffle_down\s*\(",
                r"shuffle_up\s*\(",
            ],
        },
        "sub_group_load_store": {
            "weight": 0.9,
            "patterns": [
                r"group_load\s*\(\s*sg",
                r"group_store\s*\(\s*sg",
                r"sub_group.*\.load\s*\(",
                r"sub_group.*\.store\s*\(",
            ],
        },
        "reqd_sub_group_size": {
            "weight": 0.8,
            "patterns": [
                r"\[\[intel::reqd_sub_group_size\s*\(\s*\d+\s*\)\]\]",
                r"\[\[sycl::reqd_sub_group_size\s*\(\s*\d+\s*\)\]\]",
                r"reqd_sub_group_size",
            ],
        },
    },
    # Level 3: Hierarchical / multi-level parallelism
    3: {
        "hierarchical_parallelism": {
            "weight": 1.0,
            "patterns": [
                r"parallel_for_work_group\s*\(",
                r"parallel_for_work_item\s*\(",
                r"h_item<",
                r"sycl::h_item",
            ],
        },
        "multi_level_coordination": {
            "weight": 1.0,
            "patterns": [
                # Sub-group AND work-group coordination together
                r"sub_group.*group_barrier",
                r"group_barrier.*sub_group",
                # Nested parallelism patterns
                r"sg\.get_group_id.*get_local_id",
            ],
        },
        "warp_level_tiling": {
            "weight": 0.9,
            "patterns": [
                # Warp/sub-group tiling patterns
                r"(?:WARP|SG|SUBGROUP)_(?:M|N|K|SIZE|TILE)",
                r"(?:warp|sg)_(?:row|col|id|idx)",
                r"lane_id\s*=",
            ],
        },
        "work_group_size_attr": {
            "weight": 0.7,
            "patterns": [
                r"\[\[sycl::reqd_work_group_size\s*\(",
                r"\[\[intel::max_work_group_size\s*\(",
                r"\[\[intel::num_simd_work_items\s*\(",
            ],
        },
        "task_graph": {
            "weight": 0.8,
            "patterns": [
                r"depends_on\s*\(",
                r"handler\.depends_on",
                r"host_task\s*\(",
            ],
        },
        "specialization_constants": {
            "weight": 0.6,
            "patterns": [
                r"specialization_id<",
                r"set_specialization_constant\s*\(",
                r"get_specialization_constant\s*\(",
            ],
        },
    },
}

# ========================================================================
# CUDA PATTERN DEFINITIONS (for portability)
# ========================================================================
#
# CUDA patterns mirror the SYCL structure for comprehensive classification.
# These patterns detect CUDA-specific optimizations at each level:
#
# Memory Optimization Levels:
#   0 = Naive global memory access (no patterns - absence of optimization)
#   1 = Coalesced/vectorized access, texture/LDG loads
#   2 = Shared memory tiling with synchronization
#   3 = Register blocking, async copy, double buffering, tensor cores
#
# Compute Optimization Levels:
#   0 = Multi-pass naive algorithms (no patterns)
#   1 = Fused operations, FMA, fast math
#   2 = Single-pass/streaming, warp-level reductions
#   3 = Tiled algorithms, tensor core operations, flash-style patterns
#
# Parallelism Optimization Levels:
#   0 = Thread-only (no patterns)
#   1 = Block-level synchronization and shared memory
#   2 = Warp-level intrinsics and shuffle operations
#   3 = Cooperative groups, multi-level coordination, grid-level sync

CUDA_MEMORY_OPT_PATTERNS = {
    # Level 0: No patterns - absence of optimization
    0: {},
    # Level 1: Vectorized/coalesced global memory access
    1: {
        "vector_types": {
            "weight": 1.0,
            "patterns": [
                r"\b(?:float|int|double|half|__half)[2348]\b",
                r"\b(?:uint|short|ushort|char|uchar|long|ulong)[2348]\b",
                r"make_(?:float|int|double|uint|short|char|long|half)[2348]\s*\(",
                r"__half2\b",
                r"__nv_bfloat16[2]?\b",
            ],
        },
        "vectorized_load_store": {
            "weight": 1.0,
            "patterns": [
                r"reinterpret_cast<\s*(?:float|int|double|uint)[2348]\s*\*\s*>",
                r"\*\s*reinterpret_cast<\s*(?:const\s+)?(?:float|int|double)[2348]\s*\*",
                r"(?:float|int|double|uint)[2348]\s+\w+\s*=\s*\*",
                r"\*\s*\(\s*(?:float|int|double)[2348]\s*\*\s*\)",
            ],
        },
        "texture_ldg_access": {
            "weight": 1.0,
            "patterns": [
                r"__ldg\s*\(",
                r"__ldcg\s*\(",
                r"__ldca\s*\(",
                r"__ldcs\s*\(",
                r"__ldlu\s*\(",
                r"__ldcv\s*\(",
                r"tex1Dfetch\s*\(",
                r"tex2D\s*<",
                r"tex3D\s*<",
                r"surf2Dread\s*\(",
                r"surf2Dwrite\s*\(",
            ],
        },
        "aligned_access": {
            "weight": 0.8,
            "patterns": [
                r"__align__\s*\(\s*\d+\s*\)",
                r"__attribute__\s*\(\s*\(\s*aligned\s*\(\s*\d+\s*\)\s*\)\s*\)",
                r"alignas\s*\(\s*\d+\s*\)",
                r"cudaMallocPitch\s*\(",
                r"cudaMemcpy2D\s*\(",
            ],
        },
        "memory_hints": {
            "weight": 0.6,
            "patterns": [
                r"__restrict__",
                r"__builtin_assume_aligned\s*\(",
                r"const\s+__restrict__",
            ],
        },
    },
    # Level 2: Shared memory (SMEM) usage and tiling
    2: {
        "shared_memory_declaration": {
            "weight": 1.0,
            "patterns": [
                r"__shared__\s+\w+",
                r"extern\s+__shared__\s+\w+",
                r"__shared__\s+(?:float|double|int|half|__half)\s*\w*\s*\[",
                r"extern\s+__shared__\s+(?:float|double|int|half)\s+\w+\s*\[\s*\]",
            ],
        },
        "shared_memory_tiling": {
            "weight": 0.9,
            "patterns": [
                # Common SMEM tile declarations
                r"(?:float|double|int|half)\s*(?:tile|smem|shared|shmem|s_|As|Bs)_?\w*\s*\[",
                r"\[\s*(?:TILE|BLOCK|BM|BN|BK)_?\w*\s*\]\s*\[\s*(?:TILE|BLOCK|BM|BN|BK)_?\w*\s*\]",
                r"__shared__.*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]",
            ],
        },
        "block_synchronization": {
            "weight": 1.0,
            "patterns": [
                r"__syncthreads\s*\(",
                r"__syncthreads_count\s*\(",
                r"__syncthreads_and\s*\(",
                r"__syncthreads_or\s*\(",
            ],
        },
        "bank_conflict_avoidance": {
            "weight": 0.7,
            "patterns": [
                # Padding for bank conflict avoidance
                r"\[\s*\w+\s*\+\s*[1-4]\s*\]",
                r"\[\s*\w+\s*\]\s*\[\s*\w+\s*\+\s*[1-4]\s*\]",
                r"//.*bank\s*conflict",
                r"/\*.*bank\s*conflict",
                r"SMEM_STRIDE",
            ],
        },
        "dynamic_shared_memory": {
            "weight": 0.8,
            "patterns": [
                r"extern\s+__shared__\s+\w+\s+\w+\s*\[\s*\]",
                r"<<<[^>]*,\s*\d+\s*,\s*\d+\s*>>>",  # Third parameter is dynamic smem size
                r"cudaFuncSetAttribute.*SharedMemCarveout",
            ],
        },
    },
    # Level 3: Multi-level hierarchy (registers + SMEM + async copy)
    3: {
        "register_blocking": {
            "weight": 1.0,
            "patterns": [
                # Register tile arrays for data reuse
                r"(?:float|double|half|__half)\s+(?:reg|r|frag|fragment|acc|a|b|c)_?\w*\s*\[\s*\d+\s*\]",
                r"(?:float|double|half)\s+(?:reg|r|frag)_?\w*\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]",
                r"#\s*pragma\s+unroll\s+\d*\s*\n\s*for",
                r"register\s+(?:float|double|int)\s+\w+",
            ],
        },
        "async_copy_pipeline": {
            "weight": 1.0,
            "patterns": [
                r"__pipeline_memcpy_async\s*\(",
                r"__pipeline_commit\s*\(",
                r"__pipeline_wait_prior\s*\(",
                r"cuda::memcpy_async\s*\(",
                r"cuda::pipeline\s*<",
                r"cuda::pipeline_shared_state\s*<",
                r"cooperative_groups::memcpy_async\s*\(",
                r"nvcuda::experimental::pipeline",
            ],
        },
        "double_buffering": {
            "weight": 0.9,
            "patterns": [
                # Software pipelining / double buffering
                r"(?:buffer|buf|tile|smem|shared)_?[AB01]\s*\[",
                r"(?:ping|pong)_?(?:buffer|buf|smem)",
                r"(?:stage|buffer)_?(?:idx|index)\s*\^\s*1",
                r"(?:curr|next|prev)_?(?:buffer|tile|smem)",
                r"\[\s*\d+\s*%\s*2\s*\]",  # [i % 2] style double buffering
            ],
        },
        "tensor_core_mma": {
            "weight": 1.0,
            "patterns": [
                r"nvcuda::wmma::",
                r"wmma::load_matrix_sync\s*\(",
                r"wmma::store_matrix_sync\s*\(",
                r"wmma::mma_sync\s*\(",
                r"wmma::fragment\s*<",
                r"mma::sync\s*\(",
                r"__mma_sync\s*\(",
            ],
        },
        "prefetch_hints": {
            "weight": 0.7,
            "patterns": [
                r"__builtin_prefetch\s*\(",
                r"asm.*prefetch",
                r"//.*prefetch",
            ],
        },
        "l2_cache_control": {
            "weight": 0.8,
            "patterns": [
                r"cudaAccessPolicyWindow",
                r"cudaStreamSetAttribute.*AccessPolicyWindow",
                r"cudaCtxResetPersistingL2Cache",
                r"cudaFuncSetAttribute.*PreferredSharedMemoryCarveout",
            ],
        },
        "unroll_directives": {
            "weight": 0.6,
            "patterns": [
                r"#\s*pragma\s+unroll\b",
                r"#\s*pragma\s+unroll\s+\d+",
                r"\[\[unroll\]\]",
                r"\[\[unroll\(\d+\)\]\]",
            ],
        },
    },
}

CUDA_COMPUTE_OPT_PATTERNS = {
    # Level 0: No patterns - multi-pass naive algorithms
    0: {},
    # Level 1: Operation fusion and fast math
    1: {
        "fma_operations": {
            "weight": 1.0,
            "patterns": [
                r"__fmaf?\s*\(",
                r"__fmaf_r[nudz]\s*\(",
                r"fmaf?\s*\([^)]+,[^)]+,[^)]+\)",
                r"__dmaf?\s*\(",
                r"__hfma[2]?\s*\(",
            ],
        },
        "fast_math_intrinsics": {
            "weight": 1.0,
            "patterns": [
                r"__f(?:div|mul|add|sub)_r[nudz]\s*\(",
                r"__expf?\s*\(",
                r"__logf?\s*\(",
                r"__sinf?\s*\(",
                r"__cosf?\s*\(",
                r"__tanf?\s*\(",
                r"__powf?\s*\(",
                r"__sqrtf?\s*\(",
                r"__rsqrtf?\s*\(",
                r"rsqrtf?\s*\(",
                r"__fast_sincosf?\s*\(",
                r"__saturatef?\s*\(",
            ],
        },
        "half_precision_ops": {
            "weight": 0.9,
            "patterns": [
                r"__hadd[2]?\s*\(",
                r"__hsub[2]?\s*\(",
                r"__hmul[2]?\s*\(",
                r"__hfma[2]?\s*\(",
                r"__hdiv[2]?\s*\(",
                r"__h2exp[2]?\s*\(",
                r"__h2log[2]?\s*\(",
                r"__h2sqrt[2]?\s*\(",
                r"__hmax[2]?\s*\(",
                r"__hmin[2]?\s*\(",
            ],
        },
        "fused_accumulation": {
            "weight": 0.6,
            "patterns": [
                # Accumulation patterns with FMA potential
                r"\w+\s*\+=\s*\w+\s*\*\s*\w+",
                r"\w+\s*=\s*__fmaf?\s*\(",
                r"\w+\s*=\s*fmaf?\s*\(",
            ],
        },
        "fast_math_flags": {
            "weight": 0.5,
            "patterns": [
                r"--use_fast_math",
                r"-ffast-math",
                r"__CUDA_ARCH__",
                r"//\s*fast\s*math",
            ],
        },
    },
    # Level 2: Single-pass / streaming algorithms with warp reductions
    2: {
        "warp_reduce": {
            "weight": 1.0,
            "patterns": [
                r"__reduce_add_sync\s*\(",
                r"__reduce_min_sync\s*\(",
                r"__reduce_max_sync\s*\(",
                r"__reduce_and_sync\s*\(",
                r"__reduce_or_sync\s*\(",
                r"__reduce_xor_sync\s*\(",
                r"cub::WarpReduce\s*<",
                r"cub::BlockReduce\s*<",
            ],
        },
        "shuffle_reduce_pattern": {
            "weight": 1.0,
            "patterns": [
                # Butterfly reduction pattern with shuffles
                r"__shfl_xor_sync\s*\([^,]+,\s*\w+,\s*(?:16|8|4|2|1)\s*[,)]",
                r"__shfl_down_sync\s*\([^,]+,\s*\w+,\s*(?:16|8|4|2|1)\s*[,)]",
                r"for\s*\([^)]*;\s*\w+\s*>=\s*1\s*;\s*\w+\s*(?:>>=|/=)\s*2",
            ],
        },
        "atomic_accumulation": {
            "weight": 0.9,
            "patterns": [
                r"atomicAdd\s*\(",
                r"atomicMax\s*\(",
                r"atomicMin\s*\(",
                r"atomicCAS\s*\(",
                r"atomicExch\s*\(",
                r"atomicOr\s*\(",
                r"atomicAnd\s*\(",
                r"atomicXor\s*\(",
                r"atomicAdd_block\s*\(",
                r"atomicAdd_system\s*\(",
            ],
        },
        "online_algorithm_indicators": {
            "weight": 0.8,
            "patterns": [
                r"running_(?:max|min|sum|mean|var)",
                r"(?:online|streaming)_(?:max|min|sum|softmax)",
                r"(?:local|partial|thread)_(?:max|min|sum)\s*=",
                r"//\s*[Oo]nline\s+(?:softmax|mean|variance)",
            ],
        },
        "single_pass_comments": {
            "weight": 0.5,
            "patterns": [
                r"//\s*[Ss]ingle[- ]?pass",
                r"//\s*[Ss]treaming",
                r"//\s*[Ff]used\s+kernel",
                r"//\s*[Oo]ne[- ]?pass",
            ],
        },
        "welford_pattern": {
            "weight": 1.0,
            "patterns": [
                # Welford's online algorithm for mean/variance
                r"delta\s*=.*-\s*mean",
                r"mean\s*\+=.*delta.*(?:/|\\*.*inv).*(?:count|n|i)",
                r"M2\s*\+=",
                r"//\s*[Ww]elford",
            ],
        },
    },
    # Level 3: Advanced algorithmic transformations (tiled, blocked, flash-style)
    3: {
        "tiled_algorithm": {
            "weight": 1.0,
            "patterns": [
                # Explicit tiling loops
                r"for\s*\([^)]*(?:TILE|BLOCK|BM|BN|BK|tile|block)_?\w*\s*[+<]",
                r"for\s*\([^)]*(?:t_?|k_?|i_?)(?:tile|block)\s*[<+=]",
                r"(?:tile|block)_?(?:start|offset|idx|id)\s*=",
                r"(?:num|n)_?(?:tiles|blocks)\s*=",
            ],
        },
        "tensor_core_compute": {
            "weight": 1.0,
            "patterns": [
                r"wmma::mma_sync\s*\(",
                r"nvcuda::wmma::mma_sync",
                r"mma::sync\s*\(",
                r"fragment<\s*(?:matrix_a|matrix_b|accumulator)",
                r"wmma::fill_fragment\s*\(",
                r"__mma_(?:bf16|tf32|m16n16k16|m32n8k16)\s*\(",
            ],
        },
        "cutlass_patterns": {
            "weight": 1.0,
            "patterns": [
                # CUTLASS library patterns for optimized GEMM
                r"cutlass::",
                r"cutlass::gemm::",
                r"cutlass::epilogue::",
                r"cutlass::arch::Sm\d+",
                r"GemmConfiguration\s*<",
                r"DefaultGemm\s*<",
            ],
        },
        "cub_algorithms": {
            "weight": 0.9,
            "patterns": [
                # CUB library for advanced algorithms
                r"cub::DeviceReduce::",
                r"cub::DeviceScan::",
                r"cub::DeviceSelect::",
                r"cub::DeviceRadixSort::",
                r"cub::BlockScan\s*<",
                r"cub::BlockRadixSort\s*<",
                r"cub::WarpScan\s*<",
            ],
        },
        "flash_attention_style": {
            "weight": 0.9,
            "patterns": [
                # Flash attention / memory-efficient attention patterns
                r"//\s*[Ff]lash",
                r"//\s*[Rr]ecompute",
                r"//\s*[Bb]lock[- ]?wise\s+(?:softmax|attention)",
                r"scale\s*=.*exp\s*\(.*-.*max",
                r"(?:m_ij|l_ij|O_ij)\s*=",  # Flash attention variables
                r"rescale\s*=.*exp\s*\(",
            ],
        },
        "scan_algorithms": {
            "weight": 1.0,
            "patterns": [
                r"inclusive_scan",
                r"exclusive_scan",
                r"prefix_sum",
                r"cub::BlockScan",
                r"cub::WarpScan",
                r"thrust::inclusive_scan",
                r"thrust::exclusive_scan",
            ],
        },
        "tile_size_constants": {
            "weight": 0.6,
            "patterns": [
                r"(?:constexpr|const|#define)\s+(?:int\s+)?(?:TILE|BLOCK|BM|BN|BK|TM|TN|TK|WARP)_?\w*\s*[=\s]\s*\d+",
                r"#\s*define\s+(?:TILE|BLOCK|BM|BN|BK|TM|TN|TK)_?\w*\s+\d+",
                r"template\s*<\s*int\s+(?:TILE|BLOCK|BM|BN|BK)",
            ],
        },
    },
}

CUDA_PARALLELISM_OPT_PATTERNS = {
    # Level 0: No patterns - thread-level only
    0: {},
    # Level 1: Block-level synchronization
    1: {
        "thread_indexing": {
            "weight": 1.0,
            "patterns": [
                r"threadIdx\.(?:x|y|z)\b",
                r"blockIdx\.(?:x|y|z)\b",
                r"blockDim\.(?:x|y|z)\b",
                r"gridDim\.(?:x|y|z)\b",
            ],
        },
        "block_synchronization": {
            "weight": 1.0,
            "patterns": [
                r"__syncthreads\s*\(",
                r"__syncthreads_count\s*\(",
                r"__syncthreads_and\s*\(",
                r"__syncthreads_or\s*\(",
            ],
        },
        "linear_indexing": {
            "weight": 0.8,
            "patterns": [
                # Common 1D/2D/3D index calculations
                r"threadIdx\.x\s*\+\s*blockIdx\.x\s*\*\s*blockDim\.x",
                r"blockDim\.x\s*\*\s*blockIdx\.x\s*\+\s*threadIdx\.x",
                r"(?:tid|idx|index|gid)\s*=.*threadIdx.*blockIdx",
            ],
        },
        "launch_bounds": {
            "weight": 0.7,
            "patterns": [
                r"__launch_bounds__\s*\(",
                r"<<<\s*\d+\s*,\s*\d+\s*>>>",
                r"<<<\s*dim3\s*\(",
                r"cudaOccupancyMaxPotentialBlockSize",
            ],
        },
    },
    # Level 2: Warp-level intrinsics
    2: {
        "warp_shuffle": {
            "weight": 1.0,
            "patterns": [
                r"__shfl_sync\s*\(",
                r"__shfl_up_sync\s*\(",
                r"__shfl_down_sync\s*\(",
                r"__shfl_xor_sync\s*\(",
                r"__shfl(?:_up|_down|_xor)?\s*\(",  # Legacy (deprecated but still used)
            ],
        },
        "warp_vote": {
            "weight": 1.0,
            "patterns": [
                r"__ballot_sync\s*\(",
                r"__all_sync\s*\(",
                r"__any_sync\s*\(",
                r"__uni_sync\s*\(",
                r"__activemask\s*\(",
                r"__popc\s*\(",
                r"__clz\s*\(",
                r"__ffs\s*\(",
            ],
        },
        "warp_reduce": {
            "weight": 1.0,
            "patterns": [
                r"__reduce_add_sync\s*\(",
                r"__reduce_min_sync\s*\(",
                r"__reduce_max_sync\s*\(",
                r"__reduce_and_sync\s*\(",
                r"__reduce_or_sync\s*\(",
            ],
        },
        "warp_match": {
            "weight": 0.9,
            "patterns": [
                r"__match_any_sync\s*\(",
                r"__match_all_sync\s*\(",
            ],
        },
        "warp_sync": {
            "weight": 0.9,
            "patterns": [
                r"__syncwarp\s*\(",
                r"__threadfence_block\s*\(",
                r"__threadfence\s*\(",
                r"__threadfence_system\s*\(",
            ],
        },
        "lane_indexing": {
            "weight": 0.8,
            "patterns": [
                r"(?:lane|lane_id|laneId)\s*=.*%\s*(?:32|warpSize)",
                r"threadIdx\.\w\s*%\s*(?:32|warpSize)",
                r"warp_id\s*=.*\/\s*(?:32|warpSize)",
                r"threadIdx\.\w\s*(?:>>|/)\s*5\b",  # Division by 32
            ],
        },
    },
    # Level 3: Cooperative groups and multi-level coordination
    3: {
        "cooperative_groups": {
            "weight": 1.0,
            "patterns": [
                r"cooperative_groups::",
                r"namespace\s+cg\s*=\s*cooperative_groups",
                r"using\s+namespace\s+cooperative_groups",
                r"cg::thread_block\b",
                r"cg::this_thread_block\s*\(",
                r"cg::this_grid\s*\(",
                r"cg::grid_group\b",
            ],
        },
        "tiled_partition": {
            "weight": 1.0,
            "patterns": [
                r"cg::tiled_partition\s*<\s*\d+\s*>",
                r"cg::tiled_partition\s*\(",
                r"tile32\s*=.*tiled_partition",
                r"thread_block_tile\s*<\s*\d+\s*>",
                r"\.shfl\s*\(",
                r"\.shfl_down\s*\(",
                r"\.shfl_up\s*\(",
                r"\.shfl_xor\s*\(",
            ],
        },
        "cooperative_launch": {
            "weight": 1.0,
            "patterns": [
                r"cudaLaunchCooperativeKernel\s*\(",
                r"cudaLaunchCooperativeKernelMultiDevice\s*\(",
                r"cg::sync\s*\(",
                r"this_grid\(\)\.sync\s*\(",
                r"grid_group\.sync\s*\(",
            ],
        },
        "multi_level_sync": {
            "weight": 1.0,
            "patterns": [
                # Patterns indicating multi-level synchronization
                r"cg::sync\s*\(.*thread_block",
                r"cg::sync\s*\(.*grid",
                r"__syncthreads.*__syncwarp",
                r"__syncwarp.*__syncthreads",
                r"tile\d*\.sync\s*\(",
            ],
        },
        "warp_level_tiling": {
            "weight": 0.9,
            "patterns": [
                # Warp-level tiling patterns
                r"(?:WARP|warp)_(?:M|N|K|SIZE|TILE|ROWS|COLS)",
                r"(?:warp|WARP)_(?:row|col|id|idx)",
                r"(?:lane|LANE)_(?:id|idx|ID|IDX)\s*=",
                r"warps_per_block",
                r"threads_per_warp",
            ],
        },
        "cluster_launch": {
            "weight": 1.0,
            "patterns": [
                # CUDA 11.8+ cluster features
                r"cudaLaunchKernelEx\s*\(",
                r"cudaFuncSetAttribute.*Cluster",
                r"__cluster_dims__\s*\(",
                r"cluster\.sync\s*\(",
                r"cg::cluster_group",
            ],
        },
        "dynamic_parallelism": {
            "weight": 0.8,
            "patterns": [
                # CUDA Dynamic Parallelism
                r"<<<.*>>>.*<<<.*>>>",  # Nested kernel launches
                r"cudaDeviceSynchronize\s*\(",
                r"cudaStreamSynchronize\s*\(",
                r"//.*[Dd]ynamic\s+[Pp]arallelism",
            ],
        },
        "stream_management": {
            "weight": 0.7,
            "patterns": [
                r"cudaStream_t\s+\w+",
                r"cudaStreamCreate\s*\(",
                r"cudaStreamCreateWithFlags\s*\(",
                r"cudaStreamCreateWithPriority\s*\(",
                r"cudaEventRecord\s*\(",
                r"cudaStreamWaitEvent\s*\(",
            ],
        },
    },
}

# ========================================================================
# OPENCL PATTERN DEFINITIONS (for portability)
# ========================================================================
#
# OpenCL patterns mirror the SYCL/CUDA structure for comprehensive
# classification across memory, compute, and parallelism dimensions.

OPENCL_MEMORY_OPT_PATTERNS = {
    # Level 0: No patterns - absence of optimization
    0: {},
    # Level 1: Vectorized/coalesced global memory access
    1: {
        "vector_types": {
            "weight": 1.0,
            "patterns": [
                r"\b(?:float|double|int|uint|short|ushort|char|uchar|long|ulong|half)[234816]\b",
                r"\b(?:float|int|double|half)(?:2|3|4|8|16)\b",
            ],
        },
        "vectorized_load_store": {
            "weight": 1.0,
            "patterns": [
                r"\bvload(?:2|3|4|8|16)\s*\(",
                r"\bvstore(?:2|3|4|8|16)\s*\(",
                r"\bas_(?:float|double|int|uint|short|ushort|char|uchar)(?:2|3|4|8|16)\s*\(",
            ],
        },
        "image_texture_access": {
            "weight": 0.9,
            "patterns": [
                r"\bread_image(?:f|i|ui)\s*\(",
                r"\bwrite_image(?:f|i|ui)\s*\(",
                r"\bimage(?:1d|2d|3d)_t\b",
                r"\bsampler_t\b",
            ],
        },
        "aligned_access": {
            "weight": 0.8,
            "patterns": [
                r"__attribute__\s*\(\(\s*aligned\s*\(\s*\d+\s*\)\s*\)\)",
                r"alignas\s*\(\s*\d+\s*\)",
                r"\b__global\s+[^;\n]*\brestrict\b",
            ],
        },
        "global_address_space": {
            "weight": 0.6,
            "patterns": [
                r"\b__global\b",
                r"\bglobal\s+(?:float|double|int|uint|half)\s*\*",
            ],
        },
    },
    # Level 2: Local memory usage and synchronization
    2: {
        "local_memory": {
            "weight": 1.0,
            "patterns": [
                r"\b__local\b",
                r"\blocal\s+(?:float|double|int|uint|half)\s*\*",
            ],
        },
        "local_memory_tiling": {
            "weight": 0.9,
            "patterns": [
                r"(?:float|double|int|half)\s*(?:tile|smem|slm|shared|local)_?\w*\s*\[",
                r"\[\s*(?:TILE|BLOCK|BM|BN|BK)_?\w*\s*\]\s*\[\s*(?:TILE|BLOCK|BM|BN|BK)_?\w*\s*\]",
            ],
        },
        "local_barriers": {
            "weight": 1.0,
            "patterns": [
                r"\bbarrier\s*\(",
                r"\bwork_group_barrier\s*\(",
                r"CLK_LOCAL_MEM_FENCE",
                r"mem_fence\s*\(",
            ],
        },
        "bank_conflict_avoidance": {
            "weight": 0.7,
            "patterns": [
                r"\[\s*\w+\s*\+\s*[1-4]\s*\]",
                r"\[\s*\w+\s*\]\s*\[\s*\w+\s*\+\s*[1-4]\s*\]",
                r"//.*bank\s*conflict",
            ],
        },
        "memory_fence_control": {
            "weight": 0.7,
            "patterns": [
                r"atomic_work_item_fence\s*\(",
                r"memory_scope_",
                r"memory_order_",
            ],
        },
    },
    # Level 3: Register blocking + async copy + pipelining
    3: {
        "register_blocking": {
            "weight": 1.0,
            "patterns": [
                r"(?:float|double|half)\s+(?:reg|frag|acc|tile)_?\w*\s*\[\s*\d+\s*\]",
                r"(?:float|double|half)\s+(?:reg|frag|acc)_?\w*\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]",
                r"\b__private\s+\w+\s*\[",
            ],
        },
        "async_copy": {
            "weight": 1.0,
            "patterns": [
                r"\basync_work_group_copy\s*\(",
                r"\basync_work_group_strided_copy\s*\(",
                r"\bwait_group_events\s*\(",
            ],
        },
        "prefetch": {
            "weight": 0.8,
            "patterns": [
                r"\bprefetch\s*\(",
                r"__builtin_prefetch\s*\(",
            ],
        },
        "double_buffering": {
            "weight": 0.9,
            "patterns": [
                r"(?:buffer|buf|tile|smem|shared)_?[AB01]\s*\[",
                r"(?:ping|pong)_?(?:buffer|buf|tile)",
                r"(?:stage|buffer|tile)_?(?:idx|index)\s*\^\s*1",
                r"\[\s*\d+\s*%\s*2\s*\]",
            ],
        },
        "unroll_directives": {
            "weight": 0.7,
            "patterns": [
                r"#\s*pragma\s+unroll\b",
                r"#\s*pragma\s+unroll\s+\d+",
            ],
        },
    },
}

OPENCL_COMPUTE_OPT_PATTERNS = {
    # Level 0: No patterns - multi-pass naive algorithms
    0: {},
    # Level 1: Operation fusion and native/fast math
    1: {
        "fma_operations": {
            "weight": 1.0,
            "patterns": [
                r"\bfma\s*\(",
                r"\bmad\s*\(",
                r"\bmad24\s*\(",
                r"\bmul24\s*\(",
            ],
        },
        "native_math": {
            "weight": 0.9,
            "patterns": [
                r"\bnative_(?:exp|exp2|log|log2|powr|sqrt|rsqrt|sin|cos|tan)\s*\(",
                r"\bhalf_(?:exp|log|sqrt|rsqrt)\s*\(",
            ],
        },
        "fused_accumulation": {
            "weight": 0.6,
            "patterns": [
                r"\w+\s*\+=\s*\w+\s*\*\s*\w+",
                r"\w+\s*=\s*fma\s*\(",
            ],
        },
        "fast_math_indicators": {
            "weight": 0.5,
            "patterns": [
                r"-cl-fast-relaxed-math",
                r"-cl-mad-enable",
                r"//\s*fast\s*math",
            ],
        },
    },
    # Level 2: Single-pass / streaming algorithms with group reductions
    2: {
        "group_reduce": {
            "weight": 1.0,
            "patterns": [
                r"\bwork_group_reduce_(?:add|max|min)\s*\(",
                r"\bsub_group_reduce_(?:add|max|min)\s*\(",
            ],
        },
        "atomic_accumulation": {
            "weight": 0.9,
            "patterns": [
                r"\batomic_(?:add|inc|dec|max|min|xchg|or|and|xor)\s*\(",
                r"\batomic_fetch_(?:add|sub|max|min|or|and|xor)\s*\(",
            ],
        },
        "online_algorithm_indicators": {
            "weight": 0.8,
            "patterns": [
                r"running_(?:max|min|sum|mean|var)",
                r"(?:online|streaming)_(?:max|min|sum|softmax)",
                r"(?:local|partial)_(?:max|min|sum)\s*=",
            ],
        },
        "single_pass_comments": {
            "weight": 0.5,
            "patterns": [
                r"//\s*[Ss]ingle[- ]?pass",
                r"//\s*[Ss]treaming",
                r"//\s*[Oo]nline\s+(?:softmax|algorithm|reduction)",
            ],
        },
        "welford_pattern": {
            "weight": 1.0,
            "patterns": [
                r"delta\s*=.*-\s*mean",
                r"mean\s*\+=.*delta.*(?:count|n|i)",
                r"M2\s*\+=",
            ],
        },
    },
    # Level 3: Advanced algorithmic transformations (tiled/blocked/sub-group scans)
    3: {
        "tiled_algorithm": {
            "weight": 1.0,
            "patterns": [
                r"for\s*\([^)]*(?:TILE|BLOCK|BM|BN|BK|tile|block)_?\w*\s*[+<]",
                r"(?:tile|block)_?(?:start|offset|idx|id)\s*=",
                r"(?:num|n)_?(?:tiles|blocks)\s*=",
            ],
        },
        "sub_group_matrix_or_mad": {
            "weight": 0.9,
            "patterns": [
                r"cl_intel_subgroup_matrix_multiply_accumulate",
                r"intel_sub_group_(?:block_read|block_write)",
                r"intel_sub_group_(?:shuffle|shuffle_down|shuffle_up)",
                r"sub_group_reduce_(?:add|max|min)",
            ],
        },
        "scan_algorithms": {
            "weight": 1.0,
            "patterns": [
                r"work_group_scan_(?:inclusive|exclusive)_add\s*\(",
                r"sub_group_scan_(?:inclusive|exclusive)_add\s*\(",
                r"inclusive_scan",
                r"exclusive_scan",
            ],
        },
        "recomputation_tradeoff": {
            "weight": 0.7,
            "patterns": [
                r"//\s*[Rr]ecompute",
                r"//\s*[Ff]lash",
                r"//\s*[Bb]lock[- ]?wise",
                r"scale\s*=.*exp\s*\(.*-.*max",
            ],
        },
        "tile_size_constants": {
            "weight": 0.6,
            "patterns": [
                r"(?:constexpr|const|#define)\s+(?:int\s+)?(?:TILE|BLOCK|BM|BN|BK|TM|TN|TK)_?\w*\s*[=\s]\s*\d+",
            ],
        },
    },
}

OPENCL_PARALLELISM_OPT_PATTERNS = {
    # Level 0: No patterns - thread/work-item only
    0: {},
    # Level 1: Work-group synchronization and indexing
    1: {
        "work_item_indexing": {
            "weight": 1.0,
            "patterns": [
                r"\bget_global_id\s*\(",
                r"\bget_local_id\s*\(",
                r"\bget_group_id\s*\(",
                r"\bget_local_size\s*\(",
                r"\bget_global_size\s*\(",
            ],
        },
        "work_group_barrier": {
            "weight": 1.0,
            "patterns": [
                r"\bbarrier\s*\(",
                r"\bwork_group_barrier\s*\(",
                r"CLK_LOCAL_MEM_FENCE",
                r"CLK_GLOBAL_MEM_FENCE",
            ],
        },
        "work_group_collectives": {
            "weight": 0.9,
            "patterns": [
                r"\bwork_group_(?:broadcast|reduce_(?:add|max|min)|scan_(?:inclusive|exclusive)_add)\s*\(",
            ],
        },
        "reqd_work_group_size": {
            "weight": 0.7,
            "patterns": [
                r"__attribute__\s*\(\(\s*reqd_work_group_size\s*\(",
                r"\[\[cl::reqd_work_group_size\s*\(",
            ],
        },
    },
    # Level 2: Sub-group intrinsics
    2: {
        "sub_group_identity": {
            "weight": 1.0,
            "patterns": [
                r"\bget_sub_group_id\s*\(",
                r"\bget_num_sub_groups\s*\(",
                r"\bget_sub_group_local_id\s*\(",
                r"\bget_sub_group_size\s*\(",
            ],
        },
        "sub_group_collectives": {
            "weight": 1.0,
            "patterns": [
                r"\bsub_group_(?:broadcast|reduce_(?:add|max|min))\s*\(",
                r"\bsub_group_scan_(?:inclusive|exclusive)_add\s*\(",
            ],
        },
        "sub_group_shuffle": {
            "weight": 1.0,
            "patterns": [
                r"intel_sub_group_shuffle\s*\(",
                r"intel_sub_group_shuffle_(?:down|up)\s*\(",
                r"sub_group_broadcast\s*\(",
            ],
        },
        "sub_group_block_ops": {
            "weight": 0.9,
            "patterns": [
                r"intel_sub_group_block_(?:read|write)\s*\(",
                r"intel_sub_group_(?:media_block_read|media_block_write)\s*\(",
            ],
        },
        "reqd_sub_group_size": {
            "weight": 0.8,
            "patterns": [
                r"__attribute__\s*\(\(\s*intel_reqd_sub_group_size\s*\(",
                r"\[\[intel::reqd_sub_group_size\s*\(",
                r"reqd_sub_group_size",
            ],
        },
    },
    # Level 3: Hierarchical and multi-level coordination
    3: {
        "multi_level_coordination": {
            "weight": 1.0,
            "patterns": [
                r"sub_group_.*barrier",
                r"barrier.*sub_group_",
                r"work_group_.*sub_group_",
            ],
        },
        "hierarchical_enqueue": {
            "weight": 0.9,
            "patterns": [
                r"\benqueue_kernel\s*\(",
                r"\bndrange_t\b",
                r"\bclk_event_t\b",
            ],
        },
        "warp_or_subgroup_tiling": {
            "weight": 0.9,
            "patterns": [
                r"(?:WARP|SG|SUBGROUP)_(?:M|N|K|SIZE|TILE)",
                r"(?:warp|sg|subgroup)_(?:row|col|id|idx)",
                r"lane_id\s*=",
            ],
        },
        "work_group_size_attr": {
            "weight": 0.7,
            "patterns": [
                r"__attribute__\s*\(\(\s*reqd_work_group_size\s*\(",
                r"__attribute__\s*\(\(\s*work_group_size_hint\s*\(",
            ],
        },
        "event_task_graph": {
            "weight": 0.6,
            "patterns": [
                r"\bwait_group_events\s*\(",
                r"\bevent_t\b",
                r"\bclk_event_t\b",
            ],
        },
    },
}

# ========================================================================
# ESIMD PATTERN DEFINITIONS (Intel GPU Specific)
# ========================================================================
#
# ESIMD (Explicit SIMD) is Intel's extension for low-level SIMD programming.
# These patterns detect the use of ESIMD features at different sophistication levels.

ESIMD_OPT_PATTERNS = {
    # Level 0: No ESIMD (standard SYCL) - absence of patterns
    0: {},
    # Level 1: Basic ESIMD - simd types and block operations
    1: {
        "esimd_include": {
            "weight": 1.0,
            "patterns": [
                r"#\s*include\s*<sycl/ext/intel/esimd\.hpp>",
                r"#\s*include\s*<sycl/ext/intel/experimental/esimd\.hpp>",
            ],
        },
        "esimd_namespace": {
            "weight": 1.0,
            "patterns": [
                r"sycl::ext::intel::esimd::",
                r"using\s+namespace\s+sycl::ext::intel::esimd",
                r"namespace\s+esimd\s*=\s*sycl::ext::intel::esimd",
            ],
        },
        "esimd_attribute": {
            "weight": 1.0,
            "patterns": [
                r"\[\[intel::sycl_explicit_simd\]\]",
                r"\[\[sycl::reqd_sub_group_size\s*\(\s*1\s*\)\]\].*\[\[intel::sycl_explicit_simd\]\]",
            ],
        },
        "simd_types": {
            "weight": 1.0,
            "patterns": [
                r"\bsimd<\s*(?:float|double|int|uint|short|ushort|half|char|uchar|sycl::half)\s*,\s*\d+\s*>",
                r"\bsimd_mask<\s*\d+\s*>",
                r"\bsimd<[^>]+>\s+\w+",  # simd variable declarations
            ],
        },
        "block_operations": {
            "weight": 1.0,
            "patterns": [
                r"\bblock_load\s*<",
                r"\bblock_store\s*<?\s*\(",
                r"block_load\s*\([^)]+\)",
                r"block_store\s*\([^)]+\)",
            ],
        },
    },
    # Level 2: Optimized ESIMD - LSC, cache hints, named barriers, gather/scatter
    2: {
        "lsc_operations": {
            "weight": 1.0,
            "patterns": [
                r"\blsc_block_load\s*<",
                r"\blsc_block_store\s*<",
                r"\blsc_gather\s*<",
                r"\blsc_scatter\s*<",
                r"\blsc_prefetch\s*<",
                r"\blsc_fence\s*<",
                r"\blsc_atomic_update\s*<",
            ],
        },
        "cache_hints": {
            "weight": 0.9,
            "patterns": [
                r"cache_hint::",
                r"cache_hint::cached",
                r"cache_hint::uncached",
                r"cache_hint::streaming",
                r"cache_hint::write_back",
                r"cache_hint::write_through",
                r"lsc_data_size::",
            ],
        },
        "named_barriers": {
            "weight": 1.0,
            "patterns": [
                r"named_barrier_init\s*<",
                r"named_barrier_wait\s*<",
                r"named_barrier_signal\s*<",
                r"named_barrier_arrive\s*<",
                r"nbarrier_init\s*\(",
                r"nbarrier_wait\s*\(",
                r"nbarrier_signal\s*\(",
            ],
        },
        "gather_scatter": {
            "weight": 0.9,
            "patterns": [
                r"\bgather\s*<",
                r"\bscatter\s*<",
                r"gather\s*\(\s*[^)]*simd",
                r"scatter\s*\(\s*[^)]*simd",
            ],
        },
        "slm_esimd": {
            "weight": 0.8,
            "patterns": [
                r"slm_block_load\s*<",
                r"slm_block_store\s*<",
                r"slm_gather\s*<",
                r"slm_scatter\s*<",
                r"slm_allocator\s*<",
                r"slm_init\s*<",
            ],
        },
        "experimental_esimd": {
            "weight": 0.7,
            "patterns": [
                r"sycl::ext::intel::experimental::esimd::",
                r"using\s+namespace\s+sycl::ext::intel::experimental::esimd",
            ],
        },
    },
    # Level 3: Expert ESIMD - DPAS, register tiling, software pipelining, xmx
    3: {
        "dpas_operations": {
            "weight": 1.0,
            "patterns": [
                r"\bdpas\s*<",
                r"dpas<[^>]+>",
                r"dpasw\s*<",
                r"dpasw<[^>]+>",
                r"xmx::dpas",
                r"xmx::dpasw",
            ],
        },
        "xmx_namespace": {
            "weight": 1.0,
            "patterns": [
                r"sycl::ext::intel::esimd::xmx::",
                r"using\s+namespace\s+sycl::ext::intel::esimd::xmx",
                r"namespace\s+xmx\s*=",
                r"argument_type::BF16",
                r"argument_type::FP16",
                r"argument_type::TF32",
            ],
        },
        "register_tiling_esimd": {
            "weight": 0.9,
            "patterns": [
                # ESIMD-style register tile arrays
                r"simd<[^>]+>\s+reg_?\w*\s*\[\s*\d+\s*\]",
                r"simd<[^>]+>\s+(?:acc|frag|tile)_?\w*\s*\[\s*\d+\s*\]",
                r"simd<[^>]+>\s+\w+\s*\[\s*\d+\s*\]\s*\[\s*\d+\s*\]",
            ],
        },
        "raw_send": {
            "weight": 1.0,
            "patterns": [
                r"\braw_send\s*<",
                r"\braw_sends\s*<",
                r"raw_send\s*\(",
            ],
        },
        "compile_time_properties": {
            "weight": 0.8,
            "patterns": [
                r"\[\[intel::reqd_sub_group_size\s*\(\s*1\s*\)\]\]",
                r"\[\[intel::kernel_args_restrict\]\]",
                r"\[\[intel::no_global_work_offset\]\]",
                r"\[\[intel::num_simd_work_items\s*\(\s*\d+\s*\)\]\]",
            ],
        },
        "software_pipelining_esimd": {
            "weight": 0.7,
            "patterns": [
                # Double-buffering and pipelining patterns in ESIMD context
                r"simd<[^>]+>\s+(?:ping|pong|buf[AB])_?\w*",
                r"simd<[^>]+>\s+\w+\s*\[\s*2\s*\]",  # Double buffer arrays
                r"#\s*pragma\s+unroll.*simd",
            ],
        },
        "memory_properties": {
            "weight": 0.6,
            "patterns": [
                r"properties\s*{",
                r"alignment<\s*\d+\s*>",
                r"cache_hint_L1<",
                r"cache_hint_L2<",
            ],
        },
    },
}

# ========================================================================
# TRITON PATTERN DEFINITIONS
# ========================================================================
#
# Triton is a Python DSL for GPU kernel programming. Kernels are written as
# Python functions decorated with @triton.jit. Programs operate on blocks of
# data rather than individual threads, with Triton handling many low-level
# details automatically.
#
# Memory Optimization Levels:
#   0 = Naive scalar loads/stores or absent masking
#   1 = Blocked loads with proper masks and coalesced access patterns
#   2 = Cache hints, eviction policy control, optimized pointer arithmetic
#   3 = make_block_ptr / block pointer API, pipelining, advanced tiling
#
# Compute Optimization Levels:
#   0 = Simple element-wise operations, no fusion
#   1 = Fused operations, tl.dot for matrix multiply, fast math
#   2 = Single-pass online algorithms with tl.reduce / associative scans
#   3 = Blocked flash-attention style, multi-stage pipelining
#
# Parallelism Optimization Levels:
#   0 = 1D grid only, no tuning
#   1 = Proper BLOCK_SIZE / BLOCK_M / BLOCK_N tiling, 1D blocked programs
#   2 = Multi-dimensional program grids (2D/3D), num_warps / num_stages
#   3 = Auto-tuning with triton.Config, persistent kernels, advanced scheduling

TRITON_MEMORY_OPT_PATTERNS = {
    # Level 0: No patterns - naive access
    0: {},
    # Level 1: Blocked, coalesced, masked loads/stores
    1: {
        "blocked_load_store": {
            "weight": 1.0,
            "patterns": [
                r"\btl\.load\s*\(",
                r"\btl\.store\s*\(",
            ],
        },
        "arange_indexing": {
            "weight": 0.9,
            "patterns": [
                r"\btl\.arange\s*\(",
                r"offsets\s*=.*tl\.arange",
                r"tl\.arange\s*\(\s*0\s*,\s*BLOCK",
            ],
        },
        "masked_access": {
            "weight": 1.0,
            "patterns": [
                r"mask\s*=.*<\s*\w*(?:N|M|K|n|m|k|size|len|num)",
                r"tl\.load\s*\([^)]*mask\s*=",
                r"tl\.store\s*\([^)]*mask\s*=",
                r"other\s*=.*0",  # other= parameter in masked loads
            ],
        },
        "pointer_arithmetic": {
            "weight": 0.8,
            "patterns": [
                r"\w+_ptr\s*\+\s*(?:offsets|tl\.arange|pid|block)",
                r"ptrs\s*=\s*\w+_ptr",
                r"row_start_ptr\s*=",
                r"\w+_ptr\s*\+\s*\w+\s*\*\s*\w+",
            ],
        },
        "block_size_constants": {
            "weight": 0.8,
            "patterns": [
                r"BLOCK_SIZE\s*:\s*tl\.constexpr",
                r"BLOCK_M\s*:\s*tl\.constexpr",
                r"BLOCK_N\s*:\s*tl\.constexpr",
                r"BLOCK_K\s*:\s*tl\.constexpr",
                r"\btl\.constexpr\b",
            ],
        },
    },
    # Level 2: Cache control and optimized memory access
    2: {
        "cache_hints": {
            "weight": 1.0,
            "patterns": [
                r"tl\.load\s*\([^)]*cache_modifier\s*=",
                r"tl\.load\s*\([^)]*eviction_policy\s*=",
                r"cache_modifier\s*=\s*['\"]\.cg['\"]",
                r"cache_modifier\s*=\s*['\"]\.cs['\"]",
                r"eviction_policy\s*=\s*['\"]evict_last['\"]",
                r"eviction_policy\s*=\s*['\"]evict_first['\"]",
            ],
        },
        "volatile_load": {
            "weight": 0.8,
            "patterns": [
                r"tl\.load\s*\([^)]*volatile\s*=\s*True",
                r"is_volatile\s*=\s*True",
            ],
        },
        "aligned_access": {
            "weight": 0.9,
            "patterns": [
                # Checking alignment before vectorized access
                r"tl\.multiple_of\s*\(",
                r"multiple_of\s*\(\s*\w+\s*,\s*(?:16|32|64|128)\s*\)",
                r"#\s*hint.*align",
                r"assume_aligned",
            ],
        },
        "coalesced_2d": {
            "weight": 0.9,
            "patterns": [
                # 2D tile loading patterns
                r"offs_\w*\s*=\s*tl\.arange\s*\(\s*0\s*,\s*BLOCK_(?:M|N|K|SIZE)\s*\)",
                r"offs_\w*\[:, None\]",
                r"offs_\w*\[None, :\]",
                r"\[:, None\]\s*\*\s*\w+\s*\+\s*\[None, :\]",
            ],
        },
    },
    # Level 3: Block pointer API, pipelining, advanced tiling
    3: {
        "block_pointer_api": {
            "weight": 1.0,
            "patterns": [
                r"\btl\.make_block_ptr\s*\(",
                r"\btl\.advance\s*\(",
                r"tl\.load\s*\([^)]*boundary_check",
                r"tl\.store\s*\([^)]*boundary_check",
                r"tl\.load\s*\([^)]*padding_option",
            ],
        },
        "software_pipeline": {
            "weight": 0.9,
            "patterns": [
                r"num_stages\s*=\s*[2-9]",
                r"num_stages\s*:\s*tl\.constexpr",
                r"#\s*pipeline",
                r"#\s*prefetch",
                r"num_stages\s*=.*STAGES",
            ],
        },
        "register_tiling": {
            "weight": 0.9,
            "patterns": [
                # Multiple accumulator blocks per program instance
                r"acc\s*=\s*tl\.zeros\s*\(",
                r"accumulator\s*=\s*tl\.zeros\s*\(",
                r"tl\.zeros\s*\(\s*\[BLOCK_M",
                r"tl\.zeros\s*\(\s*\[BLOCK_N",
                r"tl\.zeros\s*\(\s*\(\s*BLOCK_M",
            ],
        },
        "loop_tiling": {
            "weight": 0.8,
            "patterns": [
                # K-dimension tiling loops
                r"for\s+k\s+in\s+range\s*\(\s*0\s*,\s*\w*[Kk]\w*\s*,\s*BLOCK_K\s*\)",
                r"for\s+k\s+in\s+tl\.range\s*\(",
                r"for\s+\w+\s+in\s+range\s*\([^)]*BLOCK_(?:K|M|N)\s*\)",
                r"K\s*//\s*BLOCK_K",
            ],
        },
    },
}

TRITON_COMPUTE_OPT_PATTERNS = {
    # Level 0: No patterns - naive element-wise
    0: {},
    # Level 1: Fused operations and tl.dot usage
    1: {
        "dot_product": {
            "weight": 1.0,
            "patterns": [
                r"\btl\.dot\s*\(",
                r"tl\.dot\s*\(\s*a\s*,\s*b",
                r"tl\.dot\s*\(\s*q\s*,\s*k",
            ],
        },
        "fused_elementwise": {
            "weight": 0.9,
            "patterns": [
                # Multiple tl operations chained without intermediate storage
                r"tl\.exp\s*\(",
                r"tl\.log\s*\(",
                r"tl\.sqrt\s*\(",
                r"tl\.sigmoid\s*\(",
                r"tl\.where\s*\(",
            ],
        },
        "fast_math": {
            "weight": 0.8,
            "patterns": [
                r"tl\.math\.",
                r"tl\.libdevice\.",
                r"libdevice\.\w+\s*\(",
                r"tl\.extra\.\w+\s*\(",
            ],
        },
        "accumulation": {
            "weight": 0.7,
            "patterns": [
                r"acc\s*\+=",
                r"accumulator\s*\+=",
                r"tl\.zeros\s*\(\s*\[",
                r"tl\.full\s*\(",
            ],
        },
    },
    # Level 2: Single-pass online algorithms with reductions
    2: {
        "tl_reduce": {
            "weight": 1.0,
            "patterns": [
                r"\btl\.reduce\s*\(",
                r"\btl\.sum\s*\(",
                r"\btl\.max\s*\(",
                r"\btl\.min\s*\(",
                r"\btl\.argmax\s*\(",
                r"\btl\.argmin\s*\(",
            ],
        },
        "online_softmax": {
            "weight": 1.0,
            "patterns": [
                r"running_max\s*=",
                r"m_i\s*=",  # Flash-attention notation
                r"l_i\s*=",
                r"tl\.maximum\s*\(",
                r"tl\.where\s*\(\s*\w+\s*>\s*\w+",
                r"log_sum_exp",
            ],
        },
        "associative_scan": {
            "weight": 1.0,
            "patterns": [
                r"\btl\.associative_scan\s*\(",
                r"\btl\.cumsum\s*\(",
                r"\btl\.cumprod\s*\(",
            ],
        },
        "atomic_ops": {
            "weight": 0.9,
            "patterns": [
                r"\btl\.atomic_add\s*\(",
                r"\btl\.atomic_max\s*\(",
                r"\btl\.atomic_min\s*\(",
                r"\btl\.atomic_and\s*\(",
                r"\btl\.atomic_or\s*\(",
                r"\btl\.atomic_xor\s*\(",
                r"\btl\.atomic_cas\s*\(",
                r"\btl\.atomic_xchg\s*\(",
            ],
        },
        "welford_pattern": {
            "weight": 1.0,
            "patterns": [
                r"delta\s*=.*-\s*mean",
                r"mean\s*\+=.*delta",
                r"M2\s*\+=",
                r"#\s*[Ww]elford",
            ],
        },
    },
    # Level 3: Advanced tiled/blocked algorithms
    3: {
        "flash_attention_style": {
            "weight": 1.0,
            "patterns": [
                # Flash attention variables and patterns
                r"m_i\s*=.*-float\s*\(['\"]inf['\"]",
                r"l_i\s*=\s*tl\.zeros",
                r"acc\s*=\s*tl\.zeros",
                r"alpha\s*=\s*tl\.exp\s*\(\s*m_i",
                r"#\s*[Ff]lash",
                r"#\s*[Ff]lash[- ][Aa]ttention",
            ],
        },
        "tiled_matmul_loop": {
            "weight": 1.0,
            "patterns": [
                # Explicit tile loop over K dimension
                r"for\s+k\s+in\s+range\s*\(\s*0\s*,\s*(?:K|tl\.cdiv)",
                r"for\s+\w+\s+in\s+range\s*\([^)]*,\s*BLOCK_K\s*\)",
                r"a\s*=\s*tl\.load.*\n.*b\s*=\s*tl\.load.*\n.*acc\s*\+=\s*tl\.dot",
            ],
        },
        "tile_size_constants": {
            "weight": 0.7,
            "patterns": [
                r"BLOCK_M\s*:\s*tl\.constexpr",
                r"BLOCK_N\s*:\s*tl\.constexpr",
                r"BLOCK_K\s*:\s*tl\.constexpr",
                r"triton\.Config\s*\(\s*\{",
            ],
        },
        "recomputation": {
            "weight": 0.7,
            "patterns": [
                r"#\s*[Rr]ecompute",
                r"#\s*[Rr]ecomputation",
                r"#\s*[Bb]lock[- ]?wise",
                r"scale\s*=.*tl\.exp\s*\(\s*m_i",
            ],
        },
    },
}

TRITON_PARALLELISM_OPT_PATTERNS = {
    # Level 0: No patterns - absent or trivial grid
    0: {},
    # Level 1: Basic 1D blocked programs with program_id
    1: {
        "program_id": {
            "weight": 1.0,
            "patterns": [
                r"\btl\.program_id\s*\(",
                r"pid\s*=\s*tl\.program_id\s*\(",
                r"tl\.program_id\s*\(\s*axis\s*=\s*0\s*\)",
            ],
        },
        "block_size_grid": {
            "weight": 0.9,
            "patterns": [
                r"grid\s*=\s*(?:lambda|triton\.cdiv|\()",
                r"triton\.cdiv\s*\(",
                r"\[\s*triton\.cdiv\s*\(",
                r"lambda\s+meta\s*:",
                r"BLOCK_SIZE\s*=\s*\d+",
            ],
        },
        "block_offsets": {
            "weight": 0.9,
            "patterns": [
                r"block_start\s*=\s*pid\s*\*\s*BLOCK",
                r"offsets\s*=\s*\w*pid\w*\s*\*\s*BLOCK",
                r"pid\s*\*\s*BLOCK_SIZE",
            ],
        },
        "kernel_decorator": {
            "weight": 1.0,
            "patterns": [
                r"@triton\.jit",
                r"@triton\.autotune",
            ],
        },
    },
    # Level 2: Multi-dimensional grids and warp/stage configuration
    2: {
        "multi_dim_program_id": {
            "weight": 1.0,
            "patterns": [
                r"tl\.program_id\s*\(\s*(?:axis\s*=\s*)?[12]\s*\)",
                r"pid_m\s*=\s*tl\.program_id",
                r"pid_n\s*=\s*tl\.program_id",
                r"pid_z\s*=\s*tl\.program_id",
                r"pid_b\s*=\s*tl\.program_id",
            ],
        },
        "num_warps_stages": {
            "weight": 1.0,
            "patterns": [
                r"num_warps\s*=\s*\d+",
                r"num_stages\s*=\s*\d+",
                r"num_warps\s*:\s*tl\.constexpr",
                r"num_stages\s*:\s*tl\.constexpr",
                r"num_ctas\s*=\s*\d+",
            ],
        },
        "grid_2d_3d": {
            "weight": 0.9,
            "patterns": [
                # 2D or 3D grid patterns
                r"grid\s*=\s*\(\s*triton\.cdiv[^)]*\)\s*,\s*triton\.cdiv",
                r"grid\s*=\s*\(\s*\w+\s*,\s*\w+\s*(?:,\s*\w+)?\s*\)",
                r"num_pid_m\s*=\s*triton\.cdiv",
                r"num_pid_n\s*=\s*triton\.cdiv",
            ],
        },
        "swizzle_pid": {
            "weight": 0.8,
            "patterns": [
                # L2 cache swizzling of program IDs
                r"GROUP_SIZE_M",
                r"num_pid_in_group",
                r"group_id\s*=\s*pid\s*//",
                r"first_pid_m\s*=",
                r"#\s*swizzle",
                r"#\s*l2\s*cache",
            ],
        },
    },
    # Level 3: Auto-tuning, persistent kernels, advanced scheduling
    3: {
        "autotuning": {
            "weight": 1.0,
            "patterns": [
                r"@triton\.autotune\s*\(",
                r"triton\.Config\s*\(",
                r"configs\s*=\s*\[",
                r"key\s*=\s*\[",
                r"prune_configs_by\s*=",
                r"warmup\s*=\s*\d+",
                r"rep\s*=\s*\d+",
            ],
        },
        "persistent_kernel": {
            "weight": 1.0,
            "patterns": [
                r"num_sms\s*=",
                r"num_tiles\s*=\s*triton\.cdiv",
                r"tile_id\s*=\s*tl\.program_id",
                r"#\s*[Pp]ersistent",
                r"#\s*[Ss]treaming\s+[Kk]ernel",
            ],
        },
        "tl_num_programs": {
            "weight": 0.9,
            "patterns": [
                r"\btl\.num_programs\s*\(",
                r"num_programs\s*=\s*tl\.num_programs",
            ],
        },
        "warp_specialization": {
            "weight": 0.8,
            "patterns": [
                r"tl\.static_range\s*\(",
                r"warp_id\s*=\s*",
                r"#\s*[Ww]arp\s+[Ss]pecialization",
                r"IS_EVEN_MN\s*:\s*tl\.constexpr",
                r"IS_EVEN_K\s*:\s*tl\.constexpr",
            ],
        },
    },
}
