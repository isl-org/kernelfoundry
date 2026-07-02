"""Tests for profiler feedback collation."""

from collections import OrderedDict

from kernelfoundry.eval_pipeline.profiler_feedback import UnitraceProfilerFeedback


def _make_timeline() -> dict:
    return {
        "traceEvents": [
            {
                "name": "ittapi::model run loop::task.py::Test::test_benchmark[512]",
                "ts": 100,
                "dur": 50,
            },
            {"name": "kernel_a", "ts": 110, "dur": 10, "args": {"id": 1}},
            {
                "name": "ittapi::model run loop::task.py::Test::test_benchmark[1024]",
                "ts": 200,
                "dur": 50,
            },
            {"name": "kernel_b", "ts": 210, "dur": 10, "args": {"id": 2}},
        ]
    }


def _make_compute_basic_csv() -> str:
    return "\n".join(
        [
            "GlobalInstanceId,Kernel,GpuTime[ns],SLM_BANK_CONFLICT_COUNT[events],XVE_INST_EXECUTED_ALU0_ALL[events],GPU_MEMORY_BYTE_READ_RATE[GBpS],GPU_MEMORY_BYTE_WRITE_RATE[GBpS],GPU_MEMORY_BYTE_READ[bytes],GPU_MEMORY_BYTE_WRITE[bytes],SLM_BYTE_READ[bytes],SLM_BYTE_WRITE[bytes],XVE_STALL[%],XVE_INST_EXECUTED_ALU1_ALL[events],XVE_INST_EXECUTED_ALU2_ALL[events]",
            "1,kernel_a[simd=8x1x1],100,1,10,1,1,100,40,1,1,1,10,10",
            "2,kernel_b[simd=16x1x1],200,1,10,1,1,150,60,1,1,1,10,10",
        ]
    )


def _make_memory_profile_csv() -> str:
    return "\n".join(
        [
            "GlobalInstanceId,Kernel,GpuTime[ns],SLM_ACCESS_COUNT[events],SLM_BANK_CONFLICT_COUNT[events]",
            "1,kernel_a[simd=8x1x1],100,10,1",
            "2,kernel_b[simd=16x1x1],200,10,1",
        ]
    )


def _make_vector_engine_csv() -> str:
    return "\n".join(
        [
            "GlobalInstanceId,Kernel,GpuTime[ns],XVE_INST_EXECUTED_XMX_INT8[events],XVE_INST_EXECUTED_ALU0_ALL[events],XVE_INST_EXECUTED_FP16[events],XVE_INST_EXECUTED_FP32[events],XVE_INST_EXECUTED_ALU1_ALL[events],XVE_INST_EXECUTED_FP64[events],XVE_INST_EXECUTED_INT16[events],XVE_INST_EXECUTED_INT32[events],XVE_INST_EXECUTED_INT64[events],XVE_INST_EXECUTED_MATH[events],XVE_INST_EXECUTED_ALU2_ALL[events],XVE_INST_EXECUTED_XMX_BF16[events],XVE_INST_EXECUTED_XMX_FP16[events],XVE_INST_EXECUTED_XMX_INT2[events],XVE_INST_EXECUTED_XMX_INT4[events],XVE_THREADS_OCCUPANCY_ALL[%]",
            "1,kernel_a[simd=8x1x1],100,1,10,5,5,10,1,1,1,1,1,10,1,1,1,1,50",
            "2,kernel_b[simd=16x1x1],200,1,10,5,5,10,1,1,1,1,1,10,1,1,1,1,50",
        ]
    )


def _make_worker_info() -> dict:
    return {
        "cpu_info": "Intel(R) Core(TM) Ultra 7 258V",
        "gpu_name": "Intel(R) Core(TM) Ultra 7 258V",
        "device_id": "0x7d55",
    }


def _make_segment_data(kernel_name: str, runtime_ns: int) -> dict:
    return {
        "ComputeBasic.metrics.1": "\n".join(
            [
                "GlobalInstanceId,Kernel,GpuTime[ns],SLM_BANK_CONFLICT_COUNT[events],XVE_INST_EXECUTED_ALU0_ALL[events],GPU_MEMORY_BYTE_READ_RATE[GBpS],GPU_MEMORY_BYTE_WRITE_RATE[GBpS],GPU_MEMORY_BYTE_READ[bytes],GPU_MEMORY_BYTE_WRITE[bytes],SLM_BYTE_READ[bytes],SLM_BYTE_WRITE[bytes],XVE_STALL[%],XVE_INST_EXECUTED_ALU1_ALL[events],XVE_INST_EXECUTED_ALU2_ALL[events]",
                f"1,{kernel_name},{runtime_ns},1,10,1,1,100,40,1,1,1,10,10",
            ]
        ),
        "MemoryProfile.metrics.1": "\n".join(
            [
                "GlobalInstanceId,Kernel,GpuTime[ns],SLM_ACCESS_COUNT[events],SLM_BANK_CONFLICT_COUNT[events]",
                f"1,{kernel_name},{runtime_ns},10,1",
            ]
        ),
        "VectorEngineProfile.metrics.1": "\n".join(
            [
                "GlobalInstanceId,Kernel,GpuTime[ns],XVE_INST_EXECUTED_XMX_INT8[events],XVE_INST_EXECUTED_ALU0_ALL[events],XVE_INST_EXECUTED_FP16[events],XVE_INST_EXECUTED_FP32[events],XVE_INST_EXECUTED_ALU1_ALL[events],XVE_INST_EXECUTED_FP64[events],XVE_INST_EXECUTED_INT16[events],XVE_INST_EXECUTED_INT32[events],XVE_INST_EXECUTED_INT64[events],XVE_INST_EXECUTED_MATH[events],XVE_INST_EXECUTED_ALU2_ALL[events],XVE_INST_EXECUTED_XMX_BF16[events],XVE_INST_EXECUTED_XMX_FP16[events],XVE_INST_EXECUTED_XMX_INT2[events],XVE_INST_EXECUTED_XMX_INT4[events],XVE_THREADS_OCCUPANCY_ALL[%]",
                f"1,{kernel_name},{runtime_ns},1,10,5,5,10,1,1,1,1,1,10,1,1,1,1,50",
            ]
        ),
    }


def test_unitrace_collate_data_segments_by_labeled_model_run_loop():
    profiler_feedback = UnitraceProfilerFeedback()
    timeline = _make_timeline()
    worker_info = _make_worker_info()
    outputs = {
        "unitrace.001": {"timeline": timeline, "ComputeBasic.metrics.1": _make_compute_basic_csv()},
        "unitrace.002": {"timeline": timeline, "MemoryProfile.metrics.1": _make_memory_profile_csv()},
        "unitrace.003": {"timeline": timeline, "VectorEngineProfile.metrics.1": _make_vector_engine_csv()},
    }

    collated = profiler_feedback.collate_data(outputs)

    assert collated["timeline"] == timeline
    assert list(collated["segments"].keys()) == [
        "task.py::Test::test_benchmark[512]",
        "task.py::Test::test_benchmark[1024]",
    ]

    first_segment = collated["segments"]["task.py::Test::test_benchmark[512]"]
    second_segment = collated["segments"]["task.py::Test::test_benchmark[1024]"]

    assert "kernel_a" in first_segment["ComputeBasic.metrics.1"]
    assert "kernel_b" not in first_segment["ComputeBasic.metrics.1"]
    assert "kernel_b" in second_segment["ComputeBasic.metrics.1"]
    assert "kernel_a" not in second_segment["ComputeBasic.metrics.1"]

    assert "ComputeBasic.metrics.segmented" in collated
    assert "MemoryProfile.metrics.segmented" in collated
    assert "VectorEngineProfile.metrics.segmented" in collated

    feedback = profiler_feedback.create_feedback(collated, worker_info)

    print("\nRendered feedback:\n")
    print(feedback)

    assert feedback.count("Your code has been analyzed with a profiler. Here is a summary of the results:") == 1
    assert "# Benchmark 1: pytest test task.py::Test::test_benchmark[512]" in feedback
    assert "# Benchmark 2: pytest test task.py::Test::test_benchmark[1024]" in feedback
    assert "## Analysis of kernel kernel_a:" in feedback
    assert "## Analysis of kernel kernel_b:" in feedback
    assert "### Runtime and Occupancy:" in feedback


def test_unitrace_create_feedback_uses_all_segments_for_three_or_fewer_shapes():
    profiler_feedback = UnitraceProfilerFeedback()
    worker_info = _make_worker_info()
    data = {
        "segments": OrderedDict(
            [
                ("task.py::Test::test_benchmark[small]", _make_segment_data("kernel_small[simd=8x1x1]", 100)),
                ("task.py::Test::test_benchmark[medium]", _make_segment_data("kernel_medium[simd=8x1x1]", 200)),
                ("task.py::Test::test_benchmark[large]", _make_segment_data("kernel_large[simd=8x1x1]", 300)),
            ]
        )
    }

    feedback = profiler_feedback.create_feedback(data, worker_info)

    assert "# Benchmark 1: pytest test task.py::Test::test_benchmark[small]" in feedback
    assert "# Benchmark 2: pytest test task.py::Test::test_benchmark[medium]" in feedback
    assert "# Benchmark 3: pytest test task.py::Test::test_benchmark[large]" in feedback


def test_unitrace_create_feedback_limits_to_slowest_and_median_when_more_than_three_shapes():
    profiler_feedback = UnitraceProfilerFeedback()
    worker_info = _make_worker_info()
    data = {
        "segments": OrderedDict(
            [
                ("task.py::Test::test_benchmark[tiny]", _make_segment_data("kernel_tiny[simd=8x1x1]", 100)),
                ("task.py::Test::test_benchmark[small]", _make_segment_data("kernel_small[simd=8x1x1]", 200)),
                ("task.py::Test::test_benchmark[medium]", _make_segment_data("kernel_medium[simd=8x1x1]", 300)),
                ("task.py::Test::test_benchmark[large]", _make_segment_data("kernel_large[simd=8x1x1]", 400)),
            ]
        )
    }

    feedback = profiler_feedback.create_feedback(data, worker_info)
    print(feedback)

    assert "# Benchmark 1: pytest test task.py::Test::test_benchmark[small]" in feedback
    assert "# Benchmark 2: pytest test task.py::Test::test_benchmark[large]" in feedback
    assert "task.py::Test::test_benchmark[tiny]" not in feedback
    assert "task.py::Test::test_benchmark[medium]" not in feedback
