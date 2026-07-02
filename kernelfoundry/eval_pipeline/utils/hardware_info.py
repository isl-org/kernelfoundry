from dataclasses import dataclass
from kernelfoundry.eval_pipeline.utils.gpu_specs import *


@dataclass(frozen=True)
class HardwareRoofs:
    """Class representing hardware roofline limits for performance modeling.

    The data is based on the metrics available from the unitrace profiler
    """

    GPU_MEMORY_BYTE_READ: float | None = None  # Peak GPU memory read bandwidth in GB per second
    GPU_MEMORY_BYTE_WRITE: float | None = None  # Peak GPU memory write bandwidth in GB per second
    SLM_BYTE_READ: float | None = None  # Peak SLM read bandwidth in bytes per nanosecond
    SLM_BYTE_WRITE: float | None = None  # Peak SLM write bandwidth in bytes per nanosecond
    XVE_INST_EXECUTED_ALU0_ALL: float | None = None  # Peak ALU0 compute in instructions per nanosecond
    XVE_INST_EXECUTED_ALU1_ALL: float | None = None  # Peak ALU1 compute in instructions per nanosecond
    XVE_INST_EXECUTED_ALU2_ALL: float | None = None  # Peak ALU2 compute in instructions per nanosecond
    XVE_INST_EXECUTED_FP16: float | None = None  # Peak FP16 compute in instructions per nanosecond
    XVE_INST_EXECUTED_FP32: float | None = None  # Peak FP32 compute in instructions per nanosecond
    XVE_INST_EXECUTED_FP64: float | None = None  # Peak FP32 compute in instructions per nanosecond
    XVE_INST_EXECUTED_INT16: float | None = None  # Peak INT16 compute in instructions per nanosecond
    XVE_INST_EXECUTED_INT32: float | None = None  # Peak INT32 compute in instructions per nanosecond
    XVE_INST_EXECUTED_INT64: float | None = None  # Peak INT64 compute in instructions per nanosecond
    XVE_INST_EXECUTED_MATH: float | None = None  # Peak EMATH compute in instructions per nanosecond
    XVE_INST_EXECUTED_XMX_BF16: float | None = None  # Peak XMX BF16 compute in instructions per nanosecond
    XVE_INST_EXECUTED_XMX_FP16: float | None = None  # Peak XMX FP16 compute in instructions per nanosecond
    XVE_INST_EXECUTED_XMX_INT2: float | None = None  # Peak XMX INT2 compute in instructions per nanosecond
    XVE_INST_EXECUTED_XMX_INT4: float | None = None  # Peak XMX INT4 compute in instructions per nanosecond
    XVE_INST_EXECUTED_XMX_INT8: float | None = None  # Peak XMX INT8 compute in instructions per nanosecond

    # The following are not roofs but values that allow to compute the number of ops from the instruction counts
    OPS_PER_INST_FP16: float | None = None  # Operations per FP16 instruction
    OPS_PER_INST_FP32: float | None = None  # Operations per FP32 instruction
    OPS_PER_INST_FP64: float | None = None  # Operations per FP64 instruction

    def get(self, key: str) -> float | None:
        """Get the roofline limit value for a given key, stripping units from the key to match the field names."""
        k = key.replace("[bytes]", "").replace("[events]", "").replace("[%]", "").replace("[GBpS]", "")
        return getattr(self, k, None)

    def __getitem__(self, key: str) -> float | None:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


@dataclass(frozen=True)
class HardwareInfo:
    """Class representing hardware information including ID, specs, and roofline limits."""

    hardware_id: str
    specs_dict: dict
    roofs: HardwareRoofs


# The key is either the cpu name or the gpu pci device name
hardware_info = {
    "Intel(R) Core(TM) Ultra 7 258V": HardwareInfo(
        hardware_id="Intel(R) Core(TM) Ultra 7 258V",
        specs_dict=GPU_SPEC_INFO_BY_CPU_NAME["Intel(R) Core(TM) Ultra 7 258V"],
        roofs=HardwareRoofs(
            GPU_MEMORY_BYTE_READ=136.5,  # from datasheet 8533 MT/s, 2 channels, DDR5=64bit: 8533 MT/s * 64bit * 2ch / 8 = 136 GB/s
            GPU_MEMORY_BYTE_WRITE=136.5,  # from datasheet
            SLM_BYTE_READ=1248,  # estimated based on measurements
            SLM_BYTE_WRITE=1248,  # estimated based on measurements
            XVE_INST_EXECUTED_ALU0_ALL=124.8,  # estimated. 1.950 GHz * 8 XeCores * 8 Vector Engines
            XVE_INST_EXECUTED_ALU1_ALL=124.8,  # estimated
            XVE_INST_EXECUTED_ALU2_ALL=62.4,  # estimated as half of the ALU0/1
            XVE_INST_EXECUTED_FP16=124.8,  # estimated
            XVE_INST_EXECUTED_FP32=124.8,  # estimated
            XVE_INST_EXECUTED_FP64=124.8,  # estimated
            XVE_INST_EXECUTED_INT16=124.8,  # estimated
            XVE_INST_EXECUTED_INT32=124.8,  # estimated
            XVE_INST_EXECUTED_INT64=124.8,  # estimated
            XVE_INST_EXECUTED_MATH=124.8,  # estimated
            XVE_INST_EXECUTED_XMX_BF16=62.4,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_FP16=62.4,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT2=62.4,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT4=62.4,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT8=62.4,  # estimated based on ALU2
            OPS_PER_INST_FP16=64,  # factor measured with ze_peak and SIMD32
            OPS_PER_INST_FP32=32,  # factor measured with ze_peak and SIMD32
            OPS_PER_INST_FP64=2,  # factor measured with ze_peak and SIMD32
        ),
    ),
    "Intel(R) Core(TM) Ultra 7 268V": HardwareInfo(
        hardware_id="Intel(R) Core(TM) Ultra 7 268V",
        specs_dict=GPU_SPEC_INFO_BY_CPU_NAME["Intel(R) Core(TM) Ultra 7 268V"],
        roofs=HardwareRoofs(
            GPU_MEMORY_BYTE_READ=136.5,  # from datasheet 8533 MT/s, 2 channels, DDR5=64bit: 8533 MT/s * 64bit * 2ch / 8 = 136 GB/s
            GPU_MEMORY_BYTE_WRITE=136.5,  # from datasheet
            SLM_BYTE_READ=1280,  # estimated based on measurements
            SLM_BYTE_WRITE=1280,  # estimated based on measurements
            XVE_INST_EXECUTED_ALU0_ALL=128.0,  # estimated. 2.0 GHz * 8 XeCores * 8 Vector Engines
            XVE_INST_EXECUTED_ALU1_ALL=128.0,  # estimated
            XVE_INST_EXECUTED_ALU2_ALL=64.0,  # estimated as half of the ALU0/1
            XVE_INST_EXECUTED_FP16=128.0,  # estimated
            XVE_INST_EXECUTED_FP32=128.0,  # estimated
            XVE_INST_EXECUTED_FP64=128.0,  # estimated
            XVE_INST_EXECUTED_INT16=128.0,  # estimated
            XVE_INST_EXECUTED_INT32=128.0,  # estimated
            XVE_INST_EXECUTED_INT64=128.0,  # estimated
            XVE_INST_EXECUTED_MATH=128.0,  # estimated
            XVE_INST_EXECUTED_XMX_BF16=64.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_FP16=64.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT2=64.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT4=64.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT8=64.0,  # estimated based on ALU2
            OPS_PER_INST_FP16=64,  # factor measured with ze_peak and SIMD32
            OPS_PER_INST_FP32=32,  # factor measured with ze_peak and SIMD32
            OPS_PER_INST_FP64=2,  # factor measured with ze_peak and SIMD32
        ),
    ),
    "Intel Corporation Battlemage G21 [Arc B580]": HardwareInfo(
        hardware_id="Intel Corporation Battlemage G21 [Arc B580]",
        specs_dict=GPU_SPEC_INFO["B580"],
        roofs=HardwareRoofs(
            GPU_MEMORY_BYTE_READ=456.0,  # from datasheet
            GPU_MEMORY_BYTE_WRITE=456.0,  # from datasheet
            SLM_BYTE_READ=4480,  # estimated based on measurements
            SLM_BYTE_WRITE=4480,  # estimated based on measurements
            XVE_INST_EXECUTED_ALU0_ALL=456.0,  # estimated. 2.85 GHz * 20 XeCores * 8 Vector Engines
            XVE_INST_EXECUTED_ALU1_ALL=456.0,  # estimated
            XVE_INST_EXECUTED_ALU2_ALL=228.0,  # estimated as half of the ALU0/1
            XVE_INST_EXECUTED_FP16=456.0,  # estimated
            XVE_INST_EXECUTED_FP32=456.0,  # estimated
            XVE_INST_EXECUTED_FP64=456.0,  # estimated
            XVE_INST_EXECUTED_INT16=456.0,  # estimated
            XVE_INST_EXECUTED_INT32=456.0,  # estimated
            XVE_INST_EXECUTED_INT64=456.0,  # estimated
            XVE_INST_EXECUTED_MATH=456.0,  # estimated
            XVE_INST_EXECUTED_XMX_BF16=228.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_FP16=228.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT2=228.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT4=228.0,  # estimated based on ALU2
            XVE_INST_EXECUTED_XMX_INT8=228.0,  # estimated based on ALU2
            OPS_PER_INST_FP16=64,  # factor measured with ze_peak and SIMD32
            OPS_PER_INST_FP32=32,  # factor measured with ze_peak and SIMD32
            OPS_PER_INST_FP64=2,  # factor measured with ze_peak and SIMD32
        ),
    ),
}
# aliases
hardware_info["Intel(R) Core(TM) Ultra 7 268V 2.20GHz"] = hardware_info["Intel(R) Core(TM) Ultra 7 268V"]
