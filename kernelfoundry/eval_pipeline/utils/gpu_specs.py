"""
A List of GPU Specs to include in the prompt

"""

import kernelfoundry.eval_pipeline.utils.hw_datatypes as hw_dt

# Example of extraction of GPU ISA:
# * hw_dt.get_supported_datatypes(gpu_sp.GPU_SPEC_INFO.get('Flex 170')["ISA_GPU"])
#   returns the supported datatypes for Flex 170
# * hw_dt.get_supported_datatypes(gpu_sp.GPU_SPEC_INFO_BY_CPU_NAME.get('Intel(R) Core(TM) Ultra 7 268V 2.20GHz')["ISA_GPU"])
#   return the supported GPU datatypes for GPU of 268V

GPU_SPEC_INFO = {
    "L40S": {
        "GPU Architecture": "Ada",
        "GPU Memory": "48GB GDDR6 with ECC",
        "Memory Bandwidth": "864 GB/s",
        "RT Core Performance TFLOPS": "212",
        "FP32 TFLOPS": "91.6",
        "TF32 Tensor Core TFLOPS": "183.2 (366 with sparsity)",
        "FP16 Tensor Core TFLOPS": "362.05 (733 with sparsity)",
        "FP8 Tensor Core TFLOPS": "733 (1466 with sparsity)",
        "Peak INT8 Tensor TOPS": "733 (1466 with sparsity)",
        "Peak INT4 Tensor TOPS": "733 (1466 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "H100": {
        "GPU Architecture": "Hopper",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "3.35 TB/s",
        "FP64 TFLOPS": "34",
        "FP64 Tensor Core TFLOPS": "67",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "989 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "1979 with sparsity",
        "FP16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP8 Tensor Core TFLOPS": "3958 with sparsity",
        "INT8 Tensor Core TOPS": "3958 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    # this is 40GB (Standard)
    "A100": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "40GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A100-80GB": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "L4": {
        "GPU Architecture": "Ada",
        "GPU Memory": "24GB",
        "Memory Bandwidth": "300 GB/s",
        "FP32 TFLOPS": "30.3",
        "TF32 Tensor Core TFLOPS": "120 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "242 with sparsity",
        "FP8 Tensor Core TFLOPS": "485 with sparsity",
        "INT8 Tensor Core TOPS": "485 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "T4": {
        "GPU Architecture": "Turing",
        "GPU Memory": "16 GB GDDR6",
        "Memory Bandwidth": "300 GB/s",
        "Single-Precision TFLOPS": "8.1",
        "Mixed-Precision (FP16/FP32) TFLOPS": "65",
        "INT8 TOPS": "130",
        "INT4 TOPS": "260",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "16",
        "Shared memory capacity per SM": "64 KB",
    },
    "A10G": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "24GB GDDR6",
        "Memory Bandwidth": "600 GB/s",
        "FP32 TFLOPS": "31.2",
        "TF32 Tensor Core TFLOPS": "62.5 (125 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "125 (250 with sparsity)",
        "FP16 Tensor Core TFLOPS": "125 (250 with sparsity)",
        "INT8 Tensor Core TOPS": "250 (500 with sparsity)",
        "INT4 Tensor Core TOPS": "500 (1000 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A6000": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "48GB GDDR6",
        "Memory Bandwidth": "768 GB/s",
        "FP32 TFLOPS": "38.7",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "Flex 170": {
        "Xe-cores": 32,
        "Render Slices": 8,
        "Ray Tracing Units": 32,
        "Intel® Xe Matrix Extensions (Intel® XMX) Engines": 512,
        "Execution Units": 512,
        "Graphics Max Dynamic Clock": "2050 MHz",
        "Intel® Xe Matrix Extensions (Intel® XMX) Max Dynamic Clock": "1950 MHz",
        "TBP": "150 W",
        "Memory Size": "16 GB",
        "Memory Type": "GDDR6",
        "Memory Interface": "256 bit",
        "Memory Bandwidth": "576 GB/s",
        "Device ID": "0x56c0",
        "ISA_GPU": "Xe-HPG",
    },
    "A770": {
        "Xe-cores": 32,
        "Render Slices": 8,
        "Ray Tracing Units": 32,
        "Intel® Xe Matrix Extensions (Intel® XMX) Engines": 512,
        "Execution Units": 512,
        "Graphics Clock": "2100 MHz",
        "GPU Peak TOPS (Int8)": 262,
        "TBP": "225 W",
        "Memory Size": "16 GB",
        "Memory Type": "GDDR6",
        "Memory Interface": "256 bit",
        "Memory Bandwidth": "560 GB/s",
        "Graphics Memory Speed": "17.5 Gbps",
        "Device ID": "0x56A0",
        "ISA_GPU": "Xe-HPG",
    },
    "B580": {
        "Xe-cores": 20,
        "Render Slices": 5,
        "Ray Tracing Units": 20,
        "Intel® Xe Matrix Extensions (Intel® XMX) Engines": 160,
        "Xe Vector Engines": 160,
        "Graphics Clock": 2670,
        "GPU Peak TOPS (Int8)": 233,
        "TBP": 190,
        "PCI Express Configurations ‡": "PCI Express 4.0 x8",
        "Device ID": "0xE20B",
        "Memory": "12 GB GDDR6",
        "Memory Interface": "192 bit",
        "Memory Bandwidth": 456,
        "Memory Speed": 19,
        "ISA_GPU": "Xe2-HPG",
    },
}

GPU_SPEC_INFO_BY_CPU_NAME = {
    "Intel(R) Core(TM) Ultra 7 258V": {
        "Processor Number": "258V",
        "GPU Name": "Intel® Arc™ 140V GPU",
        "Graphics Max Dynamic Frequency": "1.95 GHz",
        "GPU Peak TOPS (Int8)": "64",
        "Xe-cores": 8,
        "Ray Tracing": "Yes",
        "Max Memory Size (dependent on memory type)": "32 GB",
        "Memory Types": "LPDDR5X up to 8533 MT/s",
        "Max # of Memory Channels": "2",
        "Memory Bandwidth": 133,
        "Device ID": "0x64A0",
        "ISA_GPU": "Xe2-LPG",
    },
    "Intel(R) Core(TM) Ultra 7 268V": {
        "Processor Number": "268V",
        "GPU Name": "Intel® Arc™ 140V GPU",
        "Graphics Max Dynamic Frequency": "2 GHz",
        "GPU Peak TOPS (Int8)": "66",
        "Xe-cores": 8,
        "Ray Tracing": "Yes",
        "Max Memory Size (dependent on memory type)": "32 GB",
        "Memory Types": "LPDDR5X up to 8533 MT/s",
        "Max # of Memory Channels": "2",
        "Memory Bandwidth": 133,
        "Device ID": "0x64A0",
        "ISA_GPU": "Xe2-LPG",
    },
    # added for codegen-lnl1
    "Intel(R) Core(TM) Ultra 7 268V 2.20GHz": {
        "Processor Number": "268V",
        "GPU Name": "Intel® Arc™ 140V GPU",
        "Graphics Max Dynamic Frequency": "2 GHz",
        "GPU Peak TOPS (Int8)": "66",
        "Xe-cores": 8,
        "Ray Tracing": "Yes",
        "Max Memory Size (dependent on memory type)": "32 GB",
        "Memory Types": "LPDDR5X up to 8533 MT/s",
        "Max # of Memory Channels": "2",
        "Memory Bandwidth": 133,
        "Device ID": "0x64A0",
        "ISA_GPU": "Xe2-LPG",
    },
}

# in the prompt, for every kernel result we need to specify which hardware it ran on
ARCH_TO_NAME = {
    "dg2": "Intel Flex 170 / Arc A770",
    # "Ampere": "Nvidia GeForce3090",
    # "ampere": "Nvidia GeForce3090",
    "Ampere": "Nvidia RTX A6000",
    "ampere": "Nvidia RTX A6000",
    "bmg": "Intel Battlemage",
    "lnl": "Intel Lunar Lake",
    "ptl": "Intel Panther Lake",
}

# for the prompt, the user specifies a gpu arch and we need to include the corresponding specs in the prompt
ARCH_TO_SPECS = {
    "dg2": GPU_SPEC_INFO["Flex 170"],
    "Ampere": GPU_SPEC_INFO["A6000"],
    "ampere": GPU_SPEC_INFO["A6000"],
    "bmg": GPU_SPEC_INFO["B580"],
    "lnl": GPU_SPEC_INFO_BY_CPU_NAME["Intel(R) Core(TM) Ultra 7 268V"],
}

# Same as above but keyed by Device ID which is easier to get programmatically
GPU_SPEC_INFO_BY_DEVICE_ID = {v["Device ID"].lower(): v for k, v in GPU_SPEC_INFO.items() if "Device ID" in v}

# Basic GPU concept definitions
GPU_DEFINITIONS = {
    "Thread": "A thread is a single execution unit that can run a single instruction at a time.",
    "Thread Block": "A thread block is a group of threads that can cooperate with each other.",
    "Warp": "A warp is a group of threads that are scheduled together and execute in parallel.",
    "Shared Memory": "Shared memory is a memory space that can be accessed by all threads in a thread block.",
    "Register": "A register is a small memory space that can be accessed by a single thread.",
    "Memory Hierarchy": "Memory hierarchy is a pyramid of memory types with different speeds and sizes.",
    "Memory Bandwidth": "Memory bandwidth is the rate at which data can be read from or stored into memory.",
    "Cache": "Cache is a small memory space that stores frequently accessed data.",
    "HBM": "HBM is a high-bandwidth memory technology that uses 3D-stacked DRAM.",
}


GPU_BEST_PRACTICES = [
    # From https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
    # CUDA Best Practices Section
    "Find ways to parallelize sequential code.",
    "Minimize data transfers between the host and the device.",
    "Adjust kernel launch configuration to maximize device utilization.",
    "Ensure that global memory accesses are coalesced.",
    "Minimize redundant accesses to global memory whenever possible.",
    "Avoid long sequences of diverged execution by threads within the same warp.",
    # we added this to reference the specific GPU architecture
    "Use specialized instructions based on the specific GPU architecture",
]

GPU_ARCH_TO_BL_TIME = {
    "Ampere": "NVIDIA_RTX_A6000",
    "ptl": "Intel_PTL_H",
    "bmg": "Intel_B580",
    "dg2": "Intel_Flex170",
    "lnl": "Intel_LNL",
    # 'A100_modal' 'Intel_A770' # currently not used
}
