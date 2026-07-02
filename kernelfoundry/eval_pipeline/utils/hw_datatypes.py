"""
provides information on the GPU ISAs and supported datatypes.


info taken from https://uxlfoundation.github.io/oneDNN/dev_guide_data_types.html

OneDNN Data Type Support by microArchitecture (uArch)

Legend:
True indicates the hardware support  this data type
False indicates no support for this data type
Note: Support via conversion to higher precision is not tracked in this simplified representation

Footnotes (CPU ISA):
s8/u8 for AVX2 and AVX-512: (1) See Nuances of int8 Computations in the Developer Guide for additional limitations related to int8 arithmetic
bf16 for AVX-512 and AVX-512 with Intel DL Boost (int8): (2) The library has functional bfloat16 support on processors with Intel AVX-512 Byte and Word Instructions (AVX512BW)
support for validation purposes. The performance of bfloat16 primitives on platforms without hardware acceleration
for bfloat16 is 3-4x lower in comparison to the same operations on the fp32 data type
f16 for Intel AVX10.1/512 with Intel AMX (int8, bf16): (3) Intel AVX-512 f16 instructions accumulate to f16. To avoid overflow, the f16 primitives might up-convert the data
to f32 before performing math operations. This can lead to scenarios where a f16 primitive may perform slower than similar f32 primitive


Footnotes (GPU ISA):
f16 for Xe-LP, Xe-LPG, Xe-HPG: (1) Xe-LP architecture does not natively support f16 operations with f32 accumulation. Consider using relaxed accumulation mode for the best performance results.
"""

# GPU/XPU ISA Support
gpu_datatype_support = {
    "Xe-LP": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Xe-LPG": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Xe-HPG": {
        "f64": False,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Xe-HPC": {
        "f64": True,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Xe2-LPG": {
        "f64": True,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Xe2-HPG": {
        "f64": True,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Xe3-LPG": {
        "f64": True,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
}

# CPU ISA Support
cpu_datatype_support = {
    "Intel SSE4.1": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": False,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": False,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX2": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX2 with Intel DL Boost (int8)": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX-512": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX-512 with Intel DL Boost (int8)": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX-512 with Intel DL Boost (int8, bf16)": {
        "f64": False,
        "f32": True,
        "bf16": True,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX2 with Intel DL Boost (int8) and NE_CONVERT": {
        "f64": False,
        "f32": True,
        "bf16": False,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX10.1/512 with Intel AMX (int8, bf16)": {
        "f64": False,
        "f32": True,
        "bf16": True,
        "f16": False,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX10.1/512 with Intel AMX (int8, bf16, f16)": {
        "f64": False,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX10.2": {
        "f64": False,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": False,
        "f4_e2m1": False,
        "s4/u4": False,
    },
    "Intel AVX10.2 with Intel AMX (int8, bf16, fp16, fp8)": {
        "f64": False,
        "f32": True,
        "bf16": True,
        "f16": True,
        "s8/u8": True,
        "f8": True,
        "f4_e2m1": False,
        "s4/u4": False,
    },
}

# Combined dictionary for backward compatibility and unified access
datatype_support = {**gpu_datatype_support, **cpu_datatype_support}


def get_datatype_support(isa, datatype=None):
    """
    Get data type support information for a specific ISA

    Args:
        isa (str): The ISA/uArch name (e.g., "Xe-HPG", "Intel AVX-512")
        datatype (str, optional): Specific data type to query (e.g., "bf16")

    Returns:
        dict or str: Support information for all datatypes or specific datatype
    """
    # Use the combined datatype_support dictionary
    support_dict = datatype_support

    if isa not in support_dict:
        return f"ISA '{isa}' not found in support matrix"

    if datatype:
        if datatype not in support_dict[isa]:
            return f"Data type '{datatype}' not found for ISA '{isa}'"
        return support_dict[isa][datatype]

    return support_dict[isa]


def get_supported_datatypes(isa, native_only=True, device_type="gpu"):
    """
    Get list of supported data types for a specific ISA

    Args:
        isa (str): The ISA/uArch name
        native_only (bool): If True, only return natively supported types (True values). Default is True.
        device_type (str, optional): "gpu" or "cpu" to search specific device type

    Returns:
        list: List of supported data types
    """
    # Determine which dictionary to use
    if device_type == "gpu":
        support_dict = gpu_datatype_support
    elif device_type == "cpu":
        support_dict = cpu_datatype_support
    else:
        support_dict = datatype_support

    if isa not in support_dict:
        return []

    supported = []
    for datatype, support in support_dict[isa].items():
        # With boolean values, we only include True values
        # native_only parameter is effectively always True now
        if support is True:
            supported.append(datatype)

    return supported


def get_isas_supporting_datatype(datatype, native_only=False, device_type=None):
    """
    Get list of ISAs that support a specific data type

    Args:
        datatype (str): The data type to query
        native_only (bool): If True, only return ISAs with native support (True values)
        device_type (str, optional): "gpu", "cpu", or None for both

    Returns:
        list: List of ISAs supporting the data type
    """
    supporting_isas = []

    # Determine which dictionaries to search
    dicts_to_search = []
    if device_type == "gpu":
        dicts_to_search = [gpu_datatype_support]
    elif device_type == "cpu":
        dicts_to_search = [cpu_datatype_support]
    else:
        dicts_to_search = [gpu_datatype_support, cpu_datatype_support]

    for support_dict in dicts_to_search:
        for isa, datatypes in support_dict.items():
            if datatype in datatypes and datatypes[datatype] is True:
                # With boolean values, we only track True (native support)
                # so native_only is effectively always True
                supporting_isas.append(isa)

    return supporting_isas


def get_cpu_isas():
    """Get list of all CPU ISAs"""
    return list(cpu_datatype_support.keys())


def get_gpu_isas():
    """Get list of all GPU ISAs"""
    return list(gpu_datatype_support.keys())


def get_all_isas():
    """Get list of all ISAs (CPU and GPU)"""
    return list(datatype_support.keys())


def print_support_matrix(device_type=None):
    """
    Print formatted support matrix

    Args:
        device_type (str, optional): "gpu", "cpu", or None for both
    """
    if device_type == "gpu":
        support_dict = gpu_datatype_support
        title = "GPU (XPU) OneDNN Data Type Support Matrix"
    elif device_type == "cpu":
        support_dict = cpu_datatype_support
        title = "CPU OneDNN Data Type Support Matrix"
    else:
        support_dict = datatype_support
        title = "Complete OneDNN Data Type Support Matrix"

    print(title)
    print("=" * len(title))

    # Dynamic column widths based on longest ISA name
    max_isa_len = max(len(isa) for isa in support_dict.keys())
    isa_width = max(max_isa_len, 12)

    header = f"{'ISA':<{isa_width}} {'f64':<6} {'f32':<6} {'bf16':<8} {'f16':<8} {'s8/u8':<8} {'f8':<8} {'f4_e2m1':<8} {'s4/u4':<6}"
    print(header)
    print("-" * len(header))

    for isa, datatypes in support_dict.items():
        row = f"{isa:<{isa_width}}"
        for dt in ["f64", "f32", "bf16", "f16", "s8/u8", "f8", "f4_e2m1", "s4/u4"]:
            # Convert boolean to display string: True -> "✓", False -> ""
            support = "✓" if datatypes[dt] else ""
            if dt in ["bf16", "f16", "s8/u8", "f8", "f4_e2m1"]:
                row += f" {support:<8}"
            else:
                row += f" {support:<6}"
        print(row)
    print()


# Example usage and testing
if __name__ == "__main__":
    print("=== OneDNN Data Type Support Examples ===\n")

    # Test GPU ISA access
    print("GPU ISA Examples:")
    print("Xe-HPG support:", get_datatype_support("Xe-HPG"))
    print("Xe-HPG bf16 support:", get_datatype_support("Xe-HPG", "bf16"))
    print("Xe-HPG supported datatypes:", get_supported_datatypes("Xe-HPG"))
    print("Xe-HPG all datatypes (native_only=False):", get_supported_datatypes("Xe-HPG", native_only=False))
    print()

    # Test CPU ISA access
    print("CPU ISA Examples:")
    print("Intel AVX-512 support:", get_datatype_support("Intel AVX-512"))
    print("Intel AVX-512 bf16 support:", get_datatype_support("Intel AVX-512", "bf16"))
    print("Intel AVX10.2 supported datatypes:", get_supported_datatypes("Intel AVX10.2", device_type="cpu"))
    print(
        "Intel AVX10.2 all datatypes (native_only=False):",
        get_supported_datatypes("Intel AVX10.2", native_only=False, device_type="cpu"),
    )
    print()

    # Test ISAs supporting specific datatype
    print("Cross-ISA Datatype Support:")
    print("All ISAs supporting bf16:", get_isas_supporting_datatype("bf16"))
    print("CPU ISAs with bf16:", get_isas_supporting_datatype("bf16", native_only=True, device_type="cpu"))
    print("GPU ISAs with bf16:", get_isas_supporting_datatype("bf16", native_only=True, device_type="gpu"))
    print()

    # Print matrices
    print_support_matrix("gpu")
    print_support_matrix("cpu")
