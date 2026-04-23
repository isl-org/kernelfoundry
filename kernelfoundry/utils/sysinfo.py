"""System information helpers."""

from pathlib import Path
import subprocess
import socket
import re


def get_intel_cpu_name():
    """Get the CPU model name from /proc/cpuinfo.

    Returns:
        str: CPU model name, or an empty string if unavailable.
    """
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" in line:
                    k, v = (s.strip() for s in line.split(":", 1))
                    if k.lower() in ("model name",):
                        if v:
                            return v
    except Exception:
        pass
    return ""


def get_torch_version():
    """Get the installed torch version without importing it.

    Returns:
        str: Torch version string.
    """
    # get torch version without importing torch
    from importlib.metadata import version

    return version("torch")


def get_nvcc_version():
    """Get the CUDA nvcc version string if available.

    Returns:
        str: nvcc version string, or an empty string if unavailable.
    """
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[3]
            match = re.search(r"V(\d+\.\d+\.\d+)", version_line)
            if match:
                return match.group(1)
    except Exception:
        pass
    return ""


def get_icpx_version():
    """Get the Intel icpx compiler version string if available.

    Returns:
        str: icpx version string, or an empty string if unavailable.
    """
    try:
        result = subprocess.run(["icpx", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            match = re.search(r"\b\d+\.\d+\.\d+\.\d+\b", version_line)
            if match:
                return match.group(0)
    except Exception:
        pass
    return ""


def get_ocl_driver_version():
    """Get the OpenCL driver version string if available.

    Returns:
        str: OpenCL driver version string, or an empty string if unavailable.
    """
    try:
        result = subprocess.run(["ocloc", "query", "OCL_DRIVER_VERSION"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            return version_line
    except Exception:
        pass
    return ""


def get_neo_revision():
    """Get the Intel NEO revision string if available.

    Returns:
        str: Intel NEO revision string, or an empty string if unavailable.
    """
    try:
        result = subprocess.run(["ocloc", "query", "NEO_REVISION"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            return version_line
    except Exception:
        pass
    return ""


def get_igc_revision():
    """Get the Intel IGC revision string if available.

    Returns:
        str: Intel IGC revision string, or an empty string if unavailable.
    """
    try:
        result = subprocess.run(["ocloc", "query", "IGC_REVISION"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            return version_line
    except Exception:
        pass
    return ""


def get_git_commit_hash() -> str | None:
    """Get the git describe hash for the current repo state.

    Returns:
        str: Git describe hash, or an empty string if unavailable.
    """
    try:
        cmd = ["git", "describe", "--always", "--dirty"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""


def discover_intel_gpus() -> list[tuple[Path, str, str]]:
    """Discover Intel GPUs by inspecting DRM devices in sysfs.

    Returns:
        list[tuple[Path, str, str]]: Tuples of (card_path, device_id, pci_slot_name).
            The device ID describes the GPU model, e.g. "0x56c0", and can be passed to
            the compiler via TORCH_XPU_ARCH_LIST.
    """
    result = []
    ignore_devices = [
        "0x4680",  # AlderLake iGPU
    ]
    for i in range(8):
        card_path = Path(f"/sys/class/drm/card{i}")
        uevent_path = card_path / "device" / "uevent"
        if uevent_path.exists():
            uevent_info = uevent_path.read_text()
            if "DRIVER=xe" in uevent_info or "DRIVER=i915" in uevent_info:
                pci_addr = uevent_info.split("PCI_SLOT_NAME=")[1].splitlines()[0]
                device_id = (uevent_path.parent / "device").read_text().strip()
                if device_id not in ignore_devices:
                    result.append((card_path, device_id, pci_addr))

    return result


def get_device_name_by_pci_address(pci_address: str) -> str:
    """Get the device name for a PCI address via lspci.

    Args:
        pci_address (str): PCI address of the device (e.g., "0000:3b:00.0").

    Returns:
        str: Device name, or an empty string if not found.
    """
    try:
        result = subprocess.run(
            ["lspci", "-s", pci_address],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            line = result.stdout.strip()
            name = line.split(":", 2)[-1].strip()
            return name
    except Exception:
        pass
    return ""


def get_nvidia_compute_capabilities() -> list[str]:
    """Get NVIDIA GPU compute capabilities from nvidia-smi.

    Returns:
        list[str]: Compute capability strings (e.g., ["7.5", "8.6"]).
    """
    capabilities = set()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                cap = line.strip()
                if cap:
                    capabilities.add(cap)
    except Exception:
        pass
    return list(capabilities)


def get_nvidia_gpu_name() -> str:
    """Get the first NVIDIA GPU name if present.

    Returns:
        str: GPU name, or an empty string if not found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True
        )
        if result.returncode == 0:
            names = result.stdout.strip().splitlines()
            return names[0] if names else ""
    except Exception:
        pass
    return ""


def get_worker_info():
    """Get system info for the current worker.

    Returns:
        dict: System info map including compiler, driver, and device details.
    """
    if get_worker_info.info is None:
        gpu_name = get_nvidia_gpu_name()
        try:
            _, device_id, pci_addr = discover_intel_gpus()[-1]
            gpu_name = get_device_name_by_pci_address(pci_addr)
        except:
            device_id = ""
        get_worker_info.info = {
            "nvcc_version": get_nvcc_version(),
            "icpx_version": get_icpx_version(),
            "ocl_driver_version": get_ocl_driver_version(),
            "neo_revision": get_neo_revision(),
            "igc_revision": get_igc_revision(),
            "cpu_info": get_intel_cpu_name(),
            "device_id": device_id,
            "gpu_name": gpu_name,
            "nvidia_compute_capabilities": " ".join(get_nvidia_compute_capabilities()),
            "git_commit_hash": get_git_commit_hash(),
            "torch_version": get_torch_version(),
            "hostname": socket.gethostname(),
        }
    return get_worker_info.info


get_worker_info.info = None
