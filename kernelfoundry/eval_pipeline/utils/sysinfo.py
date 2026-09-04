"""System information helpers."""

from pathlib import Path
import subprocess
import socket
import re
import sys


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


def get_total_system_memory_kb() -> int:
    """Get the total amount of system memory in kB from /proc/meminfo.

    Returns:
        int: Total memory in kB (from MemTotal), or 0 if unavailable.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = line.split(":", 1)[1].strip().split()[0]
                    return int(mem_total)
    except Exception:
        pass
    return 0


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


#: Intel PCI device IDs that are not useful compute targets.
IGNORED_INTEL_DEVICE_IDS = frozenset(
    {
        "0x4680",  # AlderLake iGPU
    }
)

# Fallback if other checks are not successful: hardcoded list of Intel PCI device IDs for integrated graphics
INTEGRATED_INTEL_DEVICE_IDS = frozenset(
    {
        "0xa780",  # RaptorLake UHD 770
    }
)

#: PCI vendor ID for Intel, as it appears in a Windows PNPDeviceID.
_INTEL_PNP_DEVICE_RE = re.compile(r"VEN_8086&DEV_([0-9A-Fa-f]{4})")


def _is_integrated_intel_gpu(card_path: Path, device_id: str, pci_addr: str) -> bool:
    """Best-effort integrated/discrete classification.

    Prefer runtime hardware signals, and keep the device-id list only as fallback.
    """
    # Linux DRM exposes local VRAM size for discrete GPUs.
    if pci_addr and ":" in pci_addr:
        vram_file = card_path / "device" / "mem_info_vram_total"
        try:
            if vram_file.exists():
                return int(vram_file.read_text(encoding="utf-8", errors="replace").strip()) == 0
        except (OSError, ValueError):
            pass

        # iGPU is typically root-bus-attached (0000:00:*), while dGPU commonly hangs
        # behind a downstream bridge.
        if pci_addr.startswith("0000:00:"):
            return True

    return device_id.lower() in INTEGRATED_INTEL_DEVICE_IDS


def _discover_intel_gpus_sysfs() -> list[tuple[Path, str, str]]:
    """Discover Intel GPUs through the Linux DRM sysfs tree."""
    result = []
    for i in range(8):
        card_path = Path(f"/sys/class/drm/card{i}")
        uevent_path = card_path / "device" / "uevent"
        if uevent_path.exists():
            uevent_info = uevent_path.read_text(encoding="utf-8", errors="replace")
            if "DRIVER=xe" in uevent_info or "DRIVER=i915" in uevent_info:
                pci_addr = uevent_info.split("PCI_SLOT_NAME=")[1].splitlines()[0]
                device_id = (uevent_path.parent / "device").read_text(encoding="utf-8", errors="replace").strip()
                if device_id not in IGNORED_INTEL_DEVICE_IDS:
                    result.append((card_path, device_id, pci_addr))

    return result


def _discover_intel_gpus_windows() -> list[tuple[Path, str, str]]:
    """Discover Intel GPUs through the Windows PnP database.

    Windows has no DRM sysfs, so the device ID comes from the PNPDeviceID instead --
    ``PCI\\VEN_8086&DEV_A780&SUBSYS_...`` yields ``0xa780``, the same form the sysfs path
    produces and the same form ocloc accepts as ``-device``.
    """
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.PNPDeviceID -like 'PCI\\VEN_8086*' } | "
        "ForEach-Object { $_.PNPDeviceID }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    result = []
    for line in proc.stdout.splitlines():
        pnp_id = line.strip()
        match = _INTEL_PNP_DEVICE_RE.search(pnp_id)
        if not match:
            continue
        device_id = f"0x{match.group(1).lower()}"
        if device_id in IGNORED_INTEL_DEVICE_IDS:
            continue
        # There is no card path on Windows; the PnP instance path is the closest stable
        # identifier, and it doubles as the "PCI address" the callers use for naming.
        result.append((Path(pnp_id), device_id, pnp_id))

    return result


def discover_intel_gpus() -> list[tuple[Path, str, str]]:
    """Discover Intel GPUs on this machine.
    Reads DRM devices from sysfs on Linux and the PnP database on Windows.

    Returns:
        list[tuple[Path, str, str]]: Tuples of (card_path, device_id, pci_slot_name).
            The device ID describes the GPU model, e.g. "0x56c0", and can be passed to
            the compiler via TORCH_XPU_ARCH_LIST.
    """
    if sys.platform == "win32":
        return _discover_intel_gpus_windows()
    return _discover_intel_gpus_sysfs()


def select_intel_gpu() -> tuple[Path, str, str] | None:
    """Pick the Intel GPU to target on a machine that has more than one, prefering discrete GPU.

    Returns:
        tuple[Path, str, str] | None: The chosen (card_path, device_id, pci_slot_name), or
            None if the machine has no Intel GPU.
    """
    gpus = discover_intel_gpus()
    if not gpus:
        return None
    discrete = [gpu for gpu in gpus if not _is_integrated_intel_gpu(*gpu)]
    return (discrete or gpus)[0]


def _get_device_name_windows(pnp_device_id: str) -> str:
    """Look up an adapter's friendly name in the Windows PnP database.

    Every adapter is listed and matched here rather than filtered in the query, so that a
    PNPDeviceID never gets interpolated into a PowerShell string.
    """
    script = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID + '|' + $_.Name }"
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""

    wanted = pnp_device_id.strip().casefold()
    for line in proc.stdout.splitlines():
        pnp_id, separator, name = line.partition("|")
        if separator and pnp_id.strip().casefold() == wanted:
            return name.strip()
    return ""


def get_device_name_by_pci_address(pci_address: str) -> str:
    """Get the device name for a PCI address.

    Uses lspci on Linux and the PnP database on Windows, where lspci does not exist and the
    "PCI address" the discovery helpers hand out is really a PNPDeviceID.

    Args:
        pci_address (str): PCI address of the device (e.g., "0000:3b:00.0"), or on Windows a
            PNPDeviceID as returned by :func:`discover_intel_gpus`.

    Returns:
        str: Device name, or an empty string if not found.
    """
    if sys.platform == "win32":
        return _get_device_name_windows(pci_address)
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
            _, device_id, pci_addr = select_intel_gpu()
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
            "total_system_memory_kb": get_total_system_memory_kb(),
            "device_id": device_id,
            "gpu_name": gpu_name,
            "nvidia_compute_capabilities": " ".join(get_nvidia_compute_capabilities()),
            "git_commit_hash": get_git_commit_hash(),
            "torch_version": get_torch_version(),
            "hostname": socket.gethostname(),
        }
    return get_worker_info.info


get_worker_info.info = None
