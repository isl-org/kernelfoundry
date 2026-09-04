"""Intel GPU discovery tests for Windows and Linux platforms."""

import subprocess
import sys
from pathlib import Path

import pytest

from kernelfoundry.eval_pipeline.utils import sysinfo

# PNPDeviceID for testing
UHD_770_PNP = r"PCI\VEN_8086&DEV_A780&SUBSYS_7E061462&REV_04\3&11583659&0&10"
NVIDIA_PNP = r"PCI\VEN_10DE&DEV_2786&SUBSYS_88EE1043&REV_A1\4&18DC2E69&0&0008"


def _fake_powershell(stdout: str, returncode: int = 0):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr="")

    return run


class TestWindowsDiscovery:
    def test_an_intel_gpu_is_found(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_powershell(UHD_770_PNP + "\n"))

        gpus = sysinfo._discover_intel_gpus_windows()

        assert len(gpus) == 1
        _, device_id, pci = gpus[0]
        assert device_id == "0xa780", "the device ID must match the sysfs form, lowercase and 0x-prefixed"
        assert pci == UHD_770_PNP

    def test_non_intel_adapters_are_ignored(self, monkeypatch):
        """The query filters by vendor, but the parse must not depend on that alone."""
        monkeypatch.setattr(subprocess, "run", _fake_powershell(NVIDIA_PNP + "\n"))

        assert sysinfo._discover_intel_gpus_windows() == []

    def test_ignored_devices_are_still_ignored(self, monkeypatch):
        alderlake = r"PCI\VEN_8086&DEV_4680&SUBSYS_00000000&REV_0C\3&11583659&0&10"
        monkeypatch.setattr(subprocess, "run", _fake_powershell(alderlake + "\n"))

        assert sysinfo._discover_intel_gpus_windows() == []
        assert "0x4680" in sysinfo.IGNORED_INTEL_DEVICE_IDS

    def test_several_adapters_are_all_returned(self, monkeypatch):
        second = r"PCI\VEN_8086&DEV_56C0&SUBSYS_00000000&REV_08\3&11583659&0&11"
        monkeypatch.setattr(subprocess, "run", _fake_powershell(f"{UHD_770_PNP}\n{second}\n"))

        assert [g[1] for g in sysinfo._discover_intel_gpus_windows()] == ["0xa780", "0x56c0"]

    def test_a_failed_query_is_empty_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_powershell("", returncode=1))
        assert sysinfo._discover_intel_gpus_windows() == []

    def test_a_missing_powershell_is_empty_not_an_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FileNotFoundError("powershell")

        monkeypatch.setattr(subprocess, "run", boom)
        assert sysinfo._discover_intel_gpus_windows() == []


class TestDispatch:
    def test_windows_uses_the_pnp_path(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sysinfo, "_discover_intel_gpus_windows", lambda: [(Path("w"), "0xdead", "w")])
        monkeypatch.setattr(sysinfo, "_discover_intel_gpus_sysfs", lambda: pytest.fail("sysfs used on Windows"))

        assert sysinfo.discover_intel_gpus() == [(Path("w"), "0xdead", "w")]

    def test_linux_uses_sysfs(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sysinfo, "_discover_intel_gpus_sysfs", lambda: [(Path("l"), "0xbeef", "l")])
        monkeypatch.setattr(sysinfo, "_discover_intel_gpus_windows", lambda: pytest.fail("PnP used on Linux"))

        assert sysinfo.discover_intel_gpus() == [(Path("l"), "0xbeef", "l")]


class TestSelection:
    """Which card gets built for when the machine has more than one Intel GPU."""

    B580 = (Path("b"), "0xe20b", "b")
    UHD_770 = (Path("i"), "0xa780", "i")

    def _found(self, monkeypatch, gpus):
        monkeypatch.setattr(sysinfo, "discover_intel_gpus", lambda: gpus)

    def test_a_discrete_card_wins_over_the_igpu(self, monkeypatch):
        """Discovery order is not a preference, so the iGPU must lose even when listed first."""
        self._found(monkeypatch, [self.UHD_770, self.B580])

        assert sysinfo.select_intel_gpu() == self.B580

    def test_the_igpu_is_still_used_when_it_is_all_there_is(self, monkeypatch):
        """Skipping it here would fall back to the NVIDIA arch"""
        self._found(monkeypatch, [self.UHD_770])

        assert sysinfo.select_intel_gpu() == self.UHD_770

    def test_an_unrecognised_card_keeps_discovery_order(self, monkeypatch):
        unknown = (Path("u"), "0x1234", "u")
        self._found(monkeypatch, [unknown, self.B580])

        assert sysinfo.select_intel_gpu() == unknown

    def test_no_intel_gpu_is_none_not_an_exception(self, monkeypatch):
        self._found(monkeypatch, [])

        assert sysinfo.select_intel_gpu() is None


class TestDeviceName:
    """lspci does not exist on Windows, so the adapter name came back empty there."""

    LISTING = f"{UHD_770_PNP}|Intel(R) UHD Graphics 770\n" + r"PCI\VEN_8086&DEV_E20B&X|Intel(R) Arc(TM) B580 Graphics"

    def test_the_adapter_name_is_found(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_powershell(self.LISTING))

        assert sysinfo._get_device_name_windows(UHD_770_PNP) == "Intel(R) UHD Graphics 770"

    def test_an_unknown_adapter_is_empty(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_powershell(self.LISTING))

        assert sysinfo._get_device_name_windows(NVIDIA_PNP) == ""

    def test_a_failed_query_is_empty_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _fake_powershell("", returncode=1))

        assert sysinfo._get_device_name_windows(UHD_770_PNP) == ""

    def test_windows_does_not_reach_for_lspci(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(sysinfo, "_get_device_name_windows", lambda pnp: "Intel(R) Arc(TM) B580 Graphics")
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("lspci used on Windows"))

        assert sysinfo.get_device_name_by_pci_address("anything") == "Intel(R) Arc(TM) B580 Graphics"


@pytest.mark.skipif(sys.platform != "win32", reason="exercises the real PnP database")
def test_real_machine_discovery_does_not_crash():
    """Whatever this machine has, the call must return a well-formed list rather than raise."""
    for card_path, device_id, pci in sysinfo.discover_intel_gpus():
        assert isinstance(card_path, Path)
        assert device_id.startswith("0x") and len(device_id) == 6, device_id
        assert pci
