"""Kernel compilation helpers for building PyTorch extensions.

Selected through ``eval_config.kernel_compiler``. Task authors normally do not use these
directly: see :meth:`kernelfoundry.TestBase.compile_torch_extension` for the high-level
interface called from a task's ``build`` method.
"""

import importlib.machinery
import os
import shutil
import subprocess
import shlex
import sys
import textwrap
from pathlib import Path
import sysconfig
import warnings
from abc import ABC, abstractmethod

try:
    from torch.utils.cpp_extension import _get_pybind11_abi_build_flags, include_paths, TORCH_LIB_PATH
except ImportError:
    warnings.warn("Torch not installed, note that IcpxCompiler will not work.")


IS_WINDOWS = sys.platform == "win32"


def _quote(path: str | os.PathLike) -> str:
    """Quote a path for the shell that will run the command."""
    text = str(path)
    if IS_WINDOWS:
        return f'"{text}"' if " " in text else text
    return shlex.quote(text)


def extension_suffix() -> str:
    """File extension for a loadable Python extension module (``.so`` on Linux and macOS, ``.pyd`` on Windows)."""
    return importlib.machinery.EXTENSION_SUFFIXES[-1]


def find_built_extension(build_dir: str | os.PathLike, extension_name: str) -> Path:
    """Locate the extension module a compiler just produced in ``build_dir``. Search by extension since
    ``torch.utils.cpp_extension.load`` writes ``<name>.so`` on Linux but ``<name>.pyd`` on Windows.

    Raises:
        FileNotFoundError: If no extension module is present
    """
    build_path = Path(build_dir)
    for suffix in reversed(importlib.machinery.EXTENSION_SUFFIXES):
        candidate = build_path / f"{extension_name}{suffix}"
        if candidate.exists():
            return candidate

    contents = sorted(p.name for p in build_path.iterdir()) if build_path.is_dir() else []
    raise FileNotFoundError(
        f"The build reported success but no extension module for '{extension_name}' was found in "
        f"{build_path}. Looked for suffixes {importlib.machinery.EXTENSION_SUFFIXES}. "
        f"Directory contains: {contents or '(nothing)'}"
    )


class BaseKernelCompiler(ABC):
    """Abstract base class for kernel compilers."""

    def __init__(
        self, extension_name: str, src: str, build_dir: str, gpu_arch: str, timeout: int = 120, verbose: bool = False
    ):
        """
        Initialize the Compiler.

        Args:
            extension_name (str): Name of the PyTorch extension to build.
            src (str): Path to the source file.
            build_dir (str): Directory to store the compiled outputs.
            gpu_arch (str): GPU architecture string.
            timeout (int): Timeout for each compilation step in seconds.
            verbose (bool): Whether to enable verbose output.
        """
        self.extension_name = extension_name
        self.src = src
        self.build_dir = build_dir
        self.gpu_arch = gpu_arch
        self.timeout = timeout
        self.verbose = verbose

        # Ensure output directory exists
        Path(self.build_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def compile(self):
        """Compile the source into a loadable PyTorch extension.

        Returns:
            dict: Build artifacts and captured compiler output. The exact keys depend on the
                concrete compiler; all include the collected stdout and stderr so build
                failures can be reported back to the caller.
        """
        pass


class TorchCompiler(BaseKernelCompiler):
    """Compiler class using the torch cpp_extension compiler."""

    #: Exit code the build subprocess uses when the sources compiled but the module would not
    #: import. Distinct from 1 so the caller can tell the two failures apart; see compile().
    LOAD_FAILED_RETURNCODE = 87

    @classmethod
    def failed_phase(cls, returncode: int) -> str | None:
        """Which phase failed: ``"compile"``, ``"load"``, or ``None`` when nothing did."""
        if returncode == 0:
            return None
        return "load" if returncode == cls.LOAD_FAILED_RETURNCODE else "compile"

    def compile(self):
        """Compile the source via ``torch.utils.cpp_extension``.

        Returns:
            dict: Build artifacts and captured compiler output.
        """
        args = {
            "sources": self.src,
            "name": self.extension_name,
            "verbose": self.verbose,
            "build_directory": self.build_dir,
        }

        # `load()` compiles the sources *and* imports the resulting module. Those are different
        # failures with different causes -> distinguish them by returncode
        code = textwrap.dedent(f"""
            import os, sys, json, traceback
            try:
                from torch.utils.cpp_extension import load
                args = {args}
                load(**args)
            except ImportError as e:
                sys.stderr.write(str(e) + "\\n")
                sys.stderr.write(traceback.format_exc() + "\\n")
                sys.exit({self.LOAD_FAILED_RETURNCODE})
            except Exception as e:
                sys.stderr.write(str(e) + "\\n")
                sys.stderr.write(traceback.format_exc() + "\\n")
                sys.exit(1)
        """)
        env = os.environ.copy()
        env["TORCH_CUDA_ARCH_LIST"] = self.gpu_arch
        env["TORCH_XPU_ARCH_LIST"] = self.gpu_arch

        try:
            process = subprocess.Popen(
                [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            stdout, stderr = process.communicate(timeout=self.timeout)
            returncode = process.returncode
            return {
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8"),
                "returncode": returncode,
                "phase": self.failed_phase(returncode),
            }
        except subprocess.TimeoutExpired as e:
            process.kill()
            stdout, stderr = process.communicate()
            returncode = process.returncode
            return {
                "stdout": stdout.decode("utf-8"),
                "stderr": stderr.decode("utf-8") + f"\nCompilation timed out: {e}",
                "returncode": returncode,
            }


class IcpxCompiler(BaseKernelCompiler):
    """Compiler class for SYCL programs using the Intel icpx compiler."""

    def compile(self):
        """Compile the SYCL program into a PyTorch extension."""

        stdout_all = []
        stderr_all = []

        try:
            # Step 1: Compile SYCL source to object file
            obj_file = f"{self.build_dir}/{Path(self.src).stem}.sycl.o"
            stdout, stderr = self._compile_to_object(obj_file)
            stdout_all.append(stdout)
            stderr_all.append(stderr)

            # Step 2: Generate SYCL device binary
            sycl_dlink_file = f"{self.build_dir}/sycl_dlink.o"
            stdout, stderr = self._generate_device_binary(obj_file, sycl_dlink_file)
            stdout_all.append(stdout)
            stderr_all.append(stderr)

            # Step 3: Link object file and device binary into a shared library
            shared_lib = f"{self.build_dir}/{self.extension_name}{extension_suffix()}"
            stdout, stderr = self._link_shared_library(obj_file, sycl_dlink_file, shared_lib)
            stdout_all.append(stdout)
            stderr_all.append(stderr)

            return {
                "stdout": "\n".join(stdout_all),
                "stderr": "\n".join(stderr_all),
                "returncode": 0,
            }
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            stderr_all.append(str(e))
            return {
                "stdout": "\n".join(stdout_all),
                "stderr": "\n".join(stderr_all),
                "returncode": 1,
            }

    def _compile_to_object(self, obj_file: Path):
        """Compile the SYCL source file to an object file."""
        cflags = []
        cflags.append(f"-DTORCH_EXTENSION_NAME={self.extension_name}")
        cflags.append("-DTORCH_API_INCLUDE_EXTENSION_H")

        cflags += [f"{x}" for x in _get_pybind11_abi_build_flags()]

        system_includes = include_paths("cpu")
        python_include_path = sysconfig.get_path("include", scheme="nt" if IS_WINDOWS else "posix_prefix")
        if python_include_path is not None:
            system_includes.append(python_include_path)

        cflags += [f"-isystem {_quote(include)}" for include in system_includes]
        if not IS_WINDOWS:
            cflags.append("-fPIC")
        else:
            cflags.append("-fms-runtime-lib=dll")
        cflags.append("-std=c++17")

        sycl_cflags = []
        sycl_cflags.append("-fsycl")
        sycl_cflags.append("-fsycl-targets=spir64_gen,spir64")
        sycl_cflags.append("-sycl-std=2020")
        # sycl_cflags.append("-fsycl-host-compiler=c++")

        host_cflags = cflags
        host_cflags = [item.replace('\\"', '\\\\"') for item in host_cflags]
        host_cflags = " ".join(host_cflags)

        # sycl_cflags.append(shlex.quote(f"-fsycl-host-compiler-options={host_cflags}"))

        cmd = ["icpx"] + cflags + sycl_cflags + ["-c", "-x", "c++", _quote(self.src), "-o", _quote(obj_file)]

        cmd = " ".join(cmd)

        return self._run_command(cmd, "Compiling SYCL source to object file")

    def _generate_device_binary(self, obj_file: Path, sycl_dlink_file: Path):
        """Generate the SYCL device binary."""
        cmd = [
            "icpx",
            _quote(obj_file),
            "-o",
            _quote(sycl_dlink_file),
            "-fsycl",
            "-fsycl-link",
            "--offload-compress",
            "-fsycl-targets=spir64_gen,spir64",
        ]
        if IS_WINDOWS:
            cmd.append("-fms-runtime-lib=dll")
        arch_list = self.gpu_arch
        if arch_list != "":
            cmd += [f'-Xs "-device {arch_list}"']
        cmd = " ".join(cmd)
        return self._run_command(cmd, "Generating SYCL device binary")

    #: Torch libraries the built extension links against. The first four exist in every torch
    #: build; the XPU pair only in an XPU one, and are added when present because a SYCL kernel
    #: that obtains its queue from torch references c10::xpu symbols.
    _TORCH_LINK_LIBS = ("c10", "torch_cpu", "torch", "torch_python")
    _TORCH_LINK_LIBS_XPU = ("c10_xpu", "torch_xpu")

    def _link_shared_library(self, obj_file: Path, sycl_dlink_file: Path, shared_lib: Path):
        """Link the object file and device binary into a shared library."""
        if IS_WINDOWS:
            cmd = self._windows_link_cmd(obj_file, sycl_dlink_file, shared_lib)
        else:
            cmd = [
                "c++",
                str(obj_file),
                str(sycl_dlink_file),
                "-shared",
                f"-L{TORCH_LIB_PATH}",
                "-lc10",
                "-ltorch_cpu",
                "-ltorch",
                "-ltorch_python",
                "-o",
                str(shared_lib),
            ]
        cmd = " ".join(cmd)
        return self._run_command(cmd, "Linking shared library")

    def _windows_link_cmd(self, obj_file: Path, sycl_dlink_file: Path, shared_lib: Path) -> list[str]:
        """Build the Windows/MSVC link command."""
        torch_lib_dir = Path(TORCH_LIB_PATH)
        libs = [torch_lib_dir / f"{name}.lib" for name in self._TORCH_LINK_LIBS]
        libs += [path for name in self._TORCH_LINK_LIBS_XPU if (path := torch_lib_dir / f"{name}.lib").exists()]

        python_lib = (
            Path(sysconfig.get_config_var("installed_base") or sys.base_prefix)
            / "libs"
            / f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        )
        if python_lib.exists():
            libs.append(python_lib)
        else:
            warnings.warn(f"Python import library not found at {python_lib}; the link will likely fail.")

        sycl_lib = self._find_sycl_runtime_lib()
        if sycl_lib is not None:
            libs.append(sycl_lib)
        else:
            warnings.warn("SYCL runtime import library (sycl8.lib) not found. Set CMPLR_ROOT or add icpx to PATH.")

        crt = []
        for static_lib in ("libucrt.lib", "libcmt.lib", "libcpmt.lib", "libvcruntime.lib"):
            crt += ["-Xlinker", f"/NODEFAULTLIB:{static_lib}"]
        crt += ["-Xlinker", "/DEFAULTLIB:ucrt.lib", "-Xlinker", "/DEFAULTLIB:msvcrt.lib"]
        crt += ["-Xlinker", "/DEFAULTLIB:msvcprt.lib", "-Xlinker", "/DEFAULTLIB:vcruntime.lib"]

        return (
            ["icpx", _quote(obj_file), _quote(sycl_dlink_file), "-shared", "-fms-runtime-lib=dll"]
            + crt
            + [_quote(lib) for lib in libs]
            + ["-o", _quote(shared_lib)]
        )

    @staticmethod
    def _find_sycl_runtime_lib() -> Path | None:
        """Locate the Windows SYCL runtime import library."""
        roots = []
        if cmplr_root := os.environ.get("CMPLR_ROOT"):
            roots.append(Path(cmplr_root) / "lib")
        if icpx_path := shutil.which("icpx"):
            roots.append(Path(icpx_path).resolve().parent.parent / "lib")

        for root in roots:
            for name in ("sycl8.lib", "sycl.lib"):
                if (candidate := root / name).exists():
                    return candidate
        return None

    def _run_command(self, cmd: list, description: str):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = process.communicate(timeout=self.timeout)
            returncode = process.returncode
            print(stdout)
            print(stderr)
            if returncode != 0:
                raise RuntimeError(f"[ERROR] {description} failed")
            return stdout, stderr
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[ERROR] {description} failed: {e}")


if __name__ == "__main__":
    compiler = IcpxCompiler(
        extension_name="pytorch_operation_v1",
        src="runs/test/generated_kernel_level_1_problem_19_trial_1_v0.sycl",
        build_dir="runs/test/build",
    )
    result = compiler.compile()
    if result["returncode"] == 0:
        print(f"[INFO] Compilation succeeded.")
    else:
        print(f"[ERROR] Compilation failed. Error: {result['stderr']}")
