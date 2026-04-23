"""Kernel compilation helpers for building PyTorch extensions.

See :func:`kernelfoundry.custom_test.CustomTask.compile_torch_extension`
for a high-level interface.
"""

import os
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
        pass


class TorchCompiler(BaseKernelCompiler):
    """Compiler class using the torch cpp_extension compiler."""

    def compile(self):
        args = {
            "sources": self.src,
            "name": self.extension_name,
            "verbose": self.verbose,
            "build_directory": self.build_dir,
        }

        code = textwrap.dedent(f"""
            import os, sys, json, traceback
            try:
                from torch.utils.cpp_extension import load
                args = {args}
                load(**args)
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
            shared_lib = f"{self.build_dir}/{self.extension_name}.so"
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
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")
        if python_include_path is not None:
            system_includes.append(python_include_path)

        cflags += [f"-isystem {shlex.quote(include)}" for include in system_includes]
        cflags.append("-fPIC")
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

        cmd = ["icpx"] + cflags + sycl_cflags + ["-c", "-x", "c++", str(self.src), "-o", str(obj_file)]

        cmd = " ".join(cmd)

        return self._run_command(cmd, "Compiling SYCL source to object file")

    def _generate_device_binary(self, obj_file: Path, sycl_dlink_file: Path):
        """Generate the SYCL device binary."""
        cmd = [
            "icpx",
            str(obj_file),
            "-o",
            str(sycl_dlink_file),
            "-fsycl",
            "-fsycl-link",
            "--offload-compress",
            "-fsycl-targets=spir64_gen,spir64",
        ]
        arch_list = self.gpu_arch
        if arch_list != "":
            cmd += [f'-Xs "-device {arch_list}"']
        cmd = " ".join(cmd)
        return self._run_command(cmd, "Generating SYCL device binary")

    def _link_shared_library(self, obj_file: Path, sycl_dlink_file: Path, shared_lib: Path):
        """Link the object file and device binary into a shared library."""
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

    def _run_command(self, cmd: list, description: str):
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
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
