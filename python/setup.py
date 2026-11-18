"""Build configuration for the tensorc Python package.

Compiles the vendored C11 core (tensorc/core/minitf.c + ndtensor.c,
mirrors of the repository root) together with the CPython extension into a
single native module. On macOS it links Apple Accelerate, mirroring the
CMake build. Refresh the vendored core with python/sync_core.sh after
changing the C sources.

CUDA (Linux): build with the GPU backend via

    TENSORC_CUDA=1 python3 setup.py build_ext --inplace

It locates the toolkit with CUDA_HOME/CUDA_PATH (default /usr/local/cuda),
compiles tensorc/core/tensorc_cuda.cu with nvcc and links cublas/cudart.
"""
import os
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

extra_compile_args = ["-O3", "-std=c11"]
extra_link_args = []
define_macros = []
libraries = []
library_dirs = []
extra_objects = []
if sys.platform == "darwin":
    define_macros.append(("TENSORC_HAS_ACCELERATE", "1"))
    define_macros.append(("ACCELERATE_NEW_LAPACK", "1"))
    extra_link_args.extend(["-framework", "Accelerate"])
elif sys.platform.startswith("linux"):
    libraries.append("m")
# En Windows la biblioteca matemática va incluida en el CRT: no enlazar
# "m" (no existe m.lib y LNK1181 lo rechaza).

use_cuda = os.environ.get("TENSORC_CUDA") == "1"
cuda_home = None
if use_cuda:
    cuda_home = (os.environ.get("CUDA_HOME") or
                 os.environ.get("CUDA_PATH") or
                 "/usr/local/cuda")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        nvcc = shutil.which("nvcc") or nvcc
    # Arquitectura: por defecto la de la GPU presente (-arch=native);
    # se puede fijar con TENSORC_CUDA_ARCH="86" o "80;90" para distribuir.
    cuda_arch = os.environ.get("TENSORC_CUDA_ARCH", "native")
    define_macros.append(("TENSORC_HAS_CUDA", "1"))
    libraries.extend(["cublas", "cudart"])
    # Layout oficial del toolkit: lib64; layout conda (nvidia channel): lib.
    for sub in ("lib64", "lib"):
        candidate = os.path.join(cuda_home, sub)
        if os.path.isdir(candidate):
            library_dirs.append(candidate)


class cuda_build_ext(build_ext):
    """Compila tensorc_cuda.cu con nvcc antes de enlazar la extensión."""

    def run(self):
        if use_cuda:
            obj = os.path.join(self.build_temp, "tensorc_cuda.o")
            os.makedirs(self.build_temp, exist_ok=True)
            cmd = [
                nvcc, "-O3", "-std=c++14", "-Xcompiler", "-fPIC",
                "-DTENSORC_HAS_CUDA=1",
                "-arch=native" if cuda_arch == "native" else "",
                "-Itensorc/core",
                "-c", "tensorc/core/tensorc_cuda.cu",
                "-o", obj,
            ]
            if cuda_arch != "native":
                for arch in cuda_arch.split(";"):
                    cmd.append("-gencode")
                    cmd.append("arch=compute_%s,code=sm_%s" % (arch, arch))
            cmd = [c for c in cmd if c]
            print("CUDA:", " ".join(cmd))
            subprocess.check_call(cmd)
            extra_objects.append(obj)
            self.extensions[0].extra_objects = extra_objects
        super().run()


extension = Extension(
    "tensorc._tensorc",
    sources=[
        "tensorc/core/minitf.c",
        "tensorc/core/ndtensor.c",
        "tensorc/_tensorc.c",
    ],
    include_dirs=["tensorc/core/include", "tensorc/core"],
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_objects=extra_objects,
)

setup(
    ext_modules=[extension],
    cmdclass={"build_ext": cuda_build_ext} if use_cuda else {},
)
