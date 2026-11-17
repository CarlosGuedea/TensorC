"""Build configuration for the tensorc Python package.

Compiles the vendored C11 core (tensorc/core/minitf.c + ndtensor.c,
mirrors of the repository root) together with the CPython extension into a
single native module. On macOS it links Apple Accelerate, mirroring the
CMake build. Refresh the vendored core with python/sync_core.sh after
changing the C sources.
"""
import sys

from setuptools import Extension, setup

extra_compile_args = ["-O3", "-std=c11"]
extra_link_args = []
define_macros = []
libraries = []
if sys.platform == "darwin":
    define_macros.append(("TENSORC_HAS_ACCELERATE", "1"))
    define_macros.append(("ACCELERATE_NEW_LAPACK", "1"))
    extra_link_args.extend(["-framework", "Accelerate"])
elif sys.platform.startswith("linux"):
    libraries.append("m")
# En Windows la biblioteca matemática va incluida en el CRT: no enlazar
# "m" (no existe m.lib y LNK1181 lo rechaza).

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
)

setup(ext_modules=[extension])
