cmake_minimum_required(VERSION 3.23)
project(csrc)

find_package(pybind11 REQUIRED)

# ── Source files ──────────────────────────────────────────────────────────────
# ops.cpp : tensor helpers
# cl.cpp  : OpenCL/CM kernel API, profiling, tensor memory management
FILE(GLOB CPP_SRCS ./*.cpp)

pybind11_add_module(csrc ${CPP_SRCS})
target_compile_features(csrc PRIVATE cxx_std_17)

find_package(OpenMP)
if(OPENMP_FOUND)
    target_compile_options(csrc PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(csrc PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

target_compile_options(csrc PRIVATE -march=core-avx2)

# ── OpenCL ────────────────────────────────────────────────────────────────────
target_include_directories(csrc PRIVATE ../../include)
target_link_libraries(csrc PRIVATE OpenCL)

target_compile_definitions(csrc PRIVATE CL_TARGET_OPENCL_VERSION=300)

set_target_properties(csrc
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
)

target_compile_definitions(csrc PRIVATE PACKAGE_NAME=csrc)

# Installed by scikit-build-core (see kernelfoundry/libs/pyproject.toml) next to
# clops/cl/__init__.py, which does `from .csrc import *`.
install(TARGETS csrc DESTINATION clops/cl)

