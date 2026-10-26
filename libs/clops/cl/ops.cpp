/*
 * ops.cpp — non-SYCL tensor helpers.
 *
 * The original file had ESIMD / SYCL kernel implementations guarded by
 * #if defined(SYCL_LANGUAGE_VERSION).  Those have been removed here since
 * this package is built with g++ (pure OpenCL / CM, no SYCL).
 */

#include "common.hpp"

void init_ops(py::module_& m) {
    // SYCL-only ops (test_esimd, test_dpas, rms) omitted.
}
