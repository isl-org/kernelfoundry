#!/bin/bash
set -e

# Provides icpx for SYCL compilation, vtune, and the xptifw library unitrace links against.
source /opt/intel/oneapi/setvars.sh --force > /dev/null

# setvars.sh prepends its own directories, so restore the venv as the active interpreter.
export PATH="/opt/venv/bin:$PATH"

exec "$@"
