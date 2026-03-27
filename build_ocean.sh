#!/bin/bash
# build_ocean.sh -- Build PufferFireRed standalone C binary and Python binding
#
# Usage:
#   ./build_ocean.sh           # build Python binding (for training)
#   ./build_ocean.sh play      # build SDL play binary (for human playtesting)
#   ./build_ocean.sh clean     # remove build artifacts
#
# The Python binding uses PufferLib env_binding.h for vectorized envs.
# The play binary is a standalone C executable with SDL for human input.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATIVE_SRC="${SCRIPT_DIR}/../pokefirered-native/src"
PUFFERLIB_OCEAN="${SCRIPT_DIR}/../pufferlib/pufferlib/ocean"
PYTHON_INC="$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")"
NUMPY_INC="$(python3 -c "import numpy; print(numpy.get_include())")"
EXT_SUFFIX="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"

CC="${CC:-cc}"
CFLAGS="-O2 -Wall -Wextra -Wno-unused-parameter"

build_binding() {
    echo "Building Python binding..."
    ${CC} -shared -fPIC ${CFLAGS} \
        "${SCRIPT_DIR}/binding.c" \
        "${NATIVE_SRC}/pfr_so_instance.c" \
        -I"${SCRIPT_DIR}" \
        -I"${NATIVE_SRC}" \
        -I"${PUFFERLIB_OCEAN}" \
        -I"${PYTHON_INC}" \
        -I"${NUMPY_INC}" \
        -ldl -fopenmp -lm \
        -o "${SCRIPT_DIR}/binding${EXT_SUFFIX}"
    echo "Built: binding${EXT_SUFFIX}"
}

build_play() {
    echo "Building standalone play binary..."
    echo "TODO: SDL play binary not yet implemented in pokefirered_puffer/"
    echo "Use pokefirered-native/build/pfr_play for now."
    exit 1
}

clean() {
    echo "Cleaning build artifacts..."
    rm -f "${SCRIPT_DIR}"/binding*.so
    rm -rf "${SCRIPT_DIR}"/__pycache__
    echo "Clean."
}

case "${1:-binding}" in
    binding) build_binding ;;
    play)    build_play ;;
    clean)   clean ;;
    *)
        echo "Usage: $0 [binding|play|clean]"
        exit 1
        ;;
esac
