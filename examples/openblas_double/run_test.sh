#!/usr/bin/env bash
# run_test.sh — test OpenBLAS dgemm_ and dtrsm_ with linalg-tester.
#
# Usage:
#   cd examples/openblas_double
#   ./run_test.sh [--openblas <path/to/libopenblas.so>]
#
# The script:
#   1. Builds double_conv.so (the double <-> MPFR conversion library).
#   2. Locates libopenblas.so (or uses the path provided via --openblas).
#   3. Runs gemm_tester against dgemm_ for all (transa, transb) combinations.
#   4. Runs trsm_tester against dtrsm_ for all (side, uplo, trans, diag) combinations.
#
# Prerequisites:
#   - linalg-tester must be built (run `make` in the repo root first).
#   - libmpfr-dev, libgmp-dev, and libopenblas-dev must be installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
OPENBLAS_LIB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --openblas)
            OPENBLAS_LIB="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate OpenBLAS shared library
# ---------------------------------------------------------------------------
if [[ -z "$OPENBLAS_LIB" ]]; then
    for candidate in \
        /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so \
        /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0 \
        /usr/lib/x86_64-linux-gnu/libopenblas.so \
        /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
        /usr/lib/libopenblas.so \
        /usr/lib/libopenblas.so.0 \
        /usr/local/lib/libopenblas.so \
        /usr/local/lib/libopenblas.so.0; do
        if [[ -f "$candidate" ]]; then
            OPENBLAS_LIB="$candidate"
            break
        fi
    done

    # Fall back to ldconfig
    if [[ -z "$OPENBLAS_LIB" ]]; then
        OPENBLAS_LIB=$(ldconfig -p 2>/dev/null \
            | awk '/libopenblas\.so/{print $NF}' | head -1)
    fi
fi

if [[ -z "$OPENBLAS_LIB" || ! -f "$OPENBLAS_LIB" ]]; then
    echo "ERROR: Could not locate libopenblas.so." >&2
    echo "Install libopenblas-dev or pass --openblas <path>." >&2
    exit 1
fi

echo "Using OpenBLAS: ${OPENBLAS_LIB}"

# ---------------------------------------------------------------------------
# Build the double_conv shared library
# ---------------------------------------------------------------------------
CONV_SRC="${SCRIPT_DIR}/double_conv.c"
CONV_LIB="${SCRIPT_DIR}/double_conv.so"

echo "Building double_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Provide libgfortran.so.5 stubs when the real library is not installed.
# OpenBLAS has a hard NEEDED entry for libgfortran.so.5 in its ELF header;
# the dynamic linker must open the file (not just resolve symbols) before
# dlopen() can succeed.  We satisfy this by compiling the stub source into
# a file named libgfortran.so.5 and prepending its directory to
# LD_LIBRARY_PATH so the linker finds it before the system library path.
# ---------------------------------------------------------------------------
STUBS_SRC="${SCRIPT_DIR}/gfortran_stubs.c"
STUBS_DIR="${SCRIPT_DIR}/stubs"
STUBS_LIB="${STUBS_DIR}/libgfortran.so.5"

if ! ldconfig -p 2>/dev/null | grep -q "libgfortran\.so\.5"; then
    echo "libgfortran.so.5 not found — building stub library..."
    mkdir -p "${STUBS_DIR}"
    gcc -O0 -shared -fPIC -Wl,-soname,libgfortran.so.5 \
        -o "${STUBS_LIB}" "${STUBS_SRC}"
    echo "  -> ${STUBS_LIB}"
    export LD_LIBRARY_PATH="${STUBS_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ---------------------------------------------------------------------------
# Locate tester binaries
# ---------------------------------------------------------------------------
GEMM_TESTER="${REPO_ROOT}/gemm_tester"
TRSM_TESTER="${REPO_ROOT}/trsm_tester"

for bin in "${GEMM_TESTER}" "${TRSM_TESTER}"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found. Run 'make' in the repo root first." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# GEMM test: dgemm_ — all (transa, transb) combinations
# ---------------------------------------------------------------------------
echo ""
echo "=== GEMM (dgemm_) — double precision, m=n=k=64 ==="
"${GEMM_TESTER}" \
    --lib      "${OPENBLAS_LIB}" \
    --gemm-sym dgemm_ \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --m 64 --n 64 --k 64 \
    --seed 42 \
    --prec 256

# ---------------------------------------------------------------------------
# TRSM test: dtrsm_ — all (side, uplo, trans, diag) combinations
# ---------------------------------------------------------------------------
echo ""
echo "=== TRSM (dtrsm_) — double precision, m=n=64 ==="
"${TRSM_TESTER}" \
    --lib      "${OPENBLAS_LIB}" \
    --trsm-sym dtrsm_ \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --m 64 --n 64 \
    --seed 42 \
    --prec 256

echo ""
echo "Done."
