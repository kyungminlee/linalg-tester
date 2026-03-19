#!/usr/bin/env bash
# run_test.sh — test reference/system BLAS dgemm_ and dtrsm_ with linalg-tester.
#
# "Reference BLAS" here means the system's libblas.so.3 — typically the Netlib
# reference implementation or whatever BLAS the system's alternatives mechanism
# selects (e.g. OpenBLAS on Ubuntu).  The library name is libblas.so.3 and it
# exports the standard Fortran-77 symbols dgemm_ / dtrsm_.
#
# Usage:
#   cd examples/reference_blas
#   ./run_test.sh [--blas <path/to/libblas.so.3>]
#
# The script:
#   1. Builds double_conv.so (the double <-> MPFR conversion library).
#   2. Locates libblas.so.3 (or uses the path provided via --blas).
#   3. Adds the library's parent directory to LD_LIBRARY_PATH so any
#      transitively-needed libraries (e.g. libopenblas.so.0) are found.
#   4. Builds a stub libgfortran.so.5 if the real one is not installed.
#   5. Runs gemm_tester against dgemm_ for all (transa, transb) combinations.
#   6. Runs trsm_tester against dtrsm_ for all (side, uplo, trans, diag) combos.
#
# Prerequisites:
#   - linalg-tester must be built (run `make` in the repo root first).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - libblas3 (or equivalent) must be installed, providing libblas.so.3.
#     On Ubuntu/Debian: apt install libblas3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
BLAS_LIB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --blas)
            BLAS_LIB="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--blas <path/to/libblas.so.3>]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate libblas.so.3
# ---------------------------------------------------------------------------
if [[ -z "$BLAS_LIB" ]]; then
    for candidate in \
        /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 \
        /usr/lib/x86_64-linux-gnu/libblas.so.3 \
        /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 \
        /usr/lib/libblas.so.3 \
        /usr/local/lib/libblas.so.3; do
        if [[ -f "$candidate" ]]; then
            BLAS_LIB="$candidate"
            break
        fi
    done

    if [[ -z "$BLAS_LIB" ]]; then
        BLAS_LIB=$(ldconfig -p 2>/dev/null \
            | awk '/libblas\.so\.3/{print $NF}' | head -1)
    fi
fi

if [[ -z "$BLAS_LIB" || ! -f "$BLAS_LIB" ]]; then
    echo "ERROR: Could not locate libblas.so.3." >&2
    echo "Install libblas-dev or pass --blas <path>." >&2
    exit 1
fi

echo "Using BLAS: ${BLAS_LIB}"

# ---------------------------------------------------------------------------
# Add the library's directory to LD_LIBRARY_PATH so transitively-needed
# libraries (e.g. libopenblas.so.0) in the same directory are found.
# ---------------------------------------------------------------------------
BLAS_DIR="$(dirname "${BLAS_LIB}")"
export LD_LIBRARY_PATH="${BLAS_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ---------------------------------------------------------------------------
# Build double_conv.so
# ---------------------------------------------------------------------------
CONV_SRC="${SCRIPT_DIR}/double_conv.c"
CONV_LIB="${SCRIPT_DIR}/double_conv.so"

echo "Building double_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Build stub libgfortran.so.5 if not installed.
# libblas.so.3 may depend (directly or transitively) on libgfortran.so.5;
# the dynamic linker must be able to open the file by name.
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
    --lib      "${BLAS_LIB}" \
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
    --lib      "${BLAS_LIB}" \
    --trsm-sym dtrsm_ \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --m 64 --n 64 \
    --seed 42 \
    --prec 256

echo ""
echo "Done."
