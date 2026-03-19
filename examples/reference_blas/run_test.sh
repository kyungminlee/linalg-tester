#!/usr/bin/env bash
# run_test.sh — test Netlib reference BLAS dgemm_ and dtrsm_ with linalg-tester.
#
# Uses the libblas.so.3 shipped in the Ubuntu/Debian libblas3 package, which
# is the Netlib Fortran-77 reference implementation built from the lapack
# source.  It lives in /usr/lib/x86_64-linux-gnu/blas/ (separate from the
# system BLAS alternatives symlink) and depends only on libc6 and libgcc-s1.
#
# Usage:
#   cd examples/reference_blas
#   ./run_test.sh [--blas <path/to/libblas.so.3>]
#
# The script:
#   1. Builds double_conv.so (the double <-> MPFR conversion library).
#   2. Locates libblas.so.3 from the libblas3 package (or uses --blas).
#   3. Runs gemm_tester against dgemm_ for all (transa, transb) combinations.
#   4. Runs trsm_tester against dtrsm_ for all (side, uplo, trans, diag) combos.
#
# Prerequisites:
#   - linalg-tester must be built (run `make` in the repo root first).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - libblas3 must be installed:  apt install libblas3

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
# Locate libblas.so.3 from the libblas3 package.
# The libblas3 package installs the library to the blas/ subdirectory,
# distinct from the update-alternatives-managed symlink in the parent dir.
# ---------------------------------------------------------------------------
if [[ -z "$BLAS_LIB" ]]; then
    for candidate in \
        /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 \
        /usr/lib/aarch64-linux-gnu/blas/libblas.so.3 \
        /usr/lib/blas/libblas.so.3; do
        if [[ -f "$candidate" ]]; then
            BLAS_LIB="$candidate"
            break
        fi
    done
fi

if [[ -z "$BLAS_LIB" || ! -f "$BLAS_LIB" ]]; then
    echo "ERROR: Could not locate libblas.so.3 from the libblas3 package." >&2
    echo "Install it with:  apt install libblas3" >&2
    echo "Or pass an explicit path:  $0 --blas <path/to/libblas.so.3>" >&2
    exit 1
fi

echo "Using BLAS: ${BLAS_LIB}"

# ---------------------------------------------------------------------------
# Build double_conv.so
# ---------------------------------------------------------------------------
CONV_SRC="${SCRIPT_DIR}/double_conv.c"
CONV_LIB="${SCRIPT_DIR}/double_conv.so"

echo "Building double_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

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
