#!/usr/bin/env bash
# run_test.sh — test a sparse solver with linalg-tester's mumps_tester.
#
# By default this uses lapack_solve.so (a thin dgesv_ wrapper) so that
# you can validate the tester without installing MUMPS.
#
# Usage:
#   cd examples/reference_mumps
#   ./run_test.sh                          # uses LAPACK dgesv_ wrapper
#   ./run_test.sh --solver mumps_solve.so  # uses real MUMPS wrapper
#
# Prerequisites:
#   - linalg-tester must be built (cmake + make in the repo root).
#   - libmpfr-dev, libgmp-dev must be installed.
#   - For the LAPACK solver:  apt install liblapack3 libblas3
#   - For the MUMPS solver:   apt install libmumps-seq-dev (or libmumps-dev)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SOLVER_LIB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)
            SOLVER_LIB="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--solver <path/to/solver.so>]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Build double_conv.so (reuse from reference_blas or build locally)
# ---------------------------------------------------------------------------
CONV_SRC="${REPO_ROOT}/examples/reference_blas/double_conv.c"
CONV_LIB="${SCRIPT_DIR}/double_conv.so"

echo "Building double_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Build solver wrapper (LAPACK by default)
# ---------------------------------------------------------------------------
if [[ -z "$SOLVER_LIB" ]]; then
    SOLVER_SRC="${SCRIPT_DIR}/lapack_solve.c"
    SOLVER_LIB="${SCRIPT_DIR}/lapack_solve.so"
    echo "Building lapack_solve.so..."
    # Find liblapack.so.3 and libblas.so.3 (prefer alternatives symlink)
    LAPACK_SO=""
    for p in /usr/lib/x86_64-linux-gnu/liblapack.so.3 \
             /usr/lib/aarch64-linux-gnu/liblapack.so.3 \
             /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3; do
        [[ -f "$p" ]] && LAPACK_SO="$p" && break
    done
    BLAS_SO=""
    for p in /usr/lib/x86_64-linux-gnu/libblas.so.3 \
             /usr/lib/aarch64-linux-gnu/libblas.so.3 \
             /usr/lib/x86_64-linux-gnu/blas/libblas.so.3; do
        [[ -f "$p" ]] && BLAS_SO="$p" && break
    done
    if [[ -z "$LAPACK_SO" || -z "$BLAS_SO" ]]; then
        echo "ERROR: Could not find liblapack.so.3 or libblas.so.3." >&2
        echo "Install with:  apt install liblapack3 libblas3" >&2
        exit 1
    fi
    gcc -O2 -shared -fPIC -o "${SOLVER_LIB}" "${SOLVER_SRC}" \
        "${LAPACK_SO}" "${BLAS_SO}" -lm
    echo "  -> ${SOLVER_LIB}"
fi

# ---------------------------------------------------------------------------
# Locate mumps_tester binary
# ---------------------------------------------------------------------------
MUMPS_TESTER="${REPO_ROOT}/build/mumps_tester"
if [[ ! -x "$MUMPS_TESTER" ]]; then
    echo "ERROR: ${MUMPS_TESTER} not found. Run cmake + make in the repo root first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo ""
echo "=== Sparse solver test: unsymmetric, n=64, density=0.1 ==="
"${MUMPS_TESTER}" \
    --lib      "${SOLVER_LIB}" \
    --solve-sym sparse_solve \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --n 64 \
    --sym 0 \
    --density 0.1 \
    --seed 42 \
    --prec 256

echo ""
echo "=== Sparse solver test: SPD, n=64, density=0.15 ==="
"${MUMPS_TESTER}" \
    --lib      "${SOLVER_LIB}" \
    --solve-sym sparse_solve \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --n 64 \
    --sym 1 \
    --density 0.15 \
    --seed 42 \
    --prec 256

echo ""
echo "=== Sparse solver test: general symmetric, n=64, density=0.1 ==="
"${MUMPS_TESTER}" \
    --lib      "${SOLVER_LIB}" \
    --solve-sym sparse_solve \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --n 64 \
    --sym 2 \
    --density 0.1 \
    --seed 42 \
    --prec 256

echo ""
echo "Done."
