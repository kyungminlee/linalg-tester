#!/usr/bin/env bash
# run_test.sh — test a sparse solver with double-double precision scalars.
#
# Double-double format:
#   Each scalar is stored as two consecutive IEEE 754 doubles (hi, lo),
#   representing the exact value hi + lo with |lo| < ulp(hi)/2.
#   sizeof(dd) = 16 bytes  →  --typesize 16.
#   Effective mantissa precision ≈ 106 bits; MPFR reference uses --prec 512.
#
# By default this uses dd_lapack_solve.so which converts to double and
# solves with LAPACK dgesv_.  The residual norms will be at double-
# precision level (~1e-16).  For dd-level accuracy, supply your own
# solver with --solver.
#
# Usage:
#   cd examples/dd_mumps
#   ./run_test.sh                              # uses LAPACK wrapper
#   ./run_test.sh --solver my_dd_solver.so     # uses custom dd solver
#
# Prerequisites:
#   - linalg-tester must be built (cmake + make in the repo root).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - liblapack3 and libblas3 must be installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SOLVER_LIB=""
SOLVE_SYM="sparse_solve"
OPT_N=64
OPT_DENSITY=0.1
OPT_SEED=42
OPT_PREC=512     # well above the ~106-bit DD mantissa

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver)    SOLVER_LIB="$2";  shift 2 ;;
        --solve-sym) SOLVE_SYM="$2";   shift 2 ;;
        --n)         OPT_N="$2";       shift 2 ;;
        --density)   OPT_DENSITY="$2"; shift 2 ;;
        --seed)      OPT_SEED="$2";    shift 2 ;;
        --prec)      OPT_PREC="$2";    shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--solver <path>] [--solve-sym <sym>]" >&2
            echo "          [--n N] [--density D] [--seed S] [--prec P]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Build dd_conv.so (reuse source from dd_blas example)
# ---------------------------------------------------------------------------
DD_CONV_SRC="${REPO_ROOT}/examples/dd_blas/dd_conv.c"
CONV_LIB="${SCRIPT_DIR}/dd_conv.so"

echo "Building dd_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${DD_CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Build solver wrapper (LAPACK by default)
# ---------------------------------------------------------------------------
if [[ -z "$SOLVER_LIB" ]]; then
    SOLVER_SRC="${SCRIPT_DIR}/dd_lapack_solve.c"
    SOLVER_LIB="${SCRIPT_DIR}/dd_lapack_solve.so"
    echo "Building dd_lapack_solve.so..."
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
echo "=== Sparse solver test: double-double, unsymmetric, n=${OPT_N} ==="
"${MUMPS_TESTER}" \
    --lib      "${SOLVER_LIB}" \
    --solve-sym "${SOLVE_SYM}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --n "${OPT_N}" \
    --sym 0 \
    --density "${OPT_DENSITY}" \
    --seed "${OPT_SEED}" \
    --prec "${OPT_PREC}"

echo ""
echo "=== Sparse solver test: double-double, SPD, n=${OPT_N} ==="
"${MUMPS_TESTER}" \
    --lib      "${SOLVER_LIB}" \
    --solve-sym "${SOLVE_SYM}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --n "${OPT_N}" \
    --sym 1 \
    --density 0.15 \
    --seed "${OPT_SEED}" \
    --prec "${OPT_PREC}"

echo ""
echo "=== Sparse solver test: double-double, general symmetric, n=${OPT_N} ==="
"${MUMPS_TESTER}" \
    --lib      "${SOLVER_LIB}" \
    --solve-sym "${SOLVE_SYM}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --n "${OPT_N}" \
    --sym 2 \
    --density "${OPT_DENSITY}" \
    --seed "${OPT_SEED}" \
    --prec "${OPT_PREC}"

echo ""
echo "Done."
