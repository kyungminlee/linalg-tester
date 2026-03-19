#!/usr/bin/env bash
# run_test.sh — test a sparse solver with quad-precision (IEEE binary128) scalars.
#
# Quad-precision format:
#   Each scalar is a 16-byte IEEE 754 binary128 value (gfortran REAL(16)).
#   Effective mantissa precision = 113 bits; MPFR reference uses --prec 256.
#   sizeof(quad) = 16 bytes  →  --typesize 16.
#
# By default this builds quad_solve.so, a Fortran LU solver in REAL(16),
# and tests it with mumps_tester.
#
# Usage:
#   cd examples/quad_mumps
#   ./run_test.sh                              # uses built-in quad solver
#   ./run_test.sh --solver my_quad_solver.so   # uses custom solver
#
# Prerequisites:
#   - linalg-tester must be built (cmake + make in the repo root).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - gfortran must be installed (for building quad_solve.so).

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
OPT_PREC=256     # well above the 113-bit quad mantissa

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
# Build quad_conv.so (reuse source from reference_quad example)
# ---------------------------------------------------------------------------
QUAD_CONV_SRC="${REPO_ROOT}/examples/reference_quad/quad_conv.c"
CONV_LIB="${SCRIPT_DIR}/quad_conv.so"

echo "Building quad_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${QUAD_CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Build solver wrapper (Fortran quad LU by default)
# ---------------------------------------------------------------------------
if [[ -z "$SOLVER_LIB" ]]; then
    SOLVER_SRC="${SCRIPT_DIR}/quad_solve.f90"
    SOLVER_LIB="${SCRIPT_DIR}/quad_solve.so"
    echo "Building quad_solve.so..."
    gfortran -O2 -shared -fPIC -o "${SOLVER_LIB}" "${SOLVER_SRC}"
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
echo "=== Sparse solver test: quad precision, unsymmetric, n=${OPT_N} ==="
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
echo "=== Sparse solver test: quad precision, SPD, n=${OPT_N} ==="
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
echo "=== Sparse solver test: quad precision, general symmetric, n=${OPT_N} ==="
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
