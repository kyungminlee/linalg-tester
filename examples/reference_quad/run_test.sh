#!/usr/bin/env bash
# run_test.sh — test reference quad-precision GEMM and TRSM with linalg-tester.
#
# Usage:
#   cd examples/reference_quad
#   ./run_test.sh [--m M] [--n N] [--k K] [--seed S] [--prec P]
#
# Prerequisites:
#   - linalg-tester must be built (run `make` in the repo root first).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - gfortran must be installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OPT_M=32
OPT_N=32
OPT_K=32
OPT_SEED=42
OPT_PREC=256   # Quad precision is ~113 bits; 256 is plenty.

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --m)         OPT_M="$2";        shift 2 ;;
        --n)         OPT_N="$2";        shift 2 ;;
        --k)         OPT_K="$2";        shift 2 ;;
        --seed)      OPT_SEED="$2";     shift 2 ;;
        --prec)      OPT_PREC="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--m M] [--n N] [--k K] [--seed S] [--prec P]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Build the quad-precision BLAS library
# ---------------------------------------------------------------------------
QUADBLAS_LIB="${SCRIPT_DIR}/libquadblas.so"
UTILS_SRC="${SCRIPT_DIR}/utils.f90"

echo "Creating utils.f90..."
cat > "${UTILS_SRC}" <<EOF
LOGICAL FUNCTION LSAME( CA, CB )
  CHARACTER CA, CB
  CHARACTER(1) UCA, UCB
  UCA = CA
  UCB = CB
  IF( UCA >= 'a' .AND. UCA <= 'z' ) UCA = CHAR( ICHAR( UCA ) - 32 )
  IF( UCB >= 'a' .AND. UCB <= 'z' ) UCB = CHAR( ICHAR( UCB ) - 32 )
  LSAME = UCA == UCB
END FUNCTION

SUBROUTINE XERBLA( SRNAME, INFO )
  CHARACTER(*) SRNAME
  INTEGER INFO
  WRITE(*,*) ' ** On entry to ', SRNAME, ' parameter number ', INFO, ' had an illegal value'
END SUBROUTINE
EOF

echo "Building libquadblas.so from Fortran sources..."
gfortran -O2 -shared -fPIC -o "${QUADBLAS_LIB}" \
    "${SCRIPT_DIR}/qgemm.f90" \
    "${SCRIPT_DIR}/qtrsm.f90" \
    "${UTILS_SRC}"
echo "  -> ${QUADBLAS_LIB}"

# ---------------------------------------------------------------------------
# Build quad_conv.so
# ---------------------------------------------------------------------------
CONV_SRC="${SCRIPT_DIR}/quad_conv.c"
CONV_LIB="${SCRIPT_DIR}/quad_conv.so"

echo "Building quad_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Locate tester binaries
# ---------------------------------------------------------------------------
GEMM_TESTER="${REPO_ROOT}/build/gemm_tester"
TRSM_TESTER="${REPO_ROOT}/build/trsm_tester"

for bin in "${GEMM_TESTER}" "${TRSM_TESTER}"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: $bin not found. Run 'make' in the repo root first." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# GEMM test: qgemm_ — all (transa, transb) combinations
# ---------------------------------------------------------------------------
echo ""
echo "=== GEMM (qgemm_) — quad precision, m=${OPT_M} n=${OPT_N} k=${OPT_K} ==="
"${GEMM_TESTER}" \
    --lib      "${QUADBLAS_LIB}" \
    --gemm-sym qgemm_ \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --m "${OPT_M}" --n "${OPT_N}" --k "${OPT_K}" \
    --seed     "${OPT_SEED}" \
    --prec     "${OPT_PREC}"

# ---------------------------------------------------------------------------
# TRSM test: qtrsm_ — all (side, uplo, trans, diag) combinations
# ---------------------------------------------------------------------------
echo ""
echo "=== TRSM (qtrsm_) — quad precision, m=${OPT_M} n=${OPT_N} ==="
"${TRSM_TESTER}" \
    --lib      "${QUADBLAS_LIB}" \
    --trsm-sym qtrsm_ \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --m "${OPT_M}" --n "${OPT_N}" \
    --seed     "${OPT_SEED}" \
    --prec     "${OPT_PREC}"

echo ""
echo "Done."
