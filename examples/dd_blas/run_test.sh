#!/usr/bin/env bash
# run_test.sh — test libddblas.so GEMM and TRSM with linalg-tester.
#
# Usage:
#   cd examples/dd_blas
#   ./run_test.sh [--lib <path/to/libddblas.so>]
#                 [--gemm-sym <symbol>] [--trsm-sym <symbol>]
#                 [--m M] [--n N] [--k K] [--seed S] [--prec P]
#
# The script:
#   1. Builds dd_conv.so (the double-double <-> MPFR conversion library).
#   2. Locates libddblas.so (or uses the path provided via --lib).
#   3. Runs gemm_tester against the GEMM symbol for all (transa, transb) combos.
#   4. Runs trsm_tester against the TRSM symbol for all (side, uplo, trans, diag) combos.
#
# Prerequisites:
#   - linalg-tester must be built (run `make` in the repo root first).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - libddblas.so must be present on the filesystem (not included in this repo).
#
# Symbol names:
#   The default symbols (dd_gemm_ / dd_trsm_) follow a common Fortran-style
#   convention.  If your library uses different names, override with:
#     --gemm-sym mygemm --trsm-sym mytrsm
#   To list exported symbols in your library run:
#     nm -D /path/to/libddblas.so | grep -i gemm
#
# Double-double format:
#   Each scalar is stored as two consecutive IEEE 754 doubles (hi, lo),
#   representing the exact value hi + lo with |lo| < ulp(hi)/2.
#   sizeof(dd) = 16 bytes  →  --typesize 16.
#   Effective mantissa precision ≈ 106 bits; the MPFR reference uses --prec 512.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DDBLAS_LIB=""
GEMM_SYM="dd_gemm_"
TRSM_SYM="dd_trsm_"
OPT_M=64
OPT_N=64
OPT_K=64
OPT_SEED=42
OPT_PREC=512   # well above the ~106-bit DD mantissa

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lib)       DDBLAS_LIB="$2";  shift 2 ;;
        --gemm-sym)  GEMM_SYM="$2";    shift 2 ;;
        --trsm-sym)  TRSM_SYM="$2";    shift 2 ;;
        --m)         OPT_M="$2";        shift 2 ;;
        --n)         OPT_N="$2";        shift 2 ;;
        --k)         OPT_K="$2";        shift 2 ;;
        --seed)      OPT_SEED="$2";     shift 2 ;;
        --prec)      OPT_PREC="$2";     shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--lib <path>] [--gemm-sym <sym>] [--trsm-sym <sym>]" >&2
            echo "          [--m M] [--n N] [--k K] [--seed S] [--prec P]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate libddblas.so
# ---------------------------------------------------------------------------
if [[ -z "$DDBLAS_LIB" ]]; then
    for candidate in \
        ./libddblas.so \
        "${SCRIPT_DIR}/libddblas.so" \
        /usr/local/lib/libddblas.so \
        /usr/lib/libddblas.so; do
        if [[ -f "$candidate" ]]; then
            DDBLAS_LIB="$candidate"
            break
        fi
    done

    if [[ -z "$DDBLAS_LIB" ]]; then
        DDBLAS_LIB=$(ldconfig -p 2>/dev/null \
            | awk '/libddblas\.so/{print $NF}' | head -1)
    fi
fi

if [[ -z "$DDBLAS_LIB" || ! -f "$DDBLAS_LIB" ]]; then
    echo "ERROR: Could not locate libddblas.so." >&2
    echo "Pass --lib <path/to/libddblas.so>." >&2
    exit 1
fi

echo "Using libddblas: ${DDBLAS_LIB}"

# ---------------------------------------------------------------------------
# Build dd_conv.so
# ---------------------------------------------------------------------------
CONV_SRC="${SCRIPT_DIR}/dd_conv.c"
CONV_LIB="${SCRIPT_DIR}/dd_conv.so"

echo "Building dd_conv.so..."
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
# GEMM test — all (transa, transb) combinations
# ---------------------------------------------------------------------------
echo ""
echo "=== GEMM (${GEMM_SYM}) — double-double, m=${OPT_M} n=${OPT_N} k=${OPT_K} ==="
"${GEMM_TESTER}" \
    --lib      "${DDBLAS_LIB}" \
    --gemm-sym "${GEMM_SYM}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --m "${OPT_M}" --n "${OPT_N}" --k "${OPT_K}" \
    --seed     "${OPT_SEED}" \
    --prec     "${OPT_PREC}"

# ---------------------------------------------------------------------------
# TRSM test — all (side, uplo, trans, diag) combinations
# ---------------------------------------------------------------------------
echo ""
echo "=== TRSM (${TRSM_SYM}) — double-double, m=${OPT_M} n=${OPT_N} ==="
"${TRSM_TESTER}" \
    --lib      "${DDBLAS_LIB}" \
    --trsm-sym "${TRSM_SYM}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 16 \
    --m "${OPT_M}" --n "${OPT_N}" \
    --seed     "${OPT_SEED}" \
    --prec     "${OPT_PREC}"

echo ""
echo "Done."
