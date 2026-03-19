#!/usr/bin/env bash
# run_test.sh — test MUMPS (sequential) sparse solver with linalg-tester.
#
# Uses dmumps_c() via DMUMPS_STRUC_C directly — no wrapper library needed.
#
# Usage:
#   cd examples/reference_mumps
#   ./run_test.sh
#   ./run_test.sh --mumps /path/to/libdmumps_seq.so
#
# Prerequisites:
#   - linalg-tester must be built (cmake + make in the repo root).
#   - libmpfr-dev and libgmp-dev must be installed.
#   - MUMPS sequential must be installed:  apt install libmumps-seq-dev

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MUMPS_LIB=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mumps)
            MUMPS_LIB="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--mumps <path/to/libdmumps_seq.so>]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate libdmumps_seq.so
# ---------------------------------------------------------------------------
if [[ -z "$MUMPS_LIB" ]]; then
    for candidate in \
        /usr/lib/x86_64-linux-gnu/libdmumps_seq.so \
        /usr/lib/aarch64-linux-gnu/libdmumps_seq.so \
        /usr/lib/x86_64-linux-gnu/libdmumps_seq-5.6.so \
        /usr/lib/aarch64-linux-gnu/libdmumps_seq-5.6.so \
        /usr/local/lib/libdmumps_seq.so; do
        if [[ -f "$candidate" ]]; then
            MUMPS_LIB="$candidate"
            break
        fi
    done
fi

if [[ -z "$MUMPS_LIB" || ! -f "$MUMPS_LIB" ]]; then
    echo "ERROR: Could not locate libdmumps_seq.so." >&2
    echo "Install with:  apt install libmumps-seq-dev" >&2
    echo "Or pass an explicit path:  $0 --mumps <path/to/libdmumps_seq.so>" >&2
    exit 1
fi

echo "Using MUMPS: ${MUMPS_LIB}"

# ---------------------------------------------------------------------------
# Build double_conv.so
# ---------------------------------------------------------------------------
CONV_SRC="${REPO_ROOT}/examples/reference_blas/double_conv.c"
CONV_LIB="${SCRIPT_DIR}/double_conv.so"

echo "Building double_conv.so..."
gcc -O2 -shared -fPIC -o "${CONV_LIB}" "${CONV_SRC}" -lmpfr
echo "  -> ${CONV_LIB}"

# ---------------------------------------------------------------------------
# Locate mumps_tester binary
# ---------------------------------------------------------------------------
MUMPS_TESTER="${REPO_ROOT}/build/mumps_tester"
if [[ ! -x "$MUMPS_TESTER" ]]; then
    echo "ERROR: ${MUMPS_TESTER} not found. Run cmake + make in the repo root first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run tests — dmumps_c called directly via DMUMPS_STRUC_C
# ---------------------------------------------------------------------------
echo ""
echo "=== Sparse solver test: unsymmetric, n=64, density=0.1 ==="
"${MUMPS_TESTER}" \
    --lib      "${MUMPS_LIB}" \
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
    --lib      "${MUMPS_LIB}" \
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
    --lib      "${MUMPS_LIB}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --n 64 \
    --sym 2 \
    --density 0.1 \
    --seed 42 \
    --prec 256

echo ""
echo "Done."
