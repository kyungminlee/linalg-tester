#!/usr/bin/env bash
# run_regression.sh -- Build and run linalg_tester against OpenBLAS,
# then check that all BLAS errors and LAPACK residuals are within bounds.
#
# Usage (from repo root):
#   bash test/run_regression.sh [--openblas /path/to/libopenblas.so]
#
# Prerequisites: cmake, make, cc, pkg-config, libmpfr-dev, libgmp-dev,
#                and an OpenBLAS shared library.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
CONV_SRC="${REPO_ROOT}/examples/openblas_double/double_conv.c"
STUBS_SRC="${REPO_ROOT}/examples/openblas_double/gfortran_stubs.c"

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
OS="$(uname -s)"
case "$OS" in
    Darwin) SHLIB_EXT="dylib" ;;
    *)      SHLIB_EXT="so" ;;
esac

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
    candidates=()

    if [[ "$OS" == "Darwin" ]]; then
        # Homebrew on Apple Silicon
        candidates+=(
            /opt/homebrew/opt/openblas/lib/libopenblas.dylib
            /opt/homebrew/Cellar/openblas/*/lib/libopenblas.dylib
        )
        # Homebrew on Intel Mac
        candidates+=(
            /usr/local/opt/openblas/lib/libopenblas.dylib
            /usr/local/Cellar/openblas/*/lib/libopenblas.dylib
        )
    else
        # Common Linux paths
        candidates+=(
            /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so
            /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0
            /usr/lib/x86_64-linux-gnu/libopenblas.so
            /usr/lib/x86_64-linux-gnu/libopenblas.so.0
            /usr/lib/aarch64-linux-gnu/libopenblas.so
            /usr/lib/aarch64-linux-gnu/libopenblas.so.0
            /usr/lib/libopenblas.so
            /usr/lib/libopenblas.so.0
            /usr/local/lib/libopenblas.so
            /usr/local/lib/libopenblas.so.0
        )
    fi

    # Glob expansion may produce multiple matches; iterate over them
    for candidate in "${candidates[@]}"; do
        # shellcheck disable=SC2086
        for resolved in $candidate; do
            if [[ -f "$resolved" ]]; then
                OPENBLAS_LIB="$resolved"
                break 2
            fi
        done
    done

    # Fall back to ldconfig on Linux
    if [[ -z "$OPENBLAS_LIB" && "$OS" != "Darwin" ]]; then
        OPENBLAS_LIB=$(ldconfig -p 2>/dev/null \
            | awk '/libopenblas\.so /{print $NF}' | head -1) || true
    fi
fi

if [[ -z "$OPENBLAS_LIB" || ! -f "$OPENBLAS_LIB" ]]; then
    echo "ERROR: Could not locate OpenBLAS shared library." >&2
    echo "Install OpenBLAS or pass --openblas <path>." >&2
    exit 1
fi

echo "Using OpenBLAS: ${OPENBLAS_LIB}"

# ---------------------------------------------------------------------------
# Step 1: Build linalg_tester
# ---------------------------------------------------------------------------
echo ""
echo "--- Building linalg_tester ---"
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

TESTER="${BUILD_DIR}/linalg_tester"
if [[ ! -x "$TESTER" ]]; then
    echo "ERROR: ${TESTER} not found after build." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Use CMake-built conversion libraries
# ---------------------------------------------------------------------------
echo ""
echo "--- Using CMake-built conversion libraries ---"
CONV_LIB="${BUILD_DIR}/libdouble_conv.${SHLIB_EXT}"
COMPLEX_CONV_LIB="${BUILD_DIR}/libcomplex_double_conv.${SHLIB_EXT}"

if [[ ! -f "$CONV_LIB" ]]; then
    echo "ERROR: ${CONV_LIB} not found after build." >&2
    exit 1
fi
echo "  -> ${CONV_LIB}"
echo "  -> ${COMPLEX_CONV_LIB}"

# ---------------------------------------------------------------------------
# Step 2b: Build gfortran stubs if needed (Linux only)
# ---------------------------------------------------------------------------
if [[ "$OS" != "Darwin" ]]; then
    if ! ldconfig -p 2>/dev/null | grep -q "libgfortran\.so\.5"; then
        echo "libgfortran.so.5 not found -- building stub library..."
        STUBS_DIR="${REPO_ROOT}/test/stubs"
        STUBS_LIB="${STUBS_DIR}/libgfortran.so.5"
        mkdir -p "${STUBS_DIR}"
        cc -O0 -shared -fPIC -Wl,-soname,libgfortran.so.5 \
            -o "${STUBS_LIB}" "${STUBS_SRC}"
        echo "  -> ${STUBS_LIB}"
        export LD_LIBRARY_PATH="${STUBS_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: Run linalg_tester (small matrices for speed)
# ---------------------------------------------------------------------------
echo ""
echo "--- Running linalg_tester (all routines, m=16 n=16 k=4) ---"
OUTPUT_FILE="${REPO_ROOT}/test/regression_output.csv"

"${TESTER}" \
    --routine all \
    --sym-prefix d \
    --lib "${OPENBLAS_LIB}" \
    --conv-lib "${CONV_LIB}" \
    --typesize 8 \
    --m 16 --n 16 --k 4 \
    --format csv \
    2>&1 | tee "${OUTPUT_FILE}"

# ---------------------------------------------------------------------------
# Step 3b: Run complex BLAS tests
# ---------------------------------------------------------------------------
if [[ -f "$COMPLEX_CONV_LIB" ]]; then
    echo ""
    echo "--- Running complex BLAS tests (cblas3/cblas2/cblas1, m=8 n=8 k=4) ---"
    COMPLEX_OUTPUT="${REPO_ROOT}/test/complex_regression_output.csv"

    # Determine complex return ABI
    COMPLEX_ABI="hidden"
    case "$OS" in
        Darwin)
            ARCH=$(uname -m)
            if [[ "$ARCH" == "arm64" || "$ARCH" == "x86_64" ]]; then
                COMPLEX_ABI="register"
            fi
            ;;
    esac

    for cat in cblas3 cblas2 cblas1; do
        "${TESTER}" \
            --complex \
            --routine "$cat" \
            --sym-prefix z \
            --lib "${OPENBLAS_LIB}" \
            --conv-lib "${COMPLEX_CONV_LIB}" \
            --typesize 16 \
            --m 8 --n 8 --k 4 \
            --complex-return-abi "${COMPLEX_ABI}" \
            --format csv \
            2>&1
    done | tee "${COMPLEX_OUTPUT}"

    # Append complex results to main output for unified checking
    cat "${COMPLEX_OUTPUT}" >> "${OUTPUT_FILE}"
fi

# ---------------------------------------------------------------------------
# Step 3c: Run complex LAPACK tests
# ---------------------------------------------------------------------------
if [[ -f "$COMPLEX_CONV_LIB" ]]; then
    echo ""
    echo "--- Running complex LAPACK tests (clapack_fact/solve/eig/aux, m=8 n=8 k=4) ---"
    CLAPACK_OUTPUT="${REPO_ROOT}/test/clapack_regression_output.csv"

    for cat in clapack_fact clapack_solve clapack_eig clapack_aux; do
        "${TESTER}" \
            --complex \
            --routine "$cat" \
            --sym-prefix z \
            --lib "${OPENBLAS_LIB}" \
            --conv-lib "${COMPLEX_CONV_LIB}" \
            --typesize 16 \
            --m 8 --n 8 --k 4 \
            --complex-return-abi "${COMPLEX_ABI}" \
            --format csv \
            2>&1
    done | tee "${CLAPACK_OUTPUT}"

    # Append complex LAPACK results to main output for unified checking
    cat "${CLAPACK_OUTPUT}" >> "${OUTPUT_FILE}"
fi

# ---------------------------------------------------------------------------
# Step 4: Parse CSV output and check thresholds
# ---------------------------------------------------------------------------
echo ""
echo "--- Checking results ---"

BLAS_MAX_REL_THRESHOLD="1e-10"
LAPACK_RESIDUAL_THRESHOLD="50"

total=0
pass=0
fail=0
fail_details=""

while IFS= read -r line; do
    # Skip blank lines and header-like lines (=== ... ===)
    [[ -z "$line" ]] && continue
    [[ "$line" == "==="* ]] && continue
    # Skip lines that do not look like CSV data (no commas)
    [[ "$line" != *,* ]] && continue

    routine=$(echo "$line" | cut -d',' -f1)

    # Determine if this is a BLAS or LAPACK line.
    # BLAS CSV: routine,params,max_relative,normwise_relative,sentinel_passed,sentinel_corrupted
    # LAPACK CSV: routine,params,residual,orthogonality_or_empty,info
    #
    # Heuristic: LAPACK routines have an integer INFO field as the last column.
    # BLAS routines have "true"/"false" in their 5th field.
    field5=$(echo "$line" | cut -d',' -f5)

    total=$((total + 1))

    if [[ "$field5" == "true" || "$field5" == "false" ]]; then
        # BLAS line
        max_rel=$(echo "$line" | cut -d',' -f3)

        # Check for FAIL / NaN / Inf
        if echo "$max_rel" | grep -iqE 'nan|inf|fail'; then
            fail=$((fail + 1))
            fail_details="${fail_details}  FAIL: ${line}\n"
            continue
        fi

        # Compare max_relative against threshold
        exceeded=$(awk "BEGIN { print ($max_rel > $BLAS_MAX_REL_THRESHOLD) ? 1 : 0 }")
        if [[ "$exceeded" == "1" ]]; then
            fail=$((fail + 1))
            fail_details="${fail_details}  FAIL [max_rel=${max_rel} > ${BLAS_MAX_REL_THRESHOLD}]: ${line}\n"
        else
            pass=$((pass + 1))
        fi
    else
        # LAPACK line
        residual=$(echo "$line" | cut -d',' -f3)
        # INFO is the last field
        info=$(echo "$line" | awk -F',' '{print $NF}')

        # Check for NaN/Inf/FAIL
        if echo "$residual" | grep -iqE 'nan|inf|fail'; then
            fail=$((fail + 1))
            fail_details="${fail_details}  FAIL: ${line}\n"
            continue
        fi

        # Check nonzero INFO
        if [[ -n "$info" ]] && [[ "$info" =~ ^-?[0-9]+$ ]] && [[ "$info" -ne 0 ]]; then
            fail=$((fail + 1))
            fail_details="${fail_details}  FAIL [INFO=${info}]: ${line}\n"
            continue
        fi

        # Compare residual against threshold
        exceeded=$(awk "BEGIN { print ($residual > $LAPACK_RESIDUAL_THRESHOLD) ? 1 : 0 }")
        if [[ "$exceeded" == "1" ]]; then
            fail=$((fail + 1))
            fail_details="${fail_details}  FAIL [residual=${residual} > ${LAPACK_RESIDUAL_THRESHOLD}]: ${line}\n"
        else
            pass=$((pass + 1))
        fi
    fi
done < "${OUTPUT_FILE}"

# ---------------------------------------------------------------------------
# Step 5: Print summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo "  Regression Test Summary"
echo "========================================="
echo "  Total tests:  ${total}"
echo "  Passed:       ${pass}"
echo "  Failed:       ${fail}"
echo "========================================="

if [[ "$fail" -gt 0 ]]; then
    echo ""
    echo "Failures:"
    echo -e "$fail_details"
    echo "RESULT: FAIL"
    exit 1
else
    echo ""
    echo "RESULT: PASS"
    exit 0
fi
