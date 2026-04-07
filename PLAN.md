# Plan: Full BLAS & LAPACK Accuracy Tester

## Goal

Expand linalg-tester from testing 2 routines (GEMM, TRSM) to the complete
BLAS standard (~55 routines across 3 levels, real and complex), and
subsequently all key LAPACK routines (~40+ routines). The tester's core
value proposition is unchanged: load a user-supplied shared library via
`dlopen`, call the target routine, and compare against a multi-precision
MPFR reference computation.

---

## 1. Current State

### What exists

| Component | Files | Description |
|-----------|-------|-------------|
| GEMM tester | `src/gemm_tester.cpp` | Tests all 4 (transa, transb) combos |
| TRSM tester | `src/trsm_tester.cpp` | Tests all 16 (side,uplo,trans,diag) combos |
| MPFR reference | `src/reference.cpp`, `src/reference.h` | MPFR GEMM/TRSM + error metrics |
| Utilities | `src/tester_utils.h` | `gen_random_array`, `gen_triangular_array`, loader helpers |
| CLI | `third_party/CLI11.hpp` | Header-only argument parser |
| Build | `Makefile`, `CMakeLists.txt` | Produces `gemm_tester`, `trsm_tester` |
| Examples | `examples/{openblas_double,reference_blas,dd_blas,reference_quad}/` | Test scripts + conversion libraries |

### What works well (keep)

- **Type-agnostic design** via `void*` + `typesize` + conversion functions.
  Already handles double, quad, double-double.
- **MPFR reference approach** for ground-truth computation.
- **Error metrics** (max element-wise relative + Frobenius-norm relative).
- **dlopen/dlsym** loading with preload support.
- **CLI11** for argument parsing.

### What needs to change

| Problem | Impact |
|---------|--------|
| One binary per routine (`gemm_tester`, `trsm_tester`) | Doesn't scale to 50+ routines; massive build target proliferation |
| Per-routine CLI flag (`--gemm-sym`, `--trsm-sym`) | Each routine invents its own flag name |
| `reference.cpp` bundles reference impls + error metrics + RAII wrappers | Monolithic; grows unbounded as routines are added |
| Only 2 matrix generators (random, triangular) | Need symmetric, banded, packed, positive-definite, etc. |
| Function pointer typedefs in `reference.h` | `reference.h` shouldn't define BLAS ABIs; they belong with each routine |
| Error metrics are matrix-only | Need vector/scalar variants for Level 1/2; need backward-error metrics for LAPACK |
| No complex number support | Required for full BLAS (C/Z prefixes) and LAPACK |
| No CI or automated regression tests | The tool that tests correctness has no automated verification of itself |

---

## 2. Target Architecture

### Single unified binary

Replace `gemm_tester` / `trsm_tester` with one binary: **`linalg_tester`**.

```
linalg_tester --routine gemm \
    --lib libopenblas.so --sym dgemm_ \
    --conv-lib double_conv.so --typesize 8 \
    --m 64 --n 64 --k 64 --seed 42 --prec 256
```

### CLI design

**Core flags** (always present):

| Flag | Description | Default |
|------|-------------|---------|
| `--routine <name>` | Routine to test (e.g., `gemm`, `trsm`, `dot`) | required |
| `--lib <path>` | Shared library containing the implementation | required |
| `--sym <name>` | Symbol name (e.g., `dgemm_`, `dtrsm_`) | required |
| `--conv-lib <path>` | Conversion library (`custom_to_mpfr`/`mpfr_to_custom`) | required |
| `--typesize <n>` | sizeof of the custom scalar type | required |
| `--preload <path>` | Libraries to preload (repeatable) | none |
| `--prec <bits>` | MPFR working precision | 256 |
| `--seed <n>` | Random seed | 42 |

**Dimension/stride flags** (optional, each routine uses what it needs):

| Flag | Description | Default |
|------|-------------|---------|
| `--m <n>` | Matrix rows | 64 |
| `--n <n>` | Matrix columns / vector length | 64 |
| `--k <n>` | Inner dimension (GEMM, SYRK, etc.) | 64 |
| `--kl <n>` | Sub-diagonals (banded routines) | 2 |
| `--ku <n>` | Super-diagonals (banded routines) | 2 |
| `--incx <n>` | Stride for x vector | 1 |
| `--incy <n>` | Stride for y vector | 1 |
| `--ld-pad <n>` | Padding added to all leading dimensions | 0 |

**Output/automation flags**:

| Flag | Description | Default |
|------|-------------|---------|
| `--list` | Print all available routines and exit | |
| `--threshold <val>` | Exit nonzero if any normwise error exceeds this | (none) |
| `--format <fmt>` | Output format: `text`, `json`, `csv` | `text` |

All dimension/stride flags are accepted by every routine. If a routine
does not use a flag (e.g., `--k` for TRSM), the value is silently
ignored. This avoids per-routine CLI specialization while remaining
forward-compatible.

**Batch mode**: `--routine all`, `--routine blas1`, `--routine blas2`,
`--routine blas3` run all routines in a category, auto-deriving symbol
names from a `--sym-prefix` (e.g., `--sym-prefix d` gives `dgemm_`,
`dtrsm_`, etc.). This requires `--sym` to be optional when a prefix
is given.

**`--list` output format**:
```
$ linalg_tester --list
BLAS Level 3:
  gemm     General matrix multiply           C = alpha*op(A)*op(B) + beta*C
  trsm     Triangular solve                  op(A)*X = alpha*B
  symm     Symmetric matrix multiply         C = alpha*A*B + beta*C
  ...
BLAS Level 2:
  gemv     General matrix-vector             y = alpha*op(A)*x + beta*y
  ...
```

`--list` does not require `--lib`, `--sym`, `--conv-lib`, or `--typesize`.

### Directory layout

```
src/
  core/
    tester_ctx.h             TesterCtx, ErrorResult, TestParams, conversion fn types
    mpfr_types.h             MpfrMatrix, MpfrScalar RAII wrappers
    error_metrics.h/.cpp     compute_error_{matrix,vector,scalar,index}; NaN/Inf handling
    generators.h/.cpp        All matrix/vector generators
    loader.h                 preload_libs, load_sym
    report.h/.cpp            Output formatting (text/json/csv), threshold checking
    sentinel.h/.cpp          Out-of-bounds write detection (Section 3.4)

  blas/
    blas_common.h            Shared BLAS helpers (char normalization, Fortran ABI notes)
    level3.h                 Declarations: test_gemm, test_trsm, test_symm, ...
    level3/
      gemm.cpp               fn typedef + test driver + MPFR reference
      symm.cpp
      syrk.cpp
      syr2k.cpp
      trmm.cpp
      trsm.cpp
    level2.h                 Declarations: test_gemv, test_trmv, ...
    level2/
      gemv.cpp
      gbmv.cpp
      ...16 files...
    level1.h                 Declarations: test_dot, test_axpy, ...
    level1/
      rotg.cpp
      ...14 files...

  lapack/                    (future -- Phase 6+)
    lapack_common.h
    factorizations/
    solvers/
    eigenvalues/
    svd/

  main.cpp                   Unified entry point: CLI, library loading, dispatch

test/
  run_regression.sh          CI regression test against reference BLAS

examples/                    Updated scripts use linalg_tester
third_party/
  CLI11.hpp
```

One header per BLAS level (`level3.h`, `level2.h`, `level1.h`) declares
all test functions for that level. This avoids 50+ single-declaration
header files while keeping `main.cpp`'s includes to 3 lines.

### Per-routine file structure

Each routine file (e.g., `src/blas/level3/gemm.cpp`) is self-contained:

```cpp
// 1. Fortran-ABI function pointer typedef
extern "C" typedef void (*gemm_fn_t)( ... );

// 2. MPFR reference implementation (static)
static void mpfr_gemm_ref(MpfrMatrix &C_ref, ...) { ... }

// 3. Public test driver
void test_gemm(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params)
{
    auto *fn = reinterpret_cast<gemm_fn_t>(load_sym(lib, sym));
    // iterate parameter combinations
    // generate inputs
    // call fn
    // compute MPFR reference
    // report errors via report_result()
}
```

### Dispatch in main.cpp

Simple table dispatch using a `TestParams` struct for all per-routine
parameters. No virtual classes, no registration framework:

```cpp
struct TestParams {
    int m = 64, n = 64, k = 64;
    int kl = 2, ku = 2;
    int incx = 1, incy = 1;
    int ld_pad = 0;
    unsigned seed = 42;
};

struct RoutineEntry {
    const char *name;
    const char *category;
    const char *description;
    void (*test_fn)(const TesterCtx &, void *, const char *, const TestParams &);
};

static const RoutineEntry routines[] = {
    {"gemm",  "blas3", "General matrix multiply",  test_gemm},
    {"trsm",  "blas3", "Triangular solve",         test_trsm},
    {"symm",  "blas3", "Symmetric matrix multiply", test_symm},
    // ...
};
```

Each routine reads only the `TestParams` fields it needs; the rest are
ignored. The dispatch signature is uniform for all BLAS and LAPACK
routines. Level 1 routines that return values (DOT, NRM2, ASUM, IAMAX)
handle the return-value capture internally via their own function pointer
typedefs -- the dispatch signature remains `void`.

### Build system

**CMake is the primary and only supported build system.** The existing
Makefile will be removed. Reasons:
- Nested subdirectory structure is painful to maintain in Make.
- CMake handles `pkg-config` (GMP/MPFR) natively.
- The Makefile already uses a different C++ standard (C++14 vs C++17).

`CMakeLists.txt` will use **explicit file lists** (not GLOB) grouped by
directory. This avoids CMake's stale-build problem with GLOB and serves
as documentation of which routines are implemented:

```cmake
set(CORE_SOURCES src/core/error_metrics.cpp src/core/generators.cpp src/core/report.cpp src/core/sentinel.cpp)
set(BLAS3_SOURCES src/blas/level3/gemm.cpp src/blas/level3/trsm.cpp ...)
set(BLAS2_SOURCES src/blas/level2/gemv.cpp ...)
set(BLAS1_SOURCES src/blas/level1/dot.cpp ...)
add_executable(linalg_tester src/main.cpp
    ${CORE_SOURCES} ${BLAS3_SOURCES} ${BLAS2_SOURCES} ${BLAS1_SOURCES})
```

C++ standard: **C++17** (required by both build systems, consistent).

---

## 3. Infrastructure Components

### 3.1 Matrix / Vector Generators

Current generators:
- `gen_random_array` -- general dense, values in (-1, 1)
- `gen_triangular_array` -- triangular, diagonally dominant

New generators needed:

| Generator | Used by | Description |
|-----------|---------|-------------|
| `gen_symmetric_array(n, uplo)` | SYMM, SYMV, SYR, SYR2, SYRK, SYR2K, etc. | Symmetric matrix: generate upper/lower, mirror |
| `gen_band_array(m, n, kl, ku)` | GBMV | General banded in LAPACK band storage (ldab = kl+ku+1) |
| `gen_symmetric_band_array(n, k, uplo)` | SBMV | Symmetric banded (ldab = k+1) |
| `gen_triangular_band_array(n, k, uplo, diag)` | TBMV, TBSV | Triangular banded (ldab = k+1) |
| `gen_packed_symmetric_array(n, uplo)` | SPMV, SPR, SPR2 | Symmetric packed (n*(n+1)/2 elements) |
| `gen_packed_triangular_array(n, uplo, diag)` | TPMV, TPSV | Triangular packed |
| `gen_positive_definite_array(n, cond)` | LAPACK: POTRF, POSV, etc. | SPD with controllable condition number |
| `gen_rotation_params()` | ROTG, ROTMG | Special scalar inputs for rotation generation |

All generators follow the same pattern: take `typesize`, `mpfr_to_custom_fn`,
`prec`, `seed`; return `malloc`'d memory.

**Stride support**: Add a utility `scatter_strided(void *dst,
const void *contiguous_src, int n, int inc, size_t typesize)` that places
contiguous elements at stride offsets. Used for Level 1 and Level 2 vector
arguments when testing `incx != 1` or `incx < 0`.

**Storage format reference**:

- **Banded**: Element `A(i,j)` is stored at `AB[ku + i - j + j*ldab]`
  (0-based, column-major). Leading dimension `ldab >= kl + ku + 1`.
  Valid range for column `j`: `max(0, j-ku) <= i <= min(m-1, j+kl)`.
- **Packed upper**: `A(i,j)` (i <= j) is at index `i + j*(j+1)/2`.
- **Packed lower**: `A(i,j)` (i >= j) is at index `i + j*(2*n-j-1)/2`.

### 3.2 Error Metrics

Current: `compute_error_metrics(MpfrMatrix &ref, void *custom_out, ld, ctx)`
returns `ErrorResult{max_relative, normwise_relative}`.

**Extended for BLAS**:

```cpp
// Matrix output (Level 3, Level 2 rank updates)
ErrorResult compute_error_matrix(const MpfrMatrix &ref, const void *out, int ld,
                                 const TesterCtx &ctx);

// Matrix output, comparing only one triangle (SYRK, SYR2K, SYR, SPR, etc.)
ErrorResult compute_error_matrix_triangle(const MpfrMatrix &ref, const void *out,
                                          int ld, char uplo, const TesterCtx &ctx);

// Vector output (Level 2 multiply/solve, Level 1 array-modifying ops)
// ref is MpfrMatrix(n,1,prec); inc is element stride in out.
ErrorResult compute_error_vector(const MpfrMatrix &ref, const void *out, int inc,
                                 const TesterCtx &ctx);

// Scalar output (DOT, NRM2, ASUM)
ErrorResult compute_error_scalar(const MpfrScalar &ref, const void *out,
                                 const TesterCtx &ctx);

// Integer output (IAMAX) -- exact comparison
bool compute_error_index(int ref, int out);
```

**Sentinel result** (reported alongside `ErrorResult`):

```cpp
struct SentinelResult {
    bool passed;              // true if all sentinels survived
    int corrupted_count;      // number of corrupted sentinel positions
    std::size_t first_offset; // byte offset of first corruption (for diagnostics)
};
```

When sentinel checking fails, the tester reports it as a hard error
(exit nonzero if `--threshold` is set), regardless of accuracy results.
A routine that produces correct values but corrupts adjacent memory is
still buggy.

**Edge case handling** (apply to all `compute_error_*` functions):

1. **Zero reference value**: When `ref[i,j] == 0`, the element-wise
   relative error `|err|/|ref|` is undefined. If the computed value
   is also zero, skip (current behavior). If the computed value is
   nonzero, report the absolute error as a separate metric
   (`max_absolute_at_zero`).

2. **NaN/Inf**: Before computing the ratio, check for special values:
   - If `ref` is NaN, verify `out` is also NaN; otherwise flag mismatch.
   - If `ref` is +/-Inf, verify `out` is Inf with the same sign.
   - If `ref` is finite but `out` is NaN or Inf, report as error.

3. **Triangle-only comparison**: For routines that only define one
   triangle of the output (SYRK, SYR2K, SYR, HER, etc.), use
   `compute_error_matrix_triangle` which skips the irrelevant triangle.

**For LAPACK (Phase 6+)**, add backward-error metrics:

```cpp
// Solve residual: ||A*X - B|| / (||A||_1 * ||X||_1 * n * eps)
double compute_solve_residual(...);

// Factorization residual: ||A - reconstruct(factors)|| / (||A||_1 * n * eps)
double compute_factorization_residual(...);

// Orthogonality: ||Q^T*Q - I||_1 / (max(m,n) * eps)
double compute_orthogonality(...);
```

### 3.3 MPFR Types (promote to public header)

Extract from `reference.cpp` into `src/core/mpfr_types.h`:
- `MpfrMatrix` -- RAII column-major mpfr_t matrix
- `MpfrScalar` -- RAII single mpfr_t
- `IDX(i, j, ld)` -- column-major indexing
- `custom_to_mpfr_mat()` -- bulk conversion

For vectors, use `MpfrMatrix(n, 1, prec)` directly. No separate
`MpfrVector` class (adds API surface with no functional benefit).
Add `custom_to_mpfr_vec()` for stride-aware vector conversion.

### 3.4 Out-of-Bounds Write Detection (Sentinel Checking)

A correct BLAS/LAPACK implementation must modify **only** the documented
output elements. Elements outside the active region must remain untouched.
The tester verifies this by filling the entire output buffer (including
inactive positions) with a known sentinel pattern before calling the
routine, then checking that sentinels survive.

**Three categories of inactive positions**:

1. **Leading-dimension padding** (matrices with `ld > rows`):
   A column-major matrix with `m` rows and leading dimension `ld` has
   `ld - m` padding elements per column. These are at positions
   `buf[i + j*ld]` for `m <= i < ld`. The routine must not write there.

   ```
   Column j of a matrix with m=3 rows, ld=5:
     row 0: active
     row 1: active
     row 2: active
     row 3: sentinel  ← must survive
     row 4: sentinel  ← must survive
   ```

2. **Inter-stride gaps** (vectors with `|inc| > 1`):
   A vector of length `n` with increment `inc` occupies
   `1 + (n-1)*|inc|` element slots. Only elements at positions
   `i * |inc|` (for `i = 0..n-1`) are active. The `|inc| - 1` elements
   between each pair of active elements are gaps.

   ```
   Vector with n=3, incx=3:
     [0]: active
     [1]: sentinel  ← must survive
     [2]: sentinel  ← must survive
     [3]: active
     [4]: sentinel  ← must survive
     [5]: sentinel  ← must survive
     [6]: active
   ```

3. **Unreferenced triangle** (SYRK, SYR2K, SYR, HER, etc.):
   Routines that only update one triangle (specified by `uplo`) must
   not modify the other triangle. For example, `SYRK` with `uplo='U'`
   must leave the strict lower triangle of C unchanged.

   ```
   uplo='U', n=3:
     (0,0): active    (0,1): active    (0,2): active
     (1,0): sentinel  (1,1): active    (1,2): active
     (2,0): sentinel  (2,1): sentinel  (2,2): active
   ```

**Implementation**:

```cpp
// 1. Allocate buffer with room for padding/gaps
void *buf = alloc_with_sentinel(total_elements, typesize, sentinel_seed);

// 2. Place input data at active positions only
place_matrix_active(buf, input, m, n, ld, typesize);
// or: place_vector_active(buf, input, n, inc, typesize);

// 3. Call the BLAS routine (modifies active positions)
blas_fn(..., buf, ...);

// 4. Check active elements against MPFR reference (existing error metrics)
ErrorResult err = compute_error_matrix(ref, buf, ld, ctx);

// 5. Check that ALL sentinel positions are untouched
SentinelResult sr = check_sentinels(buf, total_elements, typesize,
                                     active_mask, sentinel_seed);
```

`alloc_with_sentinel` fills every element slot with a deterministic
pattern derived from `sentinel_seed` (e.g., via a separate PRNG stream).
`check_sentinels` compares every non-active position against the
expected sentinel value byte-for-byte. Any mismatch indicates an
out-of-bounds write.

**Sentinel value**: Use a NaN bit pattern (signaling NaN) for
floating-point types. This provides a double signal:
- If an implementation accidentally **writes** to a sentinel position,
  the byte-for-byte check catches it.
- If an implementation accidentally **reads** from a sentinel position
  (e.g., the unreferenced triangle of a symmetric input matrix), the
  NaN propagates into the output and fails the accuracy check.

For input matrices with structural constraints (triangular A in TRSM,
symmetric A in SYMM), generators should fill the "don't care" region
with NaN sentinels rather than zeros, so that read-boundary violations
are also detected.

**`active_mask` specification per routine category**:

| Category | Active positions |
|----------|-----------------|
| Full matrix output (GEMM, TRSM, SYMM, TRMM, GER) | `(i,j)` where `0 <= i < m`, `0 <= j < n` |
| Triangle output (SYRK, SYR2K, SYR, HER, SPR, etc.) | `(i,j)` in the `uplo` triangle only |
| Vector output (GEMV→y, TRMV→x, AXPY→y, etc.) | `k * |inc|` for `k = 0..n-1` |
| In-place vector (SCAL, TRMV, TRSV) | Same as vector output |
| Dual vector output (SWAP, ROT) | Active in both x and y |
| No output (COPY→y is fully written, SWAP→both fully written) | All `n` strided positions |

**When to check**: Sentinel checking is always performed when
`ld_pad > 0` or `|incx| > 1` or `|incy| > 1`, or for triangle-only
routines. When `ld_pad == 0` and `inc == 1`, there are no inactive
positions to check (the buffer is exactly sized).

### 3.5 Output Formatting

Centralize output in `src/core/report.h/.cpp`:

```cpp
void report_result(const char *routine, const char *params_str,
                   const ErrorResult &err, const SentinelResult *sentinel,
                   const char *format);
```

All routines call `report_result()` instead of inline `printf`. This
enables format switching (`--format text/json/csv`) as a trivial
change rather than touching every routine.

---

## 4. Phased Implementation

### Phase 1: Architectural Refactor

**Goal**: Restructure the codebase into the target layout. Migrate GEMM and
TRSM. Establish CI. All existing examples must still work.

**Steps**:

1. **Create `src/core/` and extract shared infrastructure**:
   - `tester_ctx.h` -- move `TesterCtx`, `ErrorResult`, `TestParams`,
     conversion fn typedefs from `reference.h`
   - `mpfr_types.h` -- extract `MpfrMatrix`, `MpfrScalar`, `IDX`,
     `custom_to_mpfr_mat` from `reference.cpp`
   - `error_metrics.h/.cpp` -- extract `compute_error_metrics` from
     `reference.cpp`; rename to `compute_error_matrix`; add NaN/Inf
     and zero-reference handling
   - `generators.h/.cpp` -- move `gen_random_array`, `gen_triangular_array`
     from `tester_utils.h`; replace `rand_r` (POSIX-only, deprecated)
     with `std::mt19937` from `<random>`
   - `loader.h` -- move `preload_libs`, `close_libs`, `load_sym`
   - `report.h/.cpp` -- output formatting and threshold checking
   - `sentinel.h/.cpp` -- sentinel allocation, placement, and checking
     (Section 3.4); implement `alloc_with_sentinel`,
     `place_matrix_active`, `place_vector_active`, `check_sentinels`

2. **Create `src/blas/level3/` and migrate GEMM/TRSM**:
   - `src/blas/level3.h` -- declares `test_gemm`, `test_trsm`
   - `src/blas/level3/gemm.cpp` -- move typedef, reference, test loop;
     fix hidden char length literals from `1` to `(std::size_t)1`;
     use `ld_pad` for leading dimensions (`ldc = m + ld_pad`);
     use sentinel infrastructure to verify ld-padding is untouched
   - `src/blas/level3/trsm.cpp` -- same for TRSM

3. **Create `src/main.cpp`** -- unified entry point:
   - Parse all CLI args into `TestParams`
   - Handle `--list` (print routines, exit; no library flags required)
   - Handle `--threshold` (set exit code on exceeded)
   - Load libraries, build TesterCtx
   - Dispatch via `RoutineEntry` table

4. **Update build system**:
   - Rewrite `CMakeLists.txt` with explicit file lists, C++17, single target
   - Remove `Makefile`

5. **Update example scripts** to call `linalg_tester --routine gemm ...`
   and `linalg_tester --routine trsm ...`.

6. **Establish CI** (GitHub Actions or equivalent):
   - Build on Linux (Ubuntu) with `libblas3`, `libmpfr-dev`, `libgmp-dev`
   - Run `test/run_regression.sh` that tests all implemented routines
     against reference BLAS with `--threshold 1e-12 --seed 42 --m 32 --n 32`
   - Also run with `--ld-pad 3` to verify no out-of-bounds writes
   - Verify exit code 0

7. **Update README.md** to document new CLI, build instructions, and layout.

8. **Test**: Run all existing example scripts, verify identical output.

9. **Delete old files**: `src/gemm_tester.cpp`, `src/trsm_tester.cpp`,
   `src/reference.cpp`, `src/reference.h`, `src/tester_utils.h`.
   (Done after step 8 passes -- old files are preserved until verified.)

**CLI migration guide** (for users of existing scripts):
```
# Before:
gemm_tester --lib foo.so --gemm-sym dgemm_ --conv-lib conv.so --typesize 8

# After:
linalg_tester --routine gemm --lib foo.so --sym dgemm_ --conv-lib conv.so --typesize 8
```

**Deliverable**: Same 2 routines, new architecture, CI passing, all
examples working.

---

### Phase 2: Complete BLAS Level 3 (Real)

**Goal**: Add the remaining 4 real Level 3 BLAS routines.

**New generators needed**: `gen_symmetric_array`.

**Routines** (one `.cpp` per routine under `src/blas/level3/`):

| # | Routine | Operation | Parameters tested |
|---|---------|-----------|-------------------|
| 1 | **SYMM** | C = alpha*A*B + beta*C (A symmetric) | (side, uplo) in {L,R} x {U,L} = 4 combos |
| 2 | **SYRK** | C = alpha*A*A^T + beta*C (trans=N) or C = alpha*A^T*A + beta*C (trans=T) | (uplo, trans) in {U,L} x {N,T,C} = 6 combos |
| 3 | **SYR2K** | C = alpha*A*B^T + alpha*B*A^T + beta*C (trans=N) or transposed variant | (uplo, trans) in {U,L} x {N,T,C} = 6 combos |
| 4 | **TRMM** | B = alpha*op(A)*B or B = alpha*B*op(A) (A triangular) | (side, uplo, transa, diag) in {L,R}x{U,L}x{N,T,C}x{N,U} = 24 combos |

**Notes on 'C' transpose**: The BLAS standard defines `trans` values as
`{N, T, C}` where `C` means conjugate transpose. For real types, `C` is
equivalent to `T` but must be accepted. Testing `C` exercises the
library's character handling. All routines with a `trans` parameter should
test `{N, T, C}` (3 values), not just `{N, T}`. This also applies to
the migrated GEMM (9 combos: `{N,T,C}^2`) and TRSM (24 combos).

**SYRK detailed formulas** (corrected):
- `trans='N'`: `C = alpha * A * A^T + beta * C`, where A is n-by-k.
- `trans='T'`: `C = alpha * A^T * A + beta * C`, where A is k-by-n.
- Only the `uplo` triangle of C is computed; the other triangle must
  remain untouched.
- MPFR ref: compute the full product, then compare only the `uplo`
  triangle via `compute_error_matrix_triangle`.
- Sentinel check: fill C with sentinels before the call; verify that
  the unreferenced triangle AND any ld-padding positions survive.

**SYR2K detailed formulas**:
- `trans='N'`: `C = alpha*A*B^T + alpha*B*A^T + beta*C`, A and B are n-by-k.
- `trans='T'`: `C = alpha*A^T*B + alpha*B^T*A + beta*C`, A and B are k-by-n.

**SYMM ABI**: `(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, side_len, uplo_len)`.
**SYRK ABI**: `(uplo, trans, n, k, alpha, A, lda, beta, C, ldc, uplo_len, trans_len)`.
**SYR2K ABI**: `(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc, uplo_len, trans_len)`.
**TRMM ABI**: same signature as TRSM.

**Deliverable**: 6 of 6 real Level 3 routines implemented and tested.

---

### Phase 3: BLAS Level 2 (Real)

**Goal**: Implement all 16 real Level 2 BLAS routines.

**New generators needed**:
- `gen_band_array` (for GBMV)
- `gen_symmetric_band_array` (for SBMV)
- `gen_triangular_band_array` (for TBMV, TBSV)
- `gen_packed_symmetric_array` (for SPMV, SPR, SPR2)
- `gen_packed_triangular_array` (for TPMV, TPSV)
- `scatter_strided` utility for non-unit stride testing

**New error metrics**: `compute_error_vector`, `compute_error_matrix_triangle`.

**Stride testing**: Each Level 2 test driver should iterate over at least
`incx, incy in {1, 2, -1}` to verify stride handling. Negative increments
mean elements are accessed in reverse order per the BLAS standard.
When `|inc| > 1`, sentinel checking (Section 3.4) verifies that the
inter-stride gap elements are not written by the routine.

**Leading dimension testing**: Use `--ld-pad` to add padding to leading
dimensions (`lda = rows + ld_pad`). Testing with `ld_pad > 0` catches
bugs where `lda` and `rows` are confused. Sentinel checking verifies
that ld-padding positions survive the call.

**Sub-phase 3a: Dense matrix-vector operations**

| # | Routine | Operation | Parameters tested |
|---|---------|-----------|-------------------|
| 1 | **GEMV** | y = alpha*op(A)*x + beta*y | (trans) in {N,T,C} = 3 combos |
| 2 | **SYMV** | y = alpha*A*x + beta*y (A symmetric) | (uplo) in {U,L} = 2 combos |
| 3 | **TRMV** | x = op(A)*x (A triangular) | (uplo, trans, diag) = 2x3x2 = 12 combos |
| 4 | **TRSV** | op(A)*x = b (A triangular) | (uplo, trans, diag) = 12 combos |

**Sub-phase 3b: Rank-update operations**

| # | Routine | Operation | Parameters tested |
|---|---------|-----------|-------------------|
| 5 | **GER** | A = alpha*x*y^T + A | 1 combo |
| 6 | **SYR** | A = alpha*x*x^T + A (A symmetric) | (uplo) = 2 combos |
| 7 | **SYR2** | A = alpha*x*y^T + alpha*y*x^T + A | (uplo) = 2 combos |

SYR, SYR2 modify only the `uplo` triangle; use `compute_error_matrix_triangle`
for accuracy. Sentinel checking verifies the unreferenced triangle is untouched.

**Sub-phase 3c: Banded-storage routines**

| # | Routine | Operation | Parameters tested |
|---|---------|-----------|-------------------|
| 8 | **GBMV** | y = alpha*op(A)*x + beta*y (A banded) | (trans) = 3 combos |
| 9 | **SBMV** | y = alpha*A*x + beta*y (A symmetric banded) | (uplo) = 2 combos |
| 10 | **TBMV** | x = op(A)*x (A triangular banded) | (uplo, trans, diag) = 12 combos |
| 11 | **TBSV** | op(A)*x = b (A triangular banded) | (uplo, trans, diag) = 12 combos |

**Sub-phase 3d: Packed-storage routines**

| # | Routine | Operation | Parameters tested |
|---|---------|-----------|-------------------|
| 12 | **SPMV** | y = alpha*A*x + beta*y (A symmetric packed) | (uplo) = 2 combos |
| 13 | **TPMV** | x = op(A)*x (A triangular packed) | (uplo, trans, diag) = 12 combos |
| 14 | **TPSV** | op(A)*x = b (A triangular packed) | (uplo, trans, diag) = 12 combos |
| 15 | **SPR** | A = alpha*x*x^T + A (A symmetric packed) | (uplo) = 2 combos |
| 16 | **SPR2** | A = alpha*x*y^T + alpha*y*x^T + A (packed) | (uplo) = 2 combos |

**MPFR reference notes for Level 2**:
- Matrix-vector multiply refs are straightforward (double loop).
- Triangular solve refs reuse forward/backward substitution logic from
  TRSM but for a single RHS vector.
- Banded/packed refs must correctly interpret the storage layout (see
  index formulas in Section 3.1).
- Rank updates modify only the declared triangle in-place; the MPFR
  reference computes the expected updated triangle element-by-element.

**Deliverable**: 22 of 22 real BLAS Level 2+3 routines.

---

### Phase 4: BLAS Level 1 (Real)

**Goal**: Implement all 14 real Level 1 BLAS routines.

**New error metrics**: `compute_error_scalar` (for DOT, NRM2, ASUM),
`compute_error_index` (for IAMAX).

**Stride testing**: Test `incx in {1, 2, -1}` for all Level 1 routines.
Sentinel checking verifies inter-stride gaps are not written. For
routines that modify vectors in-place (SCAL, ROT, ROTM, SWAP), also
verify that only the `n` active strided elements are modified and all
gap elements survive.

| # | Routine | Signature (return -> args) | Notes |
|---|---------|--------------------------|-------|
| 1 | **ROTG** | `void -> (a, b, c, s)` | Modifies a,b; outputs c,s. Also outputs z (reconstruction param). |
| 2 | **ROTMG** | `void -> (d1, d2, x1, y1, param[5])` | Modified Givens; outputs param[0]=flag, param[1..4]=matrix entries. |
| 3 | **ROT** | `void -> (n, x, incx, y, incy, c, s)` | Applies rotation to x,y vectors |
| 4 | **ROTM** | `void -> (n, x, incx, y, incy, param[5])` | Applies modified Givens rotation |
| 5 | **SWAP** | `void -> (n, x, incx, y, incy)` | Swaps x and y |
| 6 | **SCAL** | `void -> (n, alpha, x, incx)` | x = alpha*x |
| 7 | **COPY** | `void -> (n, x, incx, y, incy)` | y = x |
| 8 | **AXPY** | `void -> (n, alpha, x, incx, y, incy)` | y = alpha*x + y |
| 9 | **DOT** | `scalar -> (n, x, incx, y, incy)` | Returns x^T * y |
| 10 | **NRM2** | `scalar -> (n, x, incx)` | Returns ||x||_2 |
| 11 | **ASUM** | `scalar -> (n, x, incx)` | Returns sum(\|x_i\|) |
| 12 | **IAMAX** | `int -> (n, x, incx)` | Returns argmax_i(\|x_i\|) (1-based) |
| 13 | **SDSDOT** | `float -> (n, sb, sx, incx, sy, incy)` | sb + dot(sx,sy) with double accum |
| 14 | **DSDOT** | `double -> (n, sx, incx, sy, incy)` | Double-precision dot of single vectors |

**Return value calling conventions** (see also Section 6):

Level 1 routines with scalar returns (DOT, NRM2, ASUM) have no hidden
character-length arguments since they take no character parameters. The
function pointer typedef must declare the correct return type:

```cpp
extern "C" typedef double (*ddot_fn_t)(const int *n,
    const void *x, const int *incx, const void *y, const int *incy);
extern "C" typedef int (*idamax_fn_t)(const int *n,
    const void *x, const int *incx);
```

On x86-64 and aarch64 with gfortran, `double` returns in a floating-point
register (XMM0 / D0). This is straightforward. SDSDOT returns `float`.
The dispatch signature remains `void (*)(...)`; the return-value capture
is handled inside each test driver's own typedef.

**MPFR references for Level 1**:
- **DOT**: Single loop accumulating `x[i*incx] * y[i*incy]` in MPFR.
- **NRM2**: Accumulate sum of squares in MPFR, take sqrt. The naive
  approach is intentionally used because MPFR's extended precision
  eliminates overflow/underflow concerns. The library under test is
  expected to implement proper scaling (e.g., Blue's algorithm).
- **ASUM**: Accumulate absolute values in MPFR.
- **AXPY**: Element-wise `y[i] = alpha * x[i] + y[i]`.
- **ROT**: Apply 2x2 rotation matrix `[c s; -s c]` to each `(x[i], y[i])` pair.
- **ROTG**: Compute `c`, `s`, `r`, `z` from `a`, `b` such that
  `[c s; -s c]^T [a; b] = [r; 0]`. The sign convention follows the
  Fortran reference: `sigma = sgn(a)` if `|a| > |b|`, else `sigma = sgn(b)`.
  `r = sigma * sqrt(a^2 + b^2)`. The reconstruction parameter `z`
  encodes `c` and `s` compactly.
- **ROTMG**: Produces a 5-element PARAM array where `PARAM(1)` is a flag
  value (-1, 0, 1, or -2) indicating which elements of the 2x2 matrix
  are stored in `PARAM(2:5)`. The reference must implement the full
  scaling logic to avoid overflow.
- **SWAP/COPY**: Exact operations (error should be identically zero).

**Deliverable**: All 36 real BLAS routines implemented and tested.

---

### Phase 5: Complex Number Support

**Goal**: Add complex-valued type support and implement all complex-only
BLAS routines (~19 additional routines) plus complex variants of all
existing real routines.

**Infrastructure changes**:

1. **Complex conversion API**:
   The conversion library exports two additional symbols:
   ```c
   void custom_to_mpfr_complex(mpfr_t re, mpfr_t im, const void *src);
   void mpfr_to_custom_complex(void *dst, mpfr_t re, mpfr_t im, mpfr_rnd_t rnd);
   ```
   `--typesize` is always the size of one complete element as stored in
   memory (16 for `complex double`, 32 for `complex __float128`).
   The tester detects complex mode via a `--complex` flag.
   If `--complex` is specified but the conversion library does not export
   the complex symbols, `load_sym` will report an error and exit.

2. **MPFR complex arithmetic** (in `src/core/mpfr_complex.h`):
   Use pairs of `mpfr_t` for real and imaginary parts. Implement:
   - Complex multiply, add, subtract, FMA
   - Complex conjugate
   - Complex absolute value: `sqrt(re^2 + im^2)`

3. **`MpfrComplexMatrix`**: Stores parallel real and imaginary `MpfrMatrix`.

4. **Complex error metrics**: Same formulas but using complex absolute
   values for norms.

**Design consideration -- early vs late complex**: Introducing complex
support this late means all ~34 real-only MPFR references from Phases 2-4
need complex counterparts. To mitigate this: write real references using
a style that cleanly extends to complex (e.g., always use `mpfr_fma` for
dot-product accumulation, structure loops identically). This makes the
Phase 5 extension a mechanical transformation rather than a redesign.

**Complex-only BLAS routines**:

| Level | Routine | Operation | Notes |
|-------|---------|-----------|-------|
| 3 | **HEMM** | C = alpha*A*B + beta*C (A Hermitian) | |
| 3 | **HERK** | C = alpha*A*A^H + beta*C | **alpha and beta are REAL, not complex** |
| 3 | **HER2K** | C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C | **alpha is complex, beta is REAL** |
| 2 | **HEMV** | y = alpha*A*x + beta*y (A Hermitian) | |
| 2 | **HBMV** | Hermitian banded matrix-vector | |
| 2 | **HPMV** | Hermitian packed matrix-vector | |
| 2 | **GERU** | A = alpha*x*y^T + A (unconjugated) | |
| 2 | **GERC** | A = alpha*x*conj(y)^T + A (conjugated) | |
| 2 | **HER** | A = alpha*x*conj(x)^T + A (A Hermitian) | |
| 2 | **HPR** | Hermitian packed rank-1 | |
| 2 | **HER2** | Hermitian rank-2 | |
| 2 | **HPR2** | Hermitian packed rank-2 | |
| 1 | **DOTC** | conj(x)^T * y | Return value ABI varies by platform |
| 1 | **DOTU** | x^T * y (unconjugated) | Return value ABI varies by platform |
| 1 | **CSROT/ZDROT** | Apply real rotation to complex vectors | |
| 1 | **CSSCAL/ZDSCAL** | Scale complex vector by real scalar | |

**Complex variant notes**:
- **CSYMM/ZSYMM vs CHEMM/ZHEMM**: Complex SYMM operates on complex
  symmetric matrices (transpose-equal, NOT conjugate-transpose-equal).
  The generator `gen_symmetric_array` for complex must produce A where
  `A = A^T` (not `A = A^H`). `gen_hermitian_array` produces `A = A^H`.
  These are distinct generators.
- **CSYRK/ZSYRK**: `trans` parameter accepts `{N,T}` but NOT `'C'`.
  CHERK/ZHERK accepts `{N,C}` but NOT `'T'`.
- **DOTC/DOTU return values**: On gfortran/Linux, complex function returns
  use a hidden first argument (`void *` to receive the result). On
  macOS/x86-64, complex values return in `xmm0`/`xmm1`. This will
  likely require platform-specific function pointer typedefs or thin
  wrapper functions.

**New generators**:
- `gen_hermitian_array(n, uplo)` -- Hermitian (A = A^H)
- `gen_complex_symmetric_array(n, uplo)` -- complex symmetric (A = A^T)
- `gen_hermitian_band_array(n, k, uplo)`
- `gen_packed_hermitian_array(n, uplo)`

**Deliverable**: Complete BLAS coverage (~55 routines, real and complex).

---

### Phase 6: LAPACK Infrastructure

**Goal**: Lay the groundwork for LAPACK testing.

**Key difference from BLAS**: Most LAPACK routines compute factorizations
or decompositions that are not unique (e.g., LU with different pivot
orders, eigenvectors with arbitrary sign/phase). For these, use
**backward error** and **residual-based** metrics instead of element-wise
MPFR comparison.

However, some LAPACK routines DO produce unique, deterministic results
and CAN be tested by direct MPFR comparison (like BLAS):
- **LANGE/LANSY** -- matrix norms (scalar output)
- **LACPY** -- matrix copy (should be bitwise identical)
- **LASWP** -- row permutations (unique result)
- **GETRS/POTRS** -- triangular solves with fixed factors (unique result;
  can reuse MPFR TRSM reference)

**Error metric categories**:

| Category | Formula | Used by |
|----------|---------|---------|
| Solve residual | `\|\|Ax - b\|\|_1 / (\|\|A\|\|_1 * \|\|x\|\|_1 * n * eps)` | xGESV, xPOSV, xSYSV |
| LU residual | `\|\|A - PLU\|\|_1 / (\|\|A\|\|_1 * n * eps)` | xGETRF |
| Cholesky residual | `\|\|A - LL^T\|\|_1 / (\|\|A\|\|_1 * n * eps)` | xPOTRF |
| QR residual | `\|\|A - QR\|\|_1 / (\|\|A\|\|_1 * n * eps)` | xGEQRF |
| Orthogonality | `\|\|Q^TQ - I\|\|_1 / (max(m,n) * eps)` | xGEQRF, eigen, SVD |
| Eigenvalue residual | `\|\|AV - VD\|\|_1 / (\|\|A\|\|_1 * n * eps)` | xSYEV, xGEEV |
| SVD residual | `\|\|A - USV^T\|\|_1 / (\|\|A\|\|_1 * min(m,n) * eps)` | xGESVD |
| Inverse residual | `\|\|A * A^{-1} - I\|\|_1 / (\|\|A\|\|_1 * \|\|A^{-1}\|\|_1 * n * eps)` | xGETRI, xPOTRI |

**Workspace handling**:

Most LAPACK computational routines require workspace arrays (WORK,
LWORK, IWORK, RWORK). Strategy:
1. Always perform a workspace query first (call with `LWORK=-1`) to
   get the optimal workspace size from the library under test.
2. Allocate the returned optimal size.
3. Optionally also test with the minimum documented workspace
   (`--min-workspace` flag) to catch workspace-edge-case bugs.
4. Routines without workspace (GETRF, POTRF, GETRS, POTRS) skip this.

**INFO return value checking**:

Every LAPACK routine returns an INFO integer. The test driver must:
1. Always verify `INFO == 0` after calling with valid inputs.
2. If `INFO != 0`, report it and skip the residual check.
3. The tester always provides valid parameters; negative testing
   (intentionally invalid inputs to verify XERBLA behavior) is out of
   scope.

**MPFR utility functions needed** (build incrementally as Phases 7-10
require them):

| Utility | Needed by |
|---------|-----------|
| MPFR general matrix multiply | Residual checks, factor reconstruction |
| MPFR matrix norms (1-norm, Frobenius, inf) | All residual formulas |
| MPFR identity matrix construction + comparison | Orthogonality checks |
| MPFR permutation application from IPIV | GETRF reconstruction |
| MPFR Householder accumulation (Q from reflectors + tau) | GEQRF, GELQF, GEBRD, SYTRD |
| MPFR triangular solve (forward/backward) | GETRS, POTRS independent verification |
| MPFR block diagonal extraction (Bunch-Kaufman) | SYTRF reconstruction |
| MPFR matrix subtraction | All residual computations |
| MPFR Schur form verification (quasi-upper-triangular) | GEES |
| MPFR bidiagonal/tridiagonal matrix ops | GEBRD, SYTRD |

**Critical principle**: Never call the library under test to reconstruct
factors. For example, GEQRF testing must reconstruct Q from Householder
reflectors in MPFR, not by calling the library's ORGQR (which may also
be buggy). All reconstruction happens in MPFR.

**Deliverable**: Error metric framework + MPFR utilities for LAPACK.

---

### Phase 7: LAPACK Factorizations

**Goal**: Test the most commonly used LAPACK factorization routines.

| # | Routine | Operation | Error checks |
|---|---------|-----------|-------------|
| 1 | **GETRF** | PA = LU (partial pivoting) | Residual + structural verification |
| 2 | **POTRF** | A = LL^T (Cholesky) | Residual + structural verification |
| 3 | **SYTRF** | A = LDL^T (Bunch-Kaufman) | Residual |
| 4 | **GEQRF** | A = QR (Householder reflectors) | Residual + orthogonality |
| 5 | **GELQF** | A = LQ | Residual + orthogonality |
| 6 | **GEBRD** | A = U*B*V^T (bidiagonal) | Residual + orthogonality |
| 7 | **SYTRD** | A = Q*T*Q^T (tridiagonal) | Residual + orthogonality |

**Structural verification** (in addition to residuals):

- **GETRF**: Verify L is unit lower triangular (ones on diagonal, zeros
  above) and U is upper triangular (zeros below). Verify IPIV values are
  valid (each in range `[1, min(m,n)]`). Then check
  `||A - P*L*U|| / (||A|| * n * eps)`.
- **POTRF**: Verify L (or U) is triangular and that the non-referenced
  triangle is untouched (sentinel check). Then check
  `||A - L*L^T|| / (||A|| * n * eps)`.
  Requires SPD input from `gen_positive_definite_array`.
- **SYTRF**: Bunch-Kaufman pivot interpretation: positive `IPIV(k)` means
  1x1 pivot at row k; negative `IPIV(k)` = negative `IPIV(k+1)` means
  2x2 pivot spanning rows k and k+1. Reconstruction:
  `A = P^T * L * D * L^T * P`. Alternatively, test indirectly by using
  the factors as input to SYTRS and checking the solve residual.
- **GEQRF**: Reconstruct Q from Householder reflectors in MPFR:
  `Q = (I - tau_1*v_1*v_1^T) * ... * (I - tau_k*v_k*v_k^T)`.
  Then check `||A - Q*R|| / (||A|| * n * eps)` and
  `||Q^T*Q - I|| / (max(m,n) * eps)`.

**Testing dimensions**: Test with both square (`m = n`) and rectangular
(`m != n`) matrices where applicable (GEQRF, GELQF, GEBRD). Also test
`n=1` as a degenerate case.

**Deliverable**: 7 factorization routines with structural + residual checks.

---

### Phase 8: LAPACK Solvers

**Goal**: Test LAPACK driver routines that solve linear systems.

| # | Routine | Operation | Input | Error metric |
|---|---------|-----------|-------|--------------|
| 1 | **GESV** | Solve AX = B via LU | General A | Solve residual |
| 2 | **POSV** | Solve AX = B via Cholesky | SPD A | Solve residual |
| 3 | **SYSV** | Solve AX = B via LDL^T | Symmetric A | Solve residual |
| 4 | **GBSV** | Solve AX = B (A banded) | Banded A | Solve residual |
| 5 | **GTSV** | Solve AX = B (A tridiagonal) | Tridiagonal A | Solve residual |
| 6 | **GELS** | Least squares min\|\|Ax-b\|\|_2 | Over/underdetermined | Residual + solution norm |
| 7 | **GELSD** | Least squares via SVD | Rank-deficient | Residual |

**Solve residual formula** (aligned with LAPACK test suite conventions):
`||A*X - B||_1 / (||A||_1 * ||X||_1 * n * eps)`.

**Testing approach**: Generate A and a known true solution X_true in MPFR.
Compute B = A * X_true in MPFR, then convert B to the target type. Call
the solver with B. Compute both forward error
(`||X_computed - X_true|| / ||X_true||`) and backward error (the
residual).

**Deliverable**: 7 solver routines.

---

### Phase 9: LAPACK Eigenvalue & SVD

**Goal**: Test eigenvalue decomposition and SVD routines.

| # | Routine | Operation | Error checks |
|---|---------|-----------|-------------|
| 1 | **SYEV** | Symmetric eigenvalues + vectors | Residual + orthogonality + eigenvalue ordering |
| 2 | **SYEVD** | Symmetric eigenvalues (D&C) | Same |
| 3 | **SYEVR** | Symmetric eigenvalues (MRRR) | Same |
| 4 | **GEEV** | General eigenvalues + vectors | Residual (per eigenpair) |
| 5 | **GEES** | Schur decomposition | `\|\|A - QTQ^T\|\|` + orthogonality + quasi-upper-triangular T |
| 6 | **GESVD** | SVD: A = USV^T | Residual + orthogonality(U) + orthogonality(V) + sorted + non-negative |
| 7 | **GESDD** | SVD (D&C) | Same as GESVD |
| 8 | **SYGV** | Generalized symmetric eigenvalue | Residual |

**Eigenvalue testing details**:

- **Symmetric (SYEV/SYEVD/SYEVR)**: Eigenvalues are returned in ascending
  order by the standard, so no matching is needed. Compare via:
  `||A*V - V*diag(W)||_1 / (||A||_1 * n * eps)`.
  Check `||V^T*V - I||_1 / (n * eps)`.

- **General (GEEV)**: Eigenvalues have no guaranteed order and may be
  complex even for real input. For real input, complex eigenvalues come
  in conjugate pairs, and DGEEV stores eigenvectors in a packed real
  format: for conjugate pair `(alpha +/- beta*i)`, two columns of VR
  store the real and imaginary parts. The residual for each conjugate
  pair must be computed as:
  `||A * [vr, vi] - [vr, vi] * [[alpha, beta], [-beta, alpha]]||`
  where `vr` and `vi` are the real and imaginary part columns.
  Matching library eigenvalues to reference eigenvalues requires a
  closest-match algorithm (not simple sorting), since complex eigenvalues
  cannot be totally ordered.

- **Schur (GEES)**: Verify T is quasi-upper-triangular (1x1 and 2x2
  diagonal blocks, zeros below). Eigenvalues are on the diagonal of T.

**SVD testing details**:
- Residual: `||A - U*diag(S)*V^T|| / (||A|| * min(m,n) * eps)`.
- Orthogonality: `||U^T*U - I||` and `||V^T*V - I||`.
- Singular values must be non-negative.
- Singular values must be in non-increasing order.

**Deliverable**: 8 eigenvalue/SVD routines.

---

### Phase 10: LAPACK Auxiliary & Remaining

**Goal**: Remaining LAPACK routines of practical interest.

**Routines testable by direct MPFR comparison** (unique deterministic output):

| # | Routine | Purpose | Error metric |
|---|---------|---------|-------------|
| 1 | **GETRS** | Solve using LU factors | MPFR forward comparison (reuses MPFR triangular solve) |
| 2 | **POTRS** | Solve using Cholesky factors | MPFR forward comparison |
| 3 | **ORGQR** | Generate Q from reflectors | MPFR forward comparison (MPFR Householder accumulation) |
| 4 | **ORMQR** | Multiply by Q | MPFR forward comparison |
| 5 | **LANGE** | Matrix norm | MPFR scalar comparison |
| 6 | **LANSY** | Symmetric matrix norm | MPFR scalar comparison |
| 7 | **LACPY** | Matrix copy | Bitwise identical |
| 8 | **LASWP** | Row permutations | MPFR forward comparison |

**Routines requiring residual/special testing**:

| # | Routine | Purpose | Error metric |
|---|---------|---------|-------------|
| 9 | **GETRI** | Inverse from LU | `\|\|A * A^{-1} - I\|\| / (\|\|A\|\| * \|\|A^{-1}\|\| * n * eps)` |
| 10 | **POTRI** | Inverse from Cholesky | Same as GETRI |
| 11 | **GECON** | Condition number estimate | **Estimator**: verify `RCOND` is within a factor of n of true `1/kappa(A)` (computed via MPFR SVD or inverse). Not direct comparison. |

---

## 5. Complete Routine Catalog

### BLAS Level 1 (14 real + 4 complex-only = 18)

| Routine | Category | Type | Phase |
|---------|----------|------|-------|
| ROTG | Rotation generation | Real | 4 |
| ROTMG | Modified Givens generation | Real | 4 |
| ROT | Apply rotation | Real | 4 |
| ROTM | Apply modified Givens | Real | 4 |
| SWAP | Swap vectors | Real | 4 |
| SCAL | Scale vector | Real | 4 |
| COPY | Copy vector | Real | 4 |
| AXPY | y = alpha*x + y | Real | 4 |
| DOT | Dot product | Real | 4 |
| NRM2 | Euclidean norm | Real | 4 |
| ASUM | Sum of \|x_i\| | Real | 4 |
| IAMAX | Index of max \|x_i\| | Real | 4 |
| SDSDOT | sb + dot(sx,sy), double accum | Real (mixed-prec) | 4 |
| DSDOT | Double dot of single vectors | Real (mixed-prec) | 4 |
| DOTC | Conjugated dot | Complex | 5 |
| DOTU | Unconjugated dot | Complex | 5 |
| CSROT/ZDROT | Real rotation on complex vectors | Complex | 5 |
| CSSCAL/ZDSCAL | Real scaling of complex vectors | Complex | 5 |

### BLAS Level 2 (16 real + 9 complex-only = 25)

| Routine | Category | Type | Phase |
|---------|----------|------|-------|
| GEMV | General MV | Real | 3a |
| SYMV | Symmetric MV | Real | 3a |
| TRMV | Triangular MV | Real | 3a |
| TRSV | Triangular solve | Real | 3a |
| GER | Rank-1 update | Real | 3b |
| SYR | Symmetric rank-1 | Real | 3b |
| SYR2 | Symmetric rank-2 | Real | 3b |
| GBMV | General banded MV | Real | 3c |
| SBMV | Symmetric banded MV | Real | 3c |
| TBMV | Triangular banded MV | Real | 3c |
| TBSV | Triangular banded solve | Real | 3c |
| SPMV | Symmetric packed MV | Real | 3d |
| TPMV | Triangular packed MV | Real | 3d |
| TPSV | Triangular packed solve | Real | 3d |
| SPR | Symmetric packed rank-1 | Real | 3d |
| SPR2 | Symmetric packed rank-2 | Real | 3d |
| HEMV | Hermitian MV | Complex | 5 |
| HBMV | Hermitian banded MV | Complex | 5 |
| HPMV | Hermitian packed MV | Complex | 5 |
| GERU | Unconjugated rank-1 | Complex | 5 |
| GERC | Conjugated rank-1 | Complex | 5 |
| HER | Hermitian rank-1 | Complex | 5 |
| HPR | Hermitian packed rank-1 | Complex | 5 |
| HER2 | Hermitian rank-2 | Complex | 5 |
| HPR2 | Hermitian packed rank-2 | Complex | 5 |

### BLAS Level 3 (6 real + 3 complex-only = 9)

| Routine | Category | Type | Phase |
|---------|----------|------|-------|
| GEMM | General MM | Real | 1 (existing) |
| TRSM | Triangular solve | Real | 1 (existing) |
| SYMM | Symmetric MM | Real | 2 |
| SYRK | Symmetric rank-k | Real | 2 |
| SYR2K | Symmetric rank-2k | Real | 2 |
| TRMM | Triangular MM | Real | 2 |
| HEMM | Hermitian MM | Complex | 5 |
| HERK | Hermitian rank-k | Complex | 5 |
| HER2K | Hermitian rank-2k | Complex | 5 |

**BLAS total: 14 + 16 + 6 = 36 real, + 4 + 9 + 3 = 16 complex-only
= 52 routine templates. With S/D/C/Z prefixes, the actual symbol count
is higher but the test driver logic is shared per template.**

### LAPACK (33 routines)

| Routine | Category | Phase |
|---------|----------|-------|
| GETRF | LU factorization | 7 |
| POTRF | Cholesky factorization | 7 |
| SYTRF | Symmetric LDL^T | 7 |
| GEQRF | QR factorization | 7 |
| GELQF | LQ factorization | 7 |
| GEBRD | Bidiagonal reduction | 7 |
| SYTRD | Tridiagonal reduction | 7 |
| GESV | General solve | 8 |
| POSV | SPD solve | 8 |
| SYSV | Symmetric solve | 8 |
| GBSV | Banded solve | 8 |
| GTSV | Tridiagonal solve | 8 |
| GELS | Least squares | 8 |
| GELSD | Least squares (SVD) | 8 |
| SYEV | Symmetric eigenvalue | 9 |
| SYEVD | Symmetric eigenvalue (D&C) | 9 |
| SYEVR | Symmetric eigenvalue (MRRR) | 9 |
| GEEV | General eigenvalue | 9 |
| GEES | Schur decomposition | 9 |
| GESVD | SVD | 9 |
| GESDD | SVD (D&C) | 9 |
| SYGV | Generalized symmetric eigenvalue | 9 |
| GETRS | Solve from LU | 10 |
| GETRI | Inverse from LU | 10 |
| POTRS | Solve from Cholesky | 10 |
| POTRI | Inverse from Cholesky | 10 |
| ORGQR | Generate Q | 10 |
| ORMQR | Multiply by Q | 10 |
| GECON | Condition number estimate | 10 |
| LANGE | Matrix norm | 10 |
| LANSY | Symmetric norm | 10 |
| LACPY | Matrix copy | 10 |
| LASWP | Row permutations | 10 |

### Not included / Future work

The following LAPACK categories are deliberately deferred. They can be
added incrementally after Phase 10:

| Category | Key routines | Reason for deferral |
|----------|-------------|---------------------|
| Iterative refinement | xGERFS, xPORFS, xSYRFS | Builds on Phase 8 solvers |
| Expert drivers | xGESVX, xPOSVX, xSYSVX | Combines equilibration + solve + refinement |
| Equilibration | xGEEQU, xPOEQU | Auxiliary to expert drivers |
| Condition estimation | xPOCON, xSYCON | Extends Phase 10 GECON pattern |
| Band/tridiagonal eigenvalue | xSTEQR, xSTEBZ, xSBEV | Specialized decompositions |
| Generalized SVD | xGGSVD3 | Uncommon in practice |
| Generalized nonsymmetric eigen | xGGES, xGGEV (QZ) | Complex error analysis |
| Cholesky with pivoting | xPSTRF | Specialized factorization |
| Complete orthogonal decomp | xGELSY | Alternative least-squares |
| Triangular eigenvectors | xTREVC | Post-Schur computation |

---

## 6. Technical Notes

### Fortran calling convention

All BLAS/LAPACK routines use the Fortran calling convention:
- All arguments passed by pointer.
- Character arguments (`transa`, `uplo`, etc.) are `const char *`.
- **Hidden character lengths** are appended after all explicit arguments.
  The type depends on the Fortran compiler:
  - **gfortran >= 8**: `size_t` (adopted from Fortran 2018 CFI standard).
  - **gfortran < 8, ifort/ifx, flang**: `int`.
  - On LP64 platforms (x86-64), this difference is harmless because
    integer arguments are widened to 64-bit registers. On 32-bit
    platforms or unusual ABIs it may cause issues.
  - Current code uses `std::size_t`. If portability to older compilers
    is needed, add a compile-time option (`-DCHARLEN_TYPE=int`) or a
    runtime flag (`--charlen-type`).
  - **Fix**: All call sites must use `(std::size_t)1` (not bare `1`
    which is `int`) for hidden character-length values.
- Integer arguments are Fortran `INTEGER`, typically C `int` (32-bit).
  ILP64 builds use 64-bit integers; support via a compile-time flag
  if needed.

**Scalar return values** (Level 1):
- `double`/`float` returns: Returned in a floating-point register
  (XMM0 on x86-64, D0 on aarch64). Straightforward on all platforms.
- `int` returns (IAMAX): Returned in an integer register. Straightforward.
- Complex returns (ZDOTC, ZDOTU): Platform-dependent:
  - gfortran/Linux: Hidden first argument (`void *` to receive result).
  - macOS SysV ABI: Returned in `xmm0`/`xmm1`.
  - Solution: Provide two function pointer typedefs per complex-return
    routine and detect the convention, or require the user to specify
    via `--return-convention {register,hidden}`.

### XERBLA handling

The tester always provides valid parameters to the routines under test.
Negative testing (intentionally invalid inputs to verify XERBLA behavior
and correct INFO codes) is out of scope for this project.

### Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux/gfortran** | Primary target | Full support. CI runs here. |
| **macOS/Homebrew GCC** | Supported | `dlopen`/`dlsym` work. `__float128` NOT available on Apple Silicon (aarch64). `reference_quad` example is Linux/x86_64 only. Testing against Accelerate framework requires CBLAS interface (C symbol names, no hidden char lengths). |
| **Windows** | Out of scope | Would require `LoadLibrary`/`GetProcAddress` abstraction. Not planned. |

**Portability notes**:
- Replace `rand_r()` (deprecated POSIX) with `std::mt19937` in Phase 1.
- `__float128` and `mpfr_set_float128`/`mpfr_get_float128` require
  `MPFR_WANT_FLOAT128` and are x86_64-only. The `reference_quad`
  example is scoped to Linux/x86_64/gfortran only.

### Quad-precision Fortran examples

The `examples/reference_quad/` directory currently has `qgemm.f90` and
`qtrsm.f90`. These are **examples of implementations to test**, not part
of the testing infrastructure. Scope:
- Keep existing Level 3 examples (GEMM, TRSM).
- Optionally add more Level 3 routines (SYMM, TRMM, etc.) by adapting
  Netlib BLAS source (`DOUBLE PRECISION` -> `REAL(16)`).
- Do NOT attempt full-BLAS quad-precision coverage. The double-precision
  examples against reference BLAS are the primary correctness validation.

### Output format

**Text** (default, current format):
```
[GEMM transa=N transb=T] max_rel=2.05e-12  normwise=2.77e-16  sentinel=ok
[GEMM transa=N transb=T] max_rel=2.05e-12  normwise=2.77e-16  sentinel=FAIL(3)
```

When `ld_pad > 0`, `|inc| > 1`, or the routine has a triangle-only output,
`sentinel=ok` confirms no out-of-bounds writes. `sentinel=FAIL(n)` reports
`n` corrupted positions.

**LAPACK extension** (routine-specific metrics):
```
[GETRF n=64] factor_residual=1.23e-16  structural=PASS  sentinel=ok
[SYEV  n=64] eigenvalue_residual=3.45e-16  orthogonality=1.23e-16
```

**JSON** (`--format json`): Machine-readable for CI dashboards and
automated analysis.

**CSV** (`--format csv`): For importing into spreadsheets.

All output goes through `report_result()` (Section 3.4) so format
switching is localized.

### CI / Regression testing

Established in Phase 1. The CI pipeline:
1. Builds `linalg_tester` on Ubuntu with `libblas3`, `libmpfr-dev`, `libgmp-dev`.
2. Compiles `examples/openblas_double/double_conv.c` into `double_conv.so`.
3. Runs all implemented routines against reference BLAS with
   `--threshold 1e-12 --seed 42 --m 32 --n 32 --k 32`.
4. Exit code 0 = all errors below threshold = CI passes.
5. Updated each phase as new routines are added.
