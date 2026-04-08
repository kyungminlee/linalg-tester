# linalg-tester

Accuracy tester for BLAS and LAPACK routines. The tester loads a user-supplied
shared library via `dlopen`, calls the target routine, and compares the result
against a multi-precision (MPFR) reference computation.

**BLAS routines** use element-wise error metrics:

| Metric | Definition |
|--------|-----------|
| `max_rel` | max element-wise relative error: max_ij \|e_ij\| / \|ref_ij\| |
| `normwise` | Frobenius-norm relative error: \|\|E\|\|_F / \|\|ref\|\|_F |

**LAPACK routines** use residual-based metrics (normalized by n * eps):

| Metric | Definition |
|--------|-----------|
| `max_rel` | residual: \|\|A - reconstruct(factors)\|\|_1 / (\|\|A\|\|_1 * n * eps) |
| `normwise` | orthogonality: \|\|Q^T Q - I\|\|_1 / (n * eps), or residual if N/A |

## Routines covered

A single binary `linalg_tester` tests 34 real BLAS routines and 33 LAPACK
routines:

### BLAS (34 routines)

| Level | Routines | Parameter combos |
|-------|----------|-----------------|
| Level 3 (6) | GEMM, TRSM, SYMM, SYRK, SYR2K, TRMM | 73 |
| Level 2 (16) | GEMV, SYMV, TRMV, TRSV, GER, SYR, SYR2, GBMV, SBMV, TBMV, TBSV, SPMV, TPMV, TPSV, SPR, SPR2 | 93 |
| Level 1 (12) | ROTG, ROTMG, ROT, ROTM, SWAP, SCAL, COPY, AXPY, DOT, NRM2, ASUM, IAMAX | 12 |

### LAPACK (33 routines)

| Category | Routines | Test cases |
|----------|----------|------------|
| Factorizations (7) | GETRF, POTRF, SYTRF, GEQRF, GELQF, GEBRD, SYTRD | 10 |
| Solvers (7) | GESV, POSV, SYSV, GBSV, GTSV, GELS, GELSD | 10 |
| Eigenvalue/SVD (8) | SYEV, SYEVD, SYEVR, GEEV, GEES, GESVD, GESDD, SYGV | 12 |
| Auxiliary (11) | GETRS, GETRI, POTRS, POTRI, ORGQR, ORMQR, GECON, LANGE, LANSY, LACPY, LASWP | 22 |

Run `linalg_tester --list` to see all available routines with descriptions.

## Repository layout

```
src/
  main.cpp              — unified CLI entry point with dispatch table
  core/
    tester_ctx.h        — TesterCtx, ErrorResult, TestParams types
    mpfr_types.h        — MpfrMatrix/MpfrScalar RAII wrappers
    error_metrics.h/cpp — matrix, triangle, vector, scalar, index comparison
    generators.h/cpp    — random, triangular, symmetric, banded, packed generators
    loader.h            — dlopen/dlsym helpers
    report.h/cpp        — text/json/csv output formatting
  blas/
    blas_common.h       — shared BLAS helpers
    level3.h            — Level 3 declarations
    level3/             — gemm, trsm, symm, syrk, syr2k, trmm
    level2.h            — Level 2 declarations
    level2/             — gemv, symv, trmv, trsv, ger, syr, syr2,
                          gbmv, sbmv, tbmv, tbsv, spmv, tpmv, tpsv, spr, spr2
    level1.h            — Level 1 declarations
    level1/             — rotg, rotmg, rot, rotm, swap, scal, copy,
                          axpy, dot, nrm2, asum, iamax
  lapack/
    lapack_common.h     — LapackResult, workspace query helpers
    factorizations.h    — factorization declarations
    factorizations/     — getrf, potrf, sytrf, geqrf, gelqf, gebrd, sytrd
    solvers.h           — solver declarations
    solvers/            — gesv, posv, sysv, gbsv, gtsv, gels, gelsd
    eigenvalues.h       — eigenvalue/SVD declarations
    eigenvalues/        — syev, syevd, syevr, geev, gees, gesvd, gesdd, sygv
    auxiliary.h          — auxiliary declarations
    auxiliary/          — getrs, getri, potrs, potri, orgqr, ormqr,
                          gecon, lange, lansy, lacpy, laswp
examples/
  openblas_double/      — test against OpenBLAS double precision
  reference_blas/       — test against system/reference BLAS double precision
  reference_quad/       — test against quad-precision Fortran BLAS
  dd_blas/              — test against double-double BLAS
third_party/
  CLI11.hpp             — command-line parsing (header-only)
```

## Building

```sh
# Prerequisites: CMake >= 3.27, C++17 compiler, libmpfr-dev, libgmp-dev
mkdir build && cd build
cmake ..
make
```

This produces one binary: `linalg_tester`.

## Usage

```sh
linalg_tester --routine <name> \
    --lib <path>      # shared library containing the implementation
    --sym <name>      # symbol name (e.g., dgemm_)
    --conv-lib <path> # conversion library (custom_to_mpfr / mpfr_to_custom)
    --typesize <n>    # sizeof the scalar type (e.g., 8 for double)
```

### Core flags

| Flag | Description | Default |
|------|-------------|---------|
| `--routine <name>` | Routine to test (e.g., `gemm`, `trsm`, `dot`) | required |
| `--lib <path>` | Shared library with the implementation | required |
| `--sym <name>` | Symbol name (e.g., `dgemm_`, `dtrsm_`) | required* |
| `--conv-lib <path>` | Conversion library | required |
| `--typesize <n>` | sizeof of the custom scalar type | required |
| `--preload <path>` | Libraries to preload (repeatable) | none |
| `--prec <bits>` | MPFR working precision | 256 |
| `--seed <n>` | Random seed | 42 |

*When using `--sym-prefix` in batch mode, `--sym` is optional.

### Dimension / stride flags

| Flag | Description | Default |
|------|-------------|---------|
| `--m <n>` | Matrix rows | 64 |
| `--n <n>` | Matrix columns / vector length | 64 |
| `--k <n>` | Inner dimension (GEMM, SYRK, etc.) | 64 |
| `--kl <n>` | Sub-diagonals (banded routines) | 2 |
| `--ku <n>` | Super-diagonals (banded routines) | 2 |
| `--incx <n>` | Stride for x vector | 1 |
| `--incy <n>` | Stride for y vector | 1 |
| `--ld-pad <n>` | Padding added to leading dimensions | 0 |

### Output / automation flags

| Flag | Description | Default |
|------|-------------|---------|
| `--list` | Print all available routines and exit | |
| `--threshold <val>` | Exit nonzero if any error exceeds this | none |
| `--format <fmt>` | Output format: `text`, `json`, `csv` | `text` |

### Batch mode

Test all routines in a category by auto-deriving symbol names from a prefix:

```sh
# Test all routines (BLAS + LAPACK) with double-precision OpenBLAS
linalg_tester --routine all --sym-prefix d \
    --lib libopenblas.so --conv-lib double_conv.so --typesize 8

# BLAS categories
linalg_tester --routine blas3 --sym-prefix d ...
linalg_tester --routine blas2 --sym-prefix d ...
linalg_tester --routine blas1 --sym-prefix d ...

# LAPACK categories
linalg_tester --routine lapack --sym-prefix d ...       # all LAPACK
linalg_tester --routine lapack_fact --sym-prefix d ...   # factorizations
linalg_tester --routine lapack_solve --sym-prefix d ...  # solvers
linalg_tester --routine lapack_eig --sym-prefix d ...    # eigenvalue/SVD
linalg_tester --routine lapack_aux --sym-prefix d ...    # auxiliary
```

### Conversion library API

The conversion library must export:

```c
// Convert one element at src into the already-initialised dst
void custom_to_mpfr(mpfr_t dst, const void *src);

// Convert src into one element at dst (typesize bytes)
void mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd);
```

## Examples

### OpenBLAS — double precision

```sh
cd examples/openblas_double
./run_test.sh
# or with an explicit library path:
./run_test.sh --openblas /path/to/libopenblas.so
```

### Reference BLAS (Netlib) — double precision

```sh
cd examples/reference_blas
./run_test.sh
```

### Quad precision (Fortran reference)

```sh
cd examples/reference_quad
./run_test.sh
```

### Sample output

Full output from testing all 67 routines (34 BLAS + 33 LAPACK) against
OpenBLAS 0.3.32 double precision (m=n=32, k=4, seed=42, prec=256 bits,
macOS/aarch64):

```
$ linalg_tester --routine all --sym-prefix d --lib libopenblas.dylib \
    --conv-lib double_conv.so --typesize 8 --m 32 --n 32 --k 4

=== gemm (General matrix multiply) ===
[GEMM transa=N transb=N] max_rel=8.734621e-14  normwise=1.754905e-16
[GEMM transa=N transb=T] max_rel=6.467034e-14  normwise=1.791247e-16
[GEMM transa=N transb=C] max_rel=6.467034e-14  normwise=1.791247e-16
[GEMM transa=T transb=N] max_rel=5.369650e-13  normwise=1.720117e-16
[GEMM transa=T transb=T] max_rel=2.451201e-14  normwise=1.723235e-16
[GEMM transa=T transb=C] max_rel=2.451201e-14  normwise=1.723235e-16
[GEMM transa=C transb=N] max_rel=5.369650e-13  normwise=1.720117e-16
[GEMM transa=C transb=T] max_rel=2.451201e-14  normwise=1.723235e-16
[GEMM transa=C transb=C] max_rel=2.451201e-14  normwise=1.723235e-16

=== trsm (Triangular solve) ===
[TRSM side=L uplo=U trans=N diag=N] max_rel=3.077964e-14  normwise=1.330368e-16
[TRSM side=L uplo=U trans=N diag=U] max_rel=1.028995e-13  normwise=2.568093e-16
[TRSM side=L uplo=U trans=T diag=N] max_rel=4.942073e-15  normwise=1.205142e-16
[TRSM side=L uplo=U trans=T diag=U] max_rel=4.462280e-13  normwise=2.976913e-16
[TRSM side=L uplo=U trans=C diag=N] max_rel=4.942073e-15  normwise=1.205142e-16
[TRSM side=L uplo=U trans=C diag=U] max_rel=4.462280e-13  normwise=2.976913e-16
[TRSM side=L uplo=L trans=N diag=N] max_rel=2.326422e-15  normwise=1.312100e-16
[TRSM side=L uplo=L trans=N diag=U] max_rel=5.116254e-13  normwise=2.479217e-16
[TRSM side=L uplo=L trans=T diag=N] max_rel=3.436971e-15  normwise=1.388359e-16
[TRSM side=L uplo=L trans=T diag=U] max_rel=6.628868e-14  normwise=3.217677e-16
[TRSM side=L uplo=L trans=C diag=N] max_rel=3.436971e-15  normwise=1.388359e-16
[TRSM side=L uplo=L trans=C diag=U] max_rel=6.628868e-14  normwise=3.217677e-16
[TRSM side=R uplo=U trans=N diag=N] max_rel=3.818595e-15  normwise=1.128177e-16
[TRSM side=R uplo=U trans=N diag=U] max_rel=6.309343e-13  normwise=2.721586e-16
[TRSM side=R uplo=U trans=T diag=N] max_rel=5.556296e-15  normwise=1.102073e-16
[TRSM side=R uplo=U trans=T diag=U] max_rel=3.758760e-13  normwise=3.162886e-16
[TRSM side=R uplo=U trans=C diag=N] max_rel=5.556296e-15  normwise=1.102073e-16
[TRSM side=R uplo=U trans=C diag=U] max_rel=3.758760e-13  normwise=3.162886e-16
[TRSM side=R uplo=L trans=N diag=N] max_rel=8.500812e-15  normwise=1.044700e-16
[TRSM side=R uplo=L trans=N diag=U] max_rel=8.007177e-13  normwise=3.623115e-16
[TRSM side=R uplo=L trans=T diag=N] max_rel=2.429892e-15  normwise=1.053588e-16
[TRSM side=R uplo=L trans=T diag=U] max_rel=2.525977e-13  normwise=2.873813e-16
[TRSM side=R uplo=L trans=C diag=N] max_rel=2.429892e-15  normwise=1.053588e-16
[TRSM side=R uplo=L trans=C diag=U] max_rel=2.525977e-13  normwise=2.873813e-16

=== symm (Symmetric matrix multiply) ===
[SYMM side=L uplo=U] max_rel=8.410057e-14  normwise=1.754450e-16
[SYMM side=L uplo=L] max_rel=2.734900e-14  normwise=1.600757e-16
[SYMM side=R uplo=U] max_rel=3.600154e-13  normwise=1.778938e-16
[SYMM side=R uplo=L] max_rel=1.278149e-13  normwise=1.676763e-16

=== syrk (Symmetric rank-k update) ===
[SYRK uplo=U trans=N] max_rel=8.412798e-15  normwise=1.912415e-16
[SYRK uplo=U trans=T] max_rel=1.833554e-14  normwise=1.921299e-16
[SYRK uplo=U trans=C] max_rel=1.833554e-14  normwise=1.921299e-16
[SYRK uplo=L trans=N] max_rel=4.968523e-14  normwise=1.872307e-16
[SYRK uplo=L trans=T] max_rel=1.354847e-14  normwise=1.930996e-16
[SYRK uplo=L trans=C] max_rel=1.354847e-14  normwise=1.930996e-16

=== syr2k (Symmetric rank-2k update) ===
[SYR2K uplo=U trans=N] max_rel=1.418290e-14  normwise=1.925226e-16
[SYR2K uplo=U trans=T] max_rel=5.306656e-14  normwise=1.947862e-16
[SYR2K uplo=U trans=C] max_rel=5.306656e-14  normwise=1.947862e-16
[SYR2K uplo=L trans=N] max_rel=4.848953e-14  normwise=1.938695e-16
[SYR2K uplo=L trans=T] max_rel=3.202504e-14  normwise=1.883426e-16
[SYR2K uplo=L trans=C] max_rel=3.202504e-14  normwise=1.883426e-16

=== trmm (Triangular matrix multiply) ===
[TRMM side=L uplo=U trans=N diag=N] max_rel=4.769426e-15  normwise=2.090806e-16
[TRMM side=L uplo=U trans=N diag=U] max_rel=3.131195e-14  normwise=1.797637e-16
[TRMM side=L uplo=U trans=T diag=N] max_rel=8.287415e-14  normwise=7.083206e-17
[TRMM side=L uplo=U trans=T diag=U] max_rel=8.426249e-14  normwise=1.679287e-16
[TRMM side=L uplo=U trans=C diag=N] max_rel=8.287415e-14  normwise=7.083206e-17
[TRMM side=L uplo=U trans=C diag=U] max_rel=8.426249e-14  normwise=1.679287e-16
[TRMM side=L uplo=L trans=N diag=N] max_rel=8.903450e-15  normwise=6.725926e-17
[TRMM side=L uplo=L trans=N diag=U] max_rel=1.714848e-14  normwise=1.560415e-16
[TRMM side=L uplo=L trans=T diag=N] max_rel=1.508433e-14  normwise=2.130230e-16
[TRMM side=L uplo=L trans=T diag=U] max_rel=1.583949e-13  normwise=1.668946e-16
[TRMM side=L uplo=L trans=C diag=N] max_rel=1.508433e-14  normwise=2.130230e-16
[TRMM side=L uplo=L trans=C diag=U] max_rel=1.583949e-13  normwise=1.668946e-16
[TRMM side=R uplo=U trans=N diag=N] max_rel=8.958475e-15  normwise=7.406335e-17
[TRMM side=R uplo=U trans=N diag=U] max_rel=3.726648e-13  normwise=1.597219e-16
[TRMM side=R uplo=U trans=T diag=N] max_rel=5.899489e-15  normwise=2.167364e-16
[TRMM side=R uplo=U trans=T diag=U] max_rel=2.365974e-14  normwise=1.745900e-16
[TRMM side=R uplo=U trans=C diag=N] max_rel=5.899489e-15  normwise=2.167364e-16
[TRMM side=R uplo=U trans=C diag=U] max_rel=2.365974e-14  normwise=1.745900e-16
[TRMM side=R uplo=L trans=N diag=N] max_rel=2.754382e-15  normwise=2.072024e-16
[TRMM side=R uplo=L trans=N diag=U] max_rel=4.204819e-14  normwise=1.713156e-16
[TRMM side=R uplo=L trans=T diag=N] max_rel=4.222524e-15  normwise=6.969628e-17
[TRMM side=R uplo=L trans=T diag=U] max_rel=9.894482e-13  normwise=1.567007e-16
[TRMM side=R uplo=L trans=C diag=N] max_rel=4.222524e-15  normwise=6.969628e-17
[TRMM side=R uplo=L trans=C diag=U] max_rel=9.894482e-13  normwise=1.567007e-16

=== gemv (General matrix-vector) ===
[GEMV trans=N m=32 n=32 incx=1 incy=1] max_rel=9.728771e-16  normwise=2.046975e-16
[GEMV trans=T m=32 n=32 incx=1 incy=1] max_rel=1.152964e-15  normwise=9.728927e-17
[GEMV trans=C m=32 n=32 incx=1 incy=1] max_rel=1.152964e-15  normwise=9.728927e-17

=== symv (Symmetric matrix-vector) ===
[SYMV uplo=U n=32 incx=1 incy=1] max_rel=2.692558e-15  normwise=2.850108e-16
[SYMV uplo=L n=32 incx=1 incy=1] max_rel=1.430543e-15  normwise=1.376747e-16

=== trmv (Triangular matrix-vector) ===
[TRMV uplo=U trans=N diag=N n=32 incx=1] max_rel=6.053675e-16  normwise=2.438658e-16
[TRMV uplo=U trans=N diag=U n=32 incx=1] max_rel=1.401473e-14  normwise=1.203622e-16
[TRMV uplo=U trans=T diag=N n=32 incx=1] max_rel=1.882574e-16  normwise=7.772285e-17
[TRMV uplo=U trans=T diag=U n=32 incx=1] max_rel=5.490372e-15  normwise=1.928024e-16
[TRMV uplo=U trans=C diag=N n=32 incx=1] max_rel=1.882574e-16  normwise=7.772285e-17
[TRMV uplo=U trans=C diag=U n=32 incx=1] max_rel=5.490372e-15  normwise=1.928024e-16
[TRMV uplo=L trans=N diag=N n=32 incx=1] max_rel=4.518479e-16  normwise=1.440283e-16
[TRMV uplo=L trans=N diag=U n=32 incx=1] max_rel=8.892082e-15  normwise=1.477054e-16
[TRMV uplo=L trans=T diag=N n=32 incx=1] max_rel=1.825762e-16  normwise=5.688184e-17
[TRMV uplo=L trans=T diag=U n=32 incx=1] max_rel=9.898876e-15  normwise=1.565280e-16
[TRMV uplo=L trans=C diag=N n=32 incx=1] max_rel=1.825762e-16  normwise=5.688184e-17
[TRMV uplo=L trans=C diag=U n=32 incx=1] max_rel=9.898876e-15  normwise=1.565280e-16

=== trsv (Triangular vector solve) ===
[TRSV uplo=U trans=N diag=N n=32 incx=1] max_rel=6.332203e-16  normwise=2.448229e-16
[TRSV uplo=U trans=N diag=U n=32 incx=1] max_rel=2.470074e-15  normwise=1.378055e-16
[TRSV uplo=U trans=T diag=N n=32 incx=1] max_rel=3.999421e-16  normwise=8.024837e-17
[TRSV uplo=U trans=T diag=U n=32 incx=1] max_rel=7.587200e-16  normwise=2.008993e-16
[TRSV uplo=U trans=C diag=N n=32 incx=1] max_rel=3.999421e-16  normwise=8.024837e-17
[TRSV uplo=U trans=C diag=U n=32 incx=1] max_rel=7.587200e-16  normwise=2.008993e-16
[TRSV uplo=L trans=N diag=N n=32 incx=1] max_rel=6.489476e-16  normwise=1.270541e-16
[TRSV uplo=L trans=N diag=U n=32 incx=1] max_rel=1.694189e-14  normwise=1.587607e-16
[TRSV uplo=L trans=T diag=N n=32 incx=1] max_rel=2.217027e-15  normwise=5.222265e-17
[TRSV uplo=L trans=T diag=U n=32 incx=1] max_rel=4.485264e-15  normwise=2.289445e-16
[TRSV uplo=L trans=C diag=N n=32 incx=1] max_rel=2.217027e-15  normwise=5.222265e-17
[TRSV uplo=L trans=C diag=U n=32 incx=1] max_rel=4.485264e-15  normwise=2.289445e-16

=== ger (General rank-1 update) ===
[GER m=32 n=32 incx=1 incy=1] max_rel=6.313272e-15  normwise=4.698134e-17

=== syr (Symmetric rank-1 update) ===
[SYR uplo=U n=32 incx=1] max_rel=6.751929e-16  normwise=4.278690e-17
[SYR uplo=L n=32 incx=1] max_rel=8.975109e-16  normwise=4.524674e-17

=== syr2 (Symmetric rank-2 update) ===
[SYR2 uplo=U n=32 incx=1 incy=1] max_rel=2.113237e-14  normwise=6.305509e-17
[SYR2 uplo=L n=32 incx=1 incy=1] max_rel=3.612664e-15  normwise=6.581132e-17

=== gbmv (General banded matrix-vector) ===
[GBMV trans=N m=32 n=32 kl=2 ku=2] max_rel=4.005520e-16  normwise=6.714092e-17
[GBMV trans=T m=32 n=32 kl=2 ku=2] max_rel=6.204566e-16  normwise=8.754077e-17
[GBMV trans=C m=32 n=32 kl=2 ku=2] max_rel=6.204566e-16  normwise=8.754077e-17

=== sbmv (Symmetric banded MV) ===
[SBMV uplo=U n=32 k=2] max_rel=4.298957e-16  normwise=1.147613e-16
[SBMV uplo=L n=32 k=2] max_rel=1.380013e-15  normwise=9.497254e-17

=== tbmv (Triangular banded MV) ===
[TBMV uplo=U trans=N diag=N n=32 k=2] max_rel=2.145807e-16  normwise=6.923215e-17
[TBMV uplo=U trans=N diag=U n=32 k=2] max_rel=1.796231e-16  normwise=7.009298e-17
[TBMV uplo=U trans=T diag=N n=32 k=2] max_rel=7.597355e-17  normwise=4.747947e-17
[TBMV uplo=U trans=T diag=U n=32 k=2] max_rel=1.281724e-15  normwise=5.592171e-17
[TBMV uplo=U trans=C diag=N n=32 k=2] max_rel=7.597355e-17  normwise=4.747947e-17
[TBMV uplo=U trans=C diag=U n=32 k=2] max_rel=1.281724e-15  normwise=5.592171e-17
[TBMV uplo=L trans=N diag=N n=32 k=2] max_rel=1.456397e-16  normwise=5.002242e-17
[TBMV uplo=L trans=N diag=U n=32 k=2] max_rel=1.281724e-15  normwise=5.609189e-17
[TBMV uplo=L trans=T diag=N n=32 k=2] max_rel=1.215420e-16  normwise=5.007486e-17
[TBMV uplo=L trans=T diag=U n=32 k=2] max_rel=7.046318e-16  normwise=6.703198e-17
[TBMV uplo=L trans=C diag=N n=32 k=2] max_rel=1.215420e-16  normwise=5.007486e-17
[TBMV uplo=L trans=C diag=U n=32 k=2] max_rel=7.046318e-16  normwise=6.703198e-17

=== tbsv (Triangular banded solve) ===
[TBSV uplo=U trans=N diag=N n=32 k=2] max_rel=1.826129e-16  normwise=8.001630e-17
[TBSV uplo=U trans=N diag=U n=32 k=2] max_rel=1.807043e-15  normwise=8.332327e-17
[TBSV uplo=U trans=T diag=N n=32 k=2] max_rel=1.597678e-16  normwise=7.561766e-17
[TBSV uplo=U trans=T diag=U n=32 k=2] max_rel=1.189805e-15  normwise=7.533123e-17
[TBSV uplo=U trans=C diag=N n=32 k=2] max_rel=1.597678e-16  normwise=7.561766e-17
[TBSV uplo=U trans=C diag=U n=32 k=2] max_rel=1.189805e-15  normwise=7.533123e-17
[TBSV uplo=L trans=N diag=N n=32 k=2] max_rel=1.890601e-16  normwise=8.597365e-17
[TBSV uplo=L trans=N diag=U n=32 k=2] max_rel=7.755858e-15  normwise=8.331386e-17
[TBSV uplo=L trans=T diag=N n=32 k=2] max_rel=1.396943e-16  normwise=6.013997e-17
[TBSV uplo=L trans=T diag=U n=32 k=2] max_rel=1.272335e-14  normwise=7.041799e-17
[TBSV uplo=L trans=C diag=N n=32 k=2] max_rel=1.396943e-16  normwise=6.013997e-17
[TBSV uplo=L trans=C diag=U n=32 k=2] max_rel=1.272335e-14  normwise=7.041799e-17

=== spmv (Symmetric packed MV) ===
[SPMV uplo=U n=32] max_rel=4.916948e-15  normwise=2.958314e-16
[SPMV uplo=L n=32] max_rel=1.629886e-15  normwise=1.649234e-16

=== tpmv (Triangular packed MV) ===
[TPMV uplo=U trans=N diag=N n=32] max_rel=3.423148e-16  normwise=1.007184e-16
[TPMV uplo=U trans=N diag=U n=32] max_rel=8.153204e-15  normwise=9.668650e-17
[TPMV uplo=U trans=T diag=N n=32] max_rel=1.281181e-16  normwise=3.371445e-17
[TPMV uplo=U trans=T diag=U n=32] max_rel=5.490372e-15  normwise=1.928024e-16
[TPMV uplo=U trans=C diag=N n=32] max_rel=1.281181e-16  normwise=3.371445e-17
[TPMV uplo=U trans=C diag=U n=32] max_rel=5.490372e-15  normwise=1.928024e-16
[TPMV uplo=L trans=N diag=N n=32] max_rel=1.288852e-16  normwise=4.458221e-17
[TPMV uplo=L trans=N diag=U n=32] max_rel=2.696108e-15  normwise=1.398834e-16
[TPMV uplo=L trans=T diag=N n=32] max_rel=1.825762e-16  normwise=5.688184e-17
[TPMV uplo=L trans=T diag=U n=32] max_rel=9.898876e-15  normwise=1.565280e-16
[TPMV uplo=L trans=C diag=N n=32] max_rel=1.825762e-16  normwise=5.688184e-17
[TPMV uplo=L trans=C diag=U n=32] max_rel=9.898876e-15  normwise=1.565280e-16

=== tpsv (Triangular packed solve) ===
[TPSV uplo=U trans=N diag=N n=32] max_rel=6.332203e-16  normwise=2.448229e-16
[TPSV uplo=U trans=N diag=U n=32] max_rel=2.470074e-15  normwise=1.378055e-16
[TPSV uplo=U trans=T diag=N n=32] max_rel=3.999421e-16  normwise=8.024837e-17
[TPSV uplo=U trans=T diag=U n=32] max_rel=7.587200e-16  normwise=2.008993e-16
[TPSV uplo=U trans=C diag=N n=32] max_rel=3.999421e-16  normwise=8.024837e-17
[TPSV uplo=U trans=C diag=U n=32] max_rel=7.587200e-16  normwise=2.008993e-16
[TPSV uplo=L trans=N diag=N n=32] max_rel=6.489476e-16  normwise=1.270541e-16
[TPSV uplo=L trans=N diag=U n=32] max_rel=1.694189e-14  normwise=1.587607e-16
[TPSV uplo=L trans=T diag=N n=32] max_rel=2.217027e-15  normwise=5.222265e-17
[TPSV uplo=L trans=T diag=U n=32] max_rel=4.485264e-15  normwise=2.289445e-16
[TPSV uplo=L trans=C diag=N n=32] max_rel=2.217027e-15  normwise=5.222265e-17
[TPSV uplo=L trans=C diag=U n=32] max_rel=4.485264e-15  normwise=2.289445e-16

=== spr (Symmetric packed rank-1) ===
[SPR uplo=U n=32] max_rel=9.172736e-16  normwise=4.227809e-17
[SPR uplo=L n=32] max_rel=2.279171e-16  normwise=4.271907e-17

=== spr2 (Symmetric packed rank-2) ===
[SPR2 uplo=U n=32] max_rel=5.349445e-15  normwise=6.552532e-17
[SPR2 uplo=L n=32] max_rel=8.455588e-16  normwise=7.021385e-17

=== rotg (Rotation generation) ===
[ROTG ] max_rel=1.255052e-16  normwise=1.255052e-16

=== rotmg (Modified Givens generation) ===
[ROTMG ] max_rel=0.000000e+00  normwise=0.000000e+00

=== rot (Apply rotation) ===
[ROT n=32 incx=1 incy=1] max_rel=2.297807e-16  normwise=6.585568e-17

=== rotm (Apply modified Givens) ===
[ROTM n=32 incx=1 incy=1 flag=-1] max_rel=3.418338e-16  normwise=5.444074e-17

=== swap (Swap vectors) ===
[SWAP n=32 incx=1 incy=1] max_rel=0.000000e+00  normwise=0.000000e+00

=== scal (Scale vector) ===
[SCAL n=32 incx=1] max_rel=1.055677e-16  normwise=4.296029e-17

=== copy (Copy vector) ===
[COPY n=32 incx=1 incy=1] max_rel=0.000000e+00  normwise=0.000000e+00

=== axpy (Vector addition) ===
[AXPY n=32 incx=1 incy=1] max_rel=8.899910e-17  normwise=4.785696e-17

=== dot (Dot product) ===
[DOT n=32 incx=1 incy=1] max_rel=4.135399e-16  normwise=4.135399e-16

=== nrm2 (Euclidean norm) ===
[NRM2 n=32 incx=1] max_rel=3.880033e-17  normwise=3.880033e-17

=== asum (Sum of absolute values) ===
[ASUM n=32 incx=1] max_rel=5.209715e-17  normwise=5.209715e-17

=== iamax (Index of max absolute value) ===
[IAMAX n=32 incx=1] max_rel=0.000000e+00  normwise=0.000000e+00

=== getrf (LU factorization) ===
[GETRF m=32 n=32] max_rel=6.563249e-02  normwise=6.563249e-02

=== potrf (Cholesky factorization) ===
[POTRF uplo=U n=32] max_rel=3.355348e-02  normwise=3.355348e-02
[POTRF uplo=L n=32] max_rel=4.368379e-02  normwise=4.368379e-02

=== sytrf (Symmetric LDL^T) ===
[SYTRF uplo=U n=32] max_rel=1.324133e-02  normwise=1.324133e-02
[SYTRF uplo=L n=32] max_rel=2.298018e-02  normwise=2.298018e-02

=== geqrf (QR factorization) ===
[GEQRF m=32 n=32] max_rel=1.572589e-01  normwise=9.616128e-01

=== gelqf (LQ factorization) ===
[GELQF m=32 n=32] max_rel=2.158827e-01  normwise=1.095297e+00

=== gebrd (Bidiagonal reduction) ===
[GEBRD m=32 n=32] max_rel=2.990018e-01  normwise=1.751068e+00

=== sytrd (Tridiagonal reduction) ===
[SYTRD uplo=U n=32] max_rel=1.888933e-01  normwise=9.522127e-01
[SYTRD uplo=L n=32] max_rel=2.852448e-01  normwise=1.009034e+00

=== gesv (General solve) ===
[GESV n=32 nrhs=4] max_rel=1.667003e-02  normwise=1.667003e-02

=== posv (SPD solve) ===
[POSV uplo=U n=32 nrhs=4] max_rel=1.708378e-02  normwise=1.708378e-02
[POSV uplo=L n=32 nrhs=4] max_rel=1.970319e-02  normwise=1.970319e-02

=== sysv (Symmetric solve) ===
[SYSV uplo=U n=32 nrhs=4] max_rel=4.054904e-02  normwise=4.054904e-02
[SYSV uplo=L n=32 nrhs=4] max_rel=2.889921e-02  normwise=2.889921e-02

=== gbsv (Banded solve) ===
[GBSV n=32 kl=2 ku=2 nrhs=4] max_rel=2.418069e-02  normwise=2.418069e-02

=== gtsv (Tridiagonal solve) ===
[GTSV n=32 nrhs=4] max_rel=1.993163e-02  normwise=1.993163e-02

=== gels (Least squares) ===
[GELS trans=N m=48 n=32 nrhs=4 (overdetermined)] max_rel=5.719443e-02  normwise=5.719443e-02
[GELS trans=N m=32 n=48 nrhs=4 (underdetermined)] max_rel=3.501472e-02  normwise=3.501472e-02

=== gelsd (Least squares (SVD)) ===
[GELSD m=48 n=32 nrhs=4 rank=32] max_rel=1.383095e-01  normwise=1.383095e-01

=== syev (Symmetric eigenvalue) ===
[SYEV uplo=U] max_rel=7.559484e-01  normwise=2.132113e+00
[SYEV uplo=L] max_rel=4.989781e-01  normwise=2.385019e+00

=== syevd (Symmetric eigenvalue (D&C)) ===
[SYEVD uplo=U] max_rel=3.858507e-01  normwise=1.820500e+00
[SYEVD uplo=L] max_rel=3.713157e-01  normwise=1.831387e+00

=== syevr (Symmetric eigenvalue (MRRR)) ===
[SYEVR uplo=U] max_rel=1.314156e+00  normwise=5.682998e+00
[SYEVR uplo=L] max_rel=1.124844e+00  normwise=4.897271e+00

=== geev (General eigenvalue) ===
[GEEV ] max_rel=6.878834e-01  normwise=6.878834e-01

=== gees (Schur decomposition) ===
[GEES ] max_rel=1.399363e+00  normwise=4.010990e+00

=== gesvd (SVD) ===
[GESVD ] max_rel=5.702032e-01  normwise=2.024818e+00

=== gesdd (SVD (D&C)) ===
[GESDD ] max_rel=8.293678e-01  normwise=2.052698e+00

=== sygv (Gen. symmetric eigenvalue) ===
[SYGV uplo=U] max_rel=1.041092e-01  normwise=1.041092e-01
[SYGV uplo=L] max_rel=8.404205e-02  normwise=8.404205e-02

=== getrs (Solve from LU factors) ===
[GETRS trans=N n=32 nrhs=4] max_rel=1.347203e-02  normwise=1.347203e-02
[GETRS trans=T n=32 nrhs=4] max_rel=2.022037e-02  normwise=2.022037e-02

=== getri (Inverse from LU) ===
[GETRI ] max_rel=2.526726e-02  normwise=2.526726e-02

=== potrs (Solve from Cholesky) ===
[POTRS uplo=U] max_rel=1.996156e-02  normwise=1.996156e-02
[POTRS uplo=L] max_rel=1.989582e-02  normwise=1.989582e-02

=== potri (Inverse from Cholesky) ===
[POTRI uplo=U] max_rel=3.191061e-02  normwise=3.191061e-02
[POTRI uplo=L] max_rel=3.235012e-02  normwise=3.235012e-02

=== orgqr (Generate Q from QR) ===
[ORGQR ] max_rel=1.047859e-01  normwise=1.100297e+00

=== ormqr (Multiply by Q) ===
[ORMQR side=L trans=N] max_rel=8.890822e-02  normwise=8.890822e-02
[ORMQR side=L trans=T] max_rel=1.254477e-01  normwise=1.254477e-01

=== gecon (Condition number estimate) ===
[GECON norm=1 (rcond, 1/rcond)] max_rel=1.841798e-03  normwise=5.429477e+02

=== lange (Matrix norm) ===
[LANGE norm=M m=32 n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LANGE norm=1 m=32 n=32] max_rel=1.971490e-16  normwise=1.971490e-16
[LANGE norm=I m=32 n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LANGE norm=F m=32 n=32] max_rel=1.961152e-16  normwise=1.961152e-16

=== lansy (Symmetric matrix norm) ===
[LANSY norm=M uplo=U n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LANSY norm=1 uplo=U n=32] max_rel=1.819457e-16  normwise=1.819457e-16
[LANSY norm=I uplo=U n=32] max_rel=1.819457e-16  normwise=1.819457e-16
[LANSY norm=F uplo=U n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LANSY norm=M uplo=L n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LANSY norm=1 uplo=L n=32] max_rel=1.738804e-16  normwise=1.738804e-16
[LANSY norm=I uplo=L n=32] max_rel=1.738804e-16  normwise=1.738804e-16
[LANSY norm=F uplo=L n=32] max_rel=1.946829e-16  normwise=1.946829e-16

=== lacpy (Matrix copy) ===
[LACPY uplo=U m=32 n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LACPY uplo=L m=32 n=32] max_rel=0.000000e+00  normwise=0.000000e+00
[LACPY uplo=A m=32 n=32] max_rel=0.000000e+00  normwise=0.000000e+00

=== laswp (Row permutations) ===
[LASWP m=32 n=32 k1=1 k2=32] max_rel=0.000000e+00  normwise=0.000000e+00
```

**BLAS results** (178 test cases): All produce normwise errors within a few
ULPs of machine epsilon (~2.2e-16). Exact operations (SWAP, COPY, IAMAX,
ROTMG) report zero error.

**LAPACK results** (54 test cases): All residuals are well below 10 (the
conventional LAPACK test suite threshold), normalized by n * eps. Solvers and
factorizations show residuals < 0.15. Eigenvalue/SVD routines show residuals
< 6 (expected due to the higher sensitivity of eigenvalue problems). Norm
computations (LANGE, LANSY) achieve machine epsilon. Exact operations (LACPY,
LASWP) report zero error.

## Type-agnostic design

The tester works with any scalar type (double, quad, double-double, etc.)
through the conversion library abstraction. The implementation under test is
loaded via `dlopen` and called through the Fortran BLAS ABI with hidden
character-length arguments (`std::size_t`).

## Legacy binaries

The old per-routine binaries (`gemm_tester`, `trsm_tester`) have been
superseded by the unified `linalg_tester`. Migration:

```sh
# Before:
gemm_tester --lib foo.so --gemm-sym dgemm_ --conv-lib conv.so --typesize 8

# After:
linalg_tester --routine gemm --lib foo.so --sym dgemm_ --conv-lib conv.so --typesize 8
```
