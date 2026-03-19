# linalg-tester

Accuracy tester for custom BLAS routines. Each tester loads a user-supplied
shared library, calls the target routine, and compares the result against a
multi-precision (MPFR) reference computation. Two error metrics are reported:

| Metric | Definition |
|--------|-----------|
| `max_rel` | max element-wise relative error: max_ij \|e_ij\| / \|ref_ij\| |
| `normwise` | Frobenius-norm relative error: \|\|E\|\|_F / \|\|ref\|\|_F |

## Routines covered

| Binary | BLAS routine | Combinations tested |
|--------|-------------|---------------------|
| `gemm_tester` | GEMM | all 4 `(transa, transb)` ∈ {N,T}² |
| `trsm_tester` | TRSM | all 16 `(side, uplo, trans, diag)` ∈ {L,R}×{U,L}×{N,T}×{N,U} |

## Repository layout

```
src/
  gemm_tester.cpp   — GEMM tester main
  trsm_tester.cpp   — TRSM tester main
  reference.cpp/h   — MPFR reference implementations
  tester_utils.h    — random-matrix generation, dlopen helpers
examples/
  openblas_double/  — test against OpenBLAS (libopenblas.so) double precision
  reference_blas/   — test against system/reference BLAS (libblas.so.3) double precision
  dd_blas/          — test against a double-double BLAS (libddblas.so, not included)
third_party/
  CLI11.hpp         — command-line parsing (header-only)
Makefile
```

## Building

```sh
# Prerequisites: g++, libmpfr-dev, libgmp-dev
make
```

This produces two binaries: `gemm_tester` and `trsm_tester`.

## Usage

Both testers share the same interface. You provide:

- `--lib <path>` — shared library containing the GEMM/TRSM implementation
- `--gemm-sym` / `--trsm-sym` — symbol name (e.g. `dgemm_`, `mytrsm_`)
- `--conv-lib <path>` — library exporting `custom_to_mpfr` / `mpfr_to_custom`
- `--typesize <n>` — `sizeof` of the scalar type (e.g. `8` for `double`)
- `--preload <path>` — additional libraries to open first (repeatable)
- `--m`, `--n`, `--k` — matrix dimensions (default 64)
- `--seed <n>` — random seed (default 42)
- `--prec <bits>` — MPFR working precision (default 256)

The conversion library must export:

```c
// Convert one element at src into the already-initialised dst
void custom_to_mpfr(mpfr_t dst, const void *src);

// Convert src into one element at dst (typesize bytes)
void mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd);
```

## Examples

### OpenBLAS — double precision

The `examples/openblas_double/` directory contains a ready-to-run test
against OpenBLAS `dgemm_` and `dtrsm_`.

**Prerequisites**

```sh
apt install libopenblas-dev libmpfr-dev libgmp-dev
make           # build the testers from the repo root
```

**Run**

```sh
cd examples/openblas_double
./run_test.sh
# or with an explicit library path:
./run_test.sh --openblas /path/to/libopenblas.so
```

The script:
1. Compiles `double_conv.so` (the `double` ↔ MPFR conversion library).
2. Auto-locates `libopenblas.so` (or uses the path you supply).
3. Runs `gemm_tester` then `trsm_tester`, printing error metrics for every combination.

**Sample output** (OpenBLAS 0.3.26, m=n=k=64, seed=42, prec=256 bits):

```
=== GEMM (dgemm_) — double precision, m=n=k=64 ===
[GEMM transa=N transb=N] max_rel=1.810189e-12  normwise=2.721062e-16
[GEMM transa=N transb=T] max_rel=2.050977e-12  normwise=2.768808e-16
[GEMM transa=T transb=N] max_rel=6.058418e-13  normwise=2.791001e-16
[GEMM transa=T transb=T] max_rel=1.369619e-13  normwise=2.760563e-16

=== TRSM (dtrsm_) — double precision, m=n=64 ===
[TRSM side=L uplo=U trans=N diag=N] max_rel=7.834480e-15  normwise=1.430384e-16
[TRSM side=L uplo=U trans=N diag=U] max_rel=6.625125e-12  normwise=7.460659e-16
[TRSM side=L uplo=U trans=T diag=N] max_rel=8.221091e-15  normwise=1.417661e-16
[TRSM side=L uplo=U trans=T diag=U] max_rel=6.574496e-12  normwise=4.428448e-16
[TRSM side=L uplo=L trans=N diag=N] max_rel=3.120429e-13  normwise=1.391054e-16
[TRSM side=L uplo=L trans=N diag=U] max_rel=1.940345e-13  normwise=3.450813e-16
[TRSM side=L uplo=L trans=T diag=N] max_rel=1.642527e-14  normwise=1.429462e-16
[TRSM side=L uplo=L trans=T diag=U] max_rel=5.163139e-13  normwise=4.722269e-16
[TRSM side=R uplo=U trans=N diag=N] max_rel=6.112928e-15  normwise=8.607934e-17
[TRSM side=R uplo=U trans=N diag=U] max_rel=2.242624e-12  normwise=4.835827e-16
[TRSM side=R uplo=U trans=T diag=N] max_rel=1.884041e-13  normwise=8.453538e-17
[TRSM side=R uplo=U trans=T diag=U] max_rel=2.117556e-12  normwise=9.346473e-16
[TRSM side=R uplo=L trans=N diag=N] max_rel=2.824203e-14  normwise=8.766339e-17
[TRSM side=R uplo=L trans=N diag=U] max_rel=5.244419e-13  normwise=8.667487e-16
[TRSM side=R uplo=L trans=T diag=N] max_rel=1.021187e-14  normwise=8.624015e-17
[TRSM side=R uplo=L trans=T diag=U] max_rel=2.341534e-13  normwise=4.139644e-16
```

All normwise errors are within a few ULPs of machine epsilon (≈ 2.2×10⁻¹⁶),
confirming that OpenBLAS double-precision GEMM and TRSM are numerically correct.

### Reference BLAS (Netlib) — double precision

The `examples/reference_blas/` directory tests the Netlib Fortran-77 reference
implementation of BLAS, shipped in the Ubuntu/Debian `libblas3` package
(`/usr/lib/x86_64-linux-gnu/blas/libblas.so.3`).  This is the unoptimised
reference code from [netlib.org/lapack](https://www.netlib.org/lapack/), built
from the `lapack` source package.  It depends only on `libc6` and `libgcc-s1` —
no `libgfortran` or `libopenblas` transitive dependencies.

**Prerequisites**

```sh
apt install libblas3 libmpfr-dev libgmp-dev
make
```

**Run**

```sh
cd examples/reference_blas
./run_test.sh
# or with an explicit library path:
./run_test.sh --blas /usr/lib/x86_64-linux-gnu/blas/libblas.so.3
```

**Sample output** (Netlib BLAS 3.12.0, m=n=k=64, seed=42, prec=256 bits):

```
=== GEMM (dgemm_) — double precision, m=n=k=64 ===
[GEMM transa=N transb=N] max_rel=3.967396e-12  normwise=2.850246e-16
[GEMM transa=N transb=T] max_rel=2.510413e-12  normwise=2.864925e-16
[GEMM transa=T transb=N] max_rel=1.691653e-12  normwise=2.906808e-16
[GEMM transa=T transb=T] max_rel=1.423134e-13  normwise=2.856416e-16

=== TRSM (dtrsm_) — double precision, m=n=64 ===
[TRSM side=L uplo=U trans=N diag=N] max_rel=3.059326e-15  normwise=2.511369e-16
[TRSM side=L uplo=U trans=N diag=U] max_rel=2.328139e-12  normwise=5.415153e-16
[TRSM side=L uplo=U trans=T diag=N] max_rel=2.982475e-14  normwise=2.457734e-16
[TRSM side=L uplo=U trans=T diag=U] max_rel=2.320838e-12  normwise=6.215143e-16
[TRSM side=L uplo=L trans=N diag=N] max_rel=2.307084e-13  normwise=2.505476e-16
[TRSM side=L uplo=L trans=N diag=U] max_rel=6.776894e-12  normwise=4.449071e-16
[TRSM side=L uplo=L trans=T diag=N] max_rel=1.953626e-14  normwise=2.535339e-16
[TRSM side=L uplo=L trans=T diag=U] max_rel=2.592502e-12  normwise=8.733180e-16
[TRSM side=R uplo=U trans=N diag=N] max_rel=1.740966e-14  normwise=2.460380e-16
[TRSM side=R uplo=U trans=N diag=U] max_rel=1.836137e-12  normwise=5.390227e-16
[TRSM side=R uplo=U trans=T diag=N] max_rel=9.419733e-14  normwise=2.436252e-16
[TRSM side=R uplo=U trans=T diag=U] max_rel=1.374433e-12  normwise=4.845049e-16
[TRSM side=R uplo=L trans=N diag=N] max_rel=6.069384e-14  normwise=2.530725e-16
[TRSM side=R uplo=L trans=N diag=U] max_rel=1.545166e-12  normwise=9.025050e-16
[TRSM side=R uplo=L trans=T diag=N] max_rel=1.066756e-14  normwise=2.554720e-16
[TRSM side=R uplo=L trans=T diag=U] max_rel=1.250587e-12  normwise=3.770980e-16
```

All normwise errors are within a few ULPs of machine epsilon (≈ 2.2×10⁻¹⁶).
