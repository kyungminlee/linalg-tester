/*
 * dd_lapack_solve.c — sparse solver wrapper for double-double precision.
 *
 * Exports sparse_solve(n, nnz, irn, jcn, a, rhs, sym) for the
 * linalg-tester MUMPS tester.
 *
 * Each scalar is a double-double pair (hi, lo) representing hi + lo,
 * stored in 16 bytes.  This wrapper:
 *   1. Converts the sparse dd COO matrix to a dense double matrix
 *      (hi + lo summed into a single double — exact for well-formed dd).
 *   2. Converts the dd RHS to a double vector.
 *   3. Solves with LAPACK dgesv_.
 *   4. Converts the double solution back to dd (hi = result, lo = 0).
 *
 * The solve accuracy is limited to double precision (~1e-16) since we
 * use a double-only solver.  For a true dd-precision solver, replace
 * this with your own dd-capable sparse solver library.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o dd_lapack_solve.so dd_lapack_solve.c \
 *       /usr/lib/x86_64-linux-gnu/liblapack.so.3 \
 *       /usr/lib/x86_64-linux-gnu/libblas.so.3 -lm
 */

#include <stdlib.h>
#include <string.h>

/* LAPACK dgesv_ prototype */
extern void dgesv_(const int *n, const int *nrhs,
                   double *A, const int *lda,
                   int *ipiv,
                   double *B, const int *ldb,
                   int *info);

/* Read one dd scalar (16 bytes) and return hi + lo as a double. */
static double dd_to_double(const void *src)
{
    double hi, lo;
    memcpy(&hi, (const char *)src, sizeof(double));
    memcpy(&lo, (const char *)src + sizeof(double), sizeof(double));
    return hi + lo;
}

/* Write a double value as a dd scalar (hi = val, lo = 0). */
static void double_to_dd(void *dst, double val)
{
    double lo = 0.0;
    memcpy((char *)dst, &val, sizeof(double));
    memcpy((char *)dst + sizeof(double), &lo, sizeof(double));
}

int sparse_solve(int n, int nnz, int *irn, int *jcn,
                 void *a, void *rhs, int sym)
{
    const size_t dd_size = 2 * sizeof(double); /* 16 bytes */

    /* Build dense double matrix from dd COO */
    double *A = (double *)calloc((size_t)n * n, sizeof(double));
    if (!A) return -1;

    const char *av = (const char *)a;
    for (int k = 0; k < nnz; ++k) {
        int i = irn[k] - 1;
        int j = jcn[k] - 1;
        double val = dd_to_double(av + (size_t)k * dd_size);
        A[(size_t)j * n + i] += val;  /* column-major */
        if (sym != 0 && i != j)
            A[(size_t)i * n + j] += val;
    }

    /* Convert dd RHS to double */
    double *b = (double *)malloc((size_t)n * sizeof(double));
    if (!b) { free(A); return -1; }

    const char *rp = (const char *)rhs;
    for (int i = 0; i < n; ++i)
        b[i] = dd_to_double(rp + (size_t)i * dd_size);

    /* Solve with LAPACK */
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) { free(A); free(b); return -1; }

    int nrhs = 1;
    int info = 0;
    dgesv_(&n, &nrhs, A, &n, ipiv, b, &n, &info);

    /* Convert double solution back to dd */
    if (info == 0) {
        char *wp = (char *)rhs;
        for (int i = 0; i < n; ++i)
            double_to_dd(wp + (size_t)i * dd_size, b[i]);
    }

    free(A);
    free(b);
    free(ipiv);
    return info;
}
