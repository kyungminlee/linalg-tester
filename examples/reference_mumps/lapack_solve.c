/*
 * lapack_solve.c — wrapper around LAPACK dgesv_ for testing the MUMPS tester.
 *
 * Exports sparse_solve(n, nnz, irn, jcn, a, rhs, sym) which converts the
 * sparse COO input to a dense matrix and calls dgesv_.  This is obviously
 * not how one would use MUMPS, but it validates the tester against a known-
 * good dense solver without requiring MPI or a MUMPS installation.
 *
 * For a real MUMPS wrapper, see mumps_solve.c in this directory.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o lapack_solve.so lapack_solve.c -llapack -lblas
 */

#include <stdlib.h>
#include <string.h>

/* LAPACK dgesv_ prototype (Fortran interface) */
extern void dgesv_(const int *n, const int *nrhs,
                   double *A, const int *lda,
                   int *ipiv,
                   double *B, const int *ldb,
                   int *info);

int sparse_solve(int n, int nnz, int *irn, int *jcn,
                 void *a, void *rhs, int sym)
{
    /* Build dense matrix from COO */
    double *A = (double *)calloc((size_t)n * n, sizeof(double));
    if (!A) return -1;

    double *av = (double *)a;
    for (int k = 0; k < nnz; ++k) {
        int i = irn[k] - 1; /* convert 1-based to 0-based */
        int j = jcn[k] - 1;
        A[(size_t)j * n + i] += av[k]; /* column-major */
        if (sym != 0 && i != j)
            A[(size_t)i * n + j] += av[k]; /* symmetric: fill upper */
    }

    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) { free(A); return -1; }

    int nrhs = 1;
    int info = 0;
    dgesv_(&n, &nrhs, A, &n, ipiv, (double *)rhs, &n, &info);

    free(A);
    free(ipiv);
    return info;
}
