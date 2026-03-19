/*
 * mumps_solve.c — thin wrapper around MUMPS dmumps_c for linalg-tester.
 *
 * Exports:
 *   int sparse_solve(int n, int nnz, int *irn, int *jcn,
 *                    void *a, void *rhs, int sym);
 *
 * This wrapper initialises MPI (if not already), creates a MUMPS instance,
 * feeds the COO matrix, solves, and cleans up.
 *
 * Build (sequential MUMPS, Debian/Ubuntu libmumps-seq-dev):
 *   gcc -O2 -shared -fPIC -o mumps_solve.so mumps_solve.c \
 *       -ldmumps_seq -lmumps_common_seq -lpord_seq \
 *       -llapack -lblas -lpthread -lmpiseq_seq -lgfortran -lm
 *
 * Build (parallel MUMPS with MPI):
 *   mpicc -O2 -shared -fPIC -o mumps_solve.so mumps_solve.c \
 *       -ldmumps -lmumps_common -lpord -lscalapack-openmpi \
 *       -llapack -lblas -lpthread -lgfortran -lm
 *
 * Prerequisites:
 *   apt install libmumps-seq-dev   (sequential)
 *   -- or --
 *   apt install libmumps-dev       (MPI parallel)
 */

#include <dmumps_c.h>
#include <string.h>

int sparse_solve(int n, int nnz, int *irn, int *jcn,
                 void *a, void *rhs, int sym)
{
    DMUMPS_STRUC_C id;
    memset(&id, 0, sizeof(id));

    id.par = 1;               /* host participates */
    id.sym = sym;             /* 0=unsym, 1=SPD, 2=general sym */
    id.comm_fortran = -987654; /* use MPI_COMM_WORLD (MUMPS convention) */

    /* Init */
    id.job = -1;
    dmumps_c(&id);
    if (id.infog[0] < 0) return id.infog[0];

    /* Suppress output */
    id.icntl[0] = -1; /* stream for errors   */
    id.icntl[1] = -1; /* stream for warnings  */
    id.icntl[2] = -1; /* stream for info      */
    id.icntl[3] = 0;  /* print level          */

    /* Set matrix (centralized on host) */
    id.n   = n;
    id.nnz = (MUMPS_INT8)nnz;
    id.irn = irn;
    id.jcn = jcn;
    id.a   = (double *)a;

    /* Set RHS */
    id.nrhs = 1;
    id.lrhs = n;
    id.rhs  = (double *)rhs;

    /* Analyse + factorise + solve */
    id.job = 6;
    dmumps_c(&id);
    int rc = id.infog[0];

    /* Cleanup */
    id.job = -2;
    dmumps_c(&id);

    return (rc < 0) ? rc : 0;
}
