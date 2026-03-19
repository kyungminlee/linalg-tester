/*
 * dmumps_struc.h — Bundled DMUMPS_STRUC_C definition for linalg-tester.
 *
 * This header provides a self-contained definition of the MUMPS 5.6.x
 * C structure so that the tester can be compiled without installing the
 * MUMPS development headers.  The struct layout matches MUMPS 5.6.2
 * (Ubuntu 24.04 libmumps-seq-dev).
 *
 * If your MUMPS version differs, you can replace this header with:
 *   #include <dmumps_c.h>
 * from your MUMPS installation.
 *
 * Source: MUMPS 5.6.2 (CeCILL-C license)
 *   https://mumps-solver.org/
 */

#pragma once

#include <cstdint>

/* ------------------------------------------------------------------ */
/* MUMPS type definitions (matches mumps_c_types.h for 32-bit ints)     */
/* ------------------------------------------------------------------ */

#ifndef MUMPS_INT
#define MUMPS_INT int
#endif

#ifndef MUMPS_INT8
#define MUMPS_INT8 int64_t
#endif

#ifndef DMUMPS_COMPLEX
#define DMUMPS_COMPLEX double
#endif

#ifndef DMUMPS_REAL
#define DMUMPS_REAL double
#endif

#ifndef MUMPS_VERSION_MAX_LEN
#define MUMPS_VERSION_MAX_LEN 30
#endif

/* ------------------------------------------------------------------ */
/* DMUMPS_STRUC_C  (MUMPS 5.6.2)                                       */
/* ------------------------------------------------------------------ */

typedef struct {

    MUMPS_INT      sym, par, job;
    MUMPS_INT      comm_fortran;    /* Fortran communicator */
    MUMPS_INT      icntl[60];
    MUMPS_INT      keep[500];
    DMUMPS_REAL    cntl[15];
    DMUMPS_REAL    dkeep[230];
    MUMPS_INT8     keep8[150];
    MUMPS_INT      n;
    MUMPS_INT      nblk;

    MUMPS_INT      nz_alloc;

    /* Assembled entry */
    MUMPS_INT      nz;
    MUMPS_INT8     nnz;
    MUMPS_INT      *irn;
    MUMPS_INT      *jcn;
    DMUMPS_COMPLEX *a;

    /* Distributed entry */
    MUMPS_INT      nz_loc;
    MUMPS_INT8     nnz_loc;
    MUMPS_INT      *irn_loc;
    MUMPS_INT      *jcn_loc;
    DMUMPS_COMPLEX *a_loc;

    /* Element entry */
    MUMPS_INT      nelt;
    MUMPS_INT      *eltptr;
    MUMPS_INT      *eltvar;
    DMUMPS_COMPLEX *a_elt;

    /* Matrix by blocks */
    MUMPS_INT      *blkptr;
    MUMPS_INT      *blkvar;

    /* Ordering, if given by user */
    MUMPS_INT      *perm_in;

    /* Orderings returned to user */
    MUMPS_INT      *sym_perm;    /* symmetric permutation */
    MUMPS_INT      *uns_perm;    /* column permutation */

    /* Scaling (inout but complicated) */
    DMUMPS_REAL    *colsca;
    DMUMPS_REAL    *rowsca;
    MUMPS_INT      colsca_from_mumps;
    MUMPS_INT      rowsca_from_mumps;

    /* RHS, solution, output data and statistics */
    DMUMPS_COMPLEX *rhs, *redrhs, *rhs_sparse, *sol_loc, *rhs_loc;
    MUMPS_INT      *irhs_sparse, *irhs_ptr, *isol_loc, *irhs_loc;
    MUMPS_INT      nrhs, lrhs, lredrhs, nz_rhs, lsol_loc, nloc_rhs, lrhs_loc;
    MUMPS_INT      schur_mloc, schur_nloc, schur_lld;
    MUMPS_INT      mblock, nblock, nprow, npcol;
    MUMPS_INT      info[80], infog[80];
    DMUMPS_REAL    rinfo[40], rinfog[40];

    /* Null space */
    MUMPS_INT      deficiency;
    MUMPS_INT      *pivnul_list;
    MUMPS_INT      *mapping;

    /* Schur */
    MUMPS_INT      size_schur;
    MUMPS_INT      *listvar_schur;
    DMUMPS_COMPLEX *schur;

    /* Internal parameters */
    MUMPS_INT      instance_number;
    DMUMPS_COMPLEX *wk_user;

    /* Version number */
    char version_number[MUMPS_VERSION_MAX_LEN + 1 + 1];
    /* For out-of-core */
    char ooc_tmpdir[256];
    char ooc_prefix[64];
    /* To save the matrix in matrix market format */
    char write_problem[256];
    MUMPS_INT      lwk_user;
    /* For save/restore feature */
    char save_dir[256];
    char save_prefix[256];

    /* Metis options */
    MUMPS_INT metis_options[40];
} DMUMPS_STRUC_C;

/* Function pointer type for dmumps_c loaded via dlsym */
extern "C" typedef void (*dmumps_c_fn)(DMUMPS_STRUC_C *);
