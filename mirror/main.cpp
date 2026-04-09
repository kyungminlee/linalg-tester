/* main.cpp -- Mirror tester: compare two libraries */

#include "mirror_ctx.h"
#include "derive_sym.h"
#include "blas/mirror_blas3.h"
#include "blas/mirror_blas2.h"
#include "blas/mirror_blas1.h"
#include "lapack/factorizations.h"
#include "lapack/solvers.h"
#include "lapack/eigenvalues.h"
#include "lapack/auxiliary.h"
#include "pblas/mirror_pblas3.h"
#include "pblas/mirror_pblas2.h"
#include "pblas/mirror_pblas1.h"
#include "blacs/mirror_blacs.h"
#include "scalapack/mirror_scalapack.h"
#include "../src/core/loader.h"

#include <CLI11.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/* Dispatch table                                                       */
/* ------------------------------------------------------------------ */

static const MirrorRoutineEntry routines[] = {
    /* ---- BLAS Level 3 (real) ---- */
    {"gemm",   nullptr, "blas3", "General matrix multiply",     "C = alpha*op(A)*op(B) + beta*C", mirror_test_gemm},
    {"trsm",   nullptr, "blas3", "Triangular solve (matrix)",   "op(A)*X = alpha*B",              mirror_test_trsm},
    {"symm",   nullptr, "blas3", "Symmetric multiply",          "C = alpha*A*B + beta*C",         mirror_test_symm},
    {"syrk",   nullptr, "blas3", "Symmetric rank-k update",     "C = alpha*A*A^T + beta*C",       mirror_test_syrk},
    {"syr2k",  nullptr, "blas3", "Symmetric rank-2k update",    "C = alpha*A*B^T + ... + beta*C", mirror_test_syr2k},
    {"trmm",   nullptr, "blas3", "Triangular multiply (matrix)","B = alpha*op(A)*B",              mirror_test_trmm},
    /* ---- BLAS Level 3 (complex) ---- */
    {"hemm",   nullptr, "cblas3","Hermitian multiply",          "C = alpha*A*B + beta*C",         mirror_test_hemm},
    {"herk",   nullptr, "cblas3","Hermitian rank-k update",     "C = alpha*A*A^H + beta*C",       mirror_test_herk},
    {"her2k",  nullptr, "cblas3","Hermitian rank-2k update",    "C = alpha*A*B^H + ... + beta*C", mirror_test_her2k},
    /* ---- BLAS Level 2 (real) ---- */
    {"gemv",   nullptr, "blas2", "General matrix-vector",       "y = alpha*op(A)*x + beta*y",     mirror_test_gemv},
    {"symv",   nullptr, "blas2", "Symmetric matrix-vector",     "y = alpha*A*x + beta*y",         mirror_test_symv},
    {"trmv",   nullptr, "blas2", "Triangular matrix-vector",    "x = op(A)*x",                    mirror_test_trmv},
    {"trsv",   nullptr, "blas2", "Triangular solve (vector)",   "op(A)*x = b",                    mirror_test_trsv},
    {"ger",    nullptr, "blas2", "General rank-1 update",       "A = alpha*x*y^T + A",            mirror_test_ger},
    {"syr",    nullptr, "blas2", "Symmetric rank-1 update",     "A = alpha*x*x^T + A",            mirror_test_syr},
    {"syr2",   nullptr, "blas2", "Symmetric rank-2 update",     "A = alpha*x*y^T + alpha*y*x^T + A", mirror_test_syr2},
    {"gbmv",   nullptr, "blas2", "General band matrix-vector",  "y = alpha*op(A)*x + beta*y",     mirror_test_gbmv},
    {"sbmv",   nullptr, "blas2", "Symmetric band matrix-vector","y = alpha*A*x + beta*y",         mirror_test_sbmv},
    {"tbmv",   nullptr, "blas2", "Triangular band mat-vec",     "x = op(A)*x",                    mirror_test_tbmv},
    {"tbsv",   nullptr, "blas2", "Triangular band solve",       "op(A)*x = b",                    mirror_test_tbsv},
    {"spmv",   nullptr, "blas2", "Symmetric packed mat-vec",    "y = alpha*A*x + beta*y",         mirror_test_spmv},
    {"tpmv",   nullptr, "blas2", "Triangular packed mat-vec",   "x = op(A)*x",                    mirror_test_tpmv},
    {"tpsv",   nullptr, "blas2", "Triangular packed solve",     "op(A)*x = b",                    mirror_test_tpsv},
    {"spr",    nullptr, "blas2", "Symmetric packed rank-1",     "A = alpha*x*x^T + A",            mirror_test_spr},
    {"spr2",   nullptr, "blas2", "Symmetric packed rank-2",     "A = alpha*x*y^T + alpha*y*x^T + A", mirror_test_spr2},
    /* ---- BLAS Level 2 (complex) ---- */
    {"hemv",   nullptr, "cblas2","Hermitian matrix-vector",     "y = alpha*A*x + beta*y",         mirror_test_hemv},
    {"hbmv",   nullptr, "cblas2","Hermitian band mat-vec",      "y = alpha*A*x + beta*y",         mirror_test_hbmv},
    {"hpmv",   nullptr, "cblas2","Hermitian packed mat-vec",    "y = alpha*A*x + beta*y",         mirror_test_hpmv},
    {"geru",   nullptr, "cblas2","Unconjugated rank-1 update",  "A = alpha*x*y^T + A",            mirror_test_geru},
    {"gerc",   nullptr, "cblas2","Conjugated rank-1 update",    "A = alpha*x*y^H + A",            mirror_test_gerc},
    {"her",    nullptr, "cblas2","Hermitian rank-1 update",     "A = alpha*x*x^H + A",            mirror_test_her},
    {"hpr",    nullptr, "cblas2","Hermitian packed rank-1",     "A = alpha*x*x^H + A",            mirror_test_hpr},
    {"her2",   nullptr, "cblas2","Hermitian rank-2 update",     "A = alpha*x*y^H + conj(alpha)*y*x^H + A", mirror_test_her2},
    {"hpr2",   nullptr, "cblas2","Hermitian packed rank-2",     "A = alpha*x*y^H + conj(alpha)*y*x^H + A", mirror_test_hpr2},
    /* ---- BLAS Level 1 (real) ---- */
    {"rotg",   nullptr, "blas1", "Givens rotation setup",       "[c,s] = rotg(a,b)",              mirror_test_rotg},
    {"rotmg",  nullptr, "blas1", "Modified Givens setup",       "param = rotmg(d1,d2,x1,y1)",     mirror_test_rotmg},
    {"rot",    nullptr, "blas1", "Givens rotation apply",       "[x,y] = rot(x,y,c,s)",           mirror_test_rot},
    {"rotm",   nullptr, "blas1", "Modified Givens apply",       "[x,y] = rotm(x,y,param)",        mirror_test_rotm},
    {"swap",   nullptr, "blas1", "Vector swap",                 "x <-> y",                        mirror_test_swap},
    {"scal",   nullptr, "blas1", "Vector scale",                "x = alpha*x",                    mirror_test_scal},
    {"copy",   nullptr, "blas1", "Vector copy",                 "y = x",                          mirror_test_copy},
    {"axpy",   nullptr, "blas1", "Vector axpy",                 "y = alpha*x + y",                mirror_test_axpy},
    {"dot",    nullptr, "blas1", "Dot product",                 "result = x^T*y",                 mirror_test_dot},
    {"nrm2",   nullptr, "blas1", "Vector 2-norm",               "result = ||x||_2",               mirror_test_nrm2},
    {"asum",   nullptr, "blas1", "Sum of absolute values",      "result = sum|x_i|",              mirror_test_asum},
    {"iamax",  nullptr, "blas1", "Index of max abs value",      "result = argmax|x_i|",           mirror_test_iamax},
    /* ---- BLAS Level 1 (complex) ---- */
    {"dotc",   nullptr, "cblas1","Conjugated dot product",      "result = x^H*y",                 mirror_test_dotc},
    {"dotu",   nullptr, "cblas1","Unconjugated dot product",    "result = x^T*y",                 mirror_test_dotu},
    {"crot",   nullptr, "cblas1","Complex rotation apply",      "[x,y] = rot(x,y,c,s)",           mirror_test_crot},
    {"crscal", nullptr, "cblas1","Real scale of complex",       "x = alpha*x (alpha real)",        mirror_test_crscal},

    /* ---- LAPACK Factorizations (real) ---- */
    {"getrf",  nullptr, "lapack_fact", "LU factorization",           "A = P*L*U",           mirror_test_getrf},
    {"potrf",  nullptr, "lapack_fact", "Cholesky factorization",     "A = L*L^T or U^T*U",  mirror_test_potrf},
    {"sytrf",  nullptr, "lapack_fact", "Symmetric indefinite fact",  "A = U*D*U^T or L*D*L^T", mirror_test_sytrf},
    {"geqrf",  nullptr, "lapack_fact", "QR factorization",           "A = Q*R",             mirror_test_geqrf},
    {"gelqf",  nullptr, "lapack_fact", "LQ factorization",           "A = L*Q",             mirror_test_gelqf},
    {"gebrd",  nullptr, "lapack_fact", "Bidiagonal reduction",       "A = Q*B*P^T",         mirror_test_gebrd},
    {"sytrd",  nullptr, "lapack_fact", "Tridiagonal reduction",      "A = Q*T*Q^T",         mirror_test_sytrd},
    /* ---- LAPACK Factorizations (complex) ---- */
    {"cgetrf", "getrf", "clapack_fact","Complex LU factorization",   "A = P*L*U",           mirror_test_cgetrf},
    {"cpotrf", "potrf", "clapack_fact","Complex Cholesky fact",      "A = L*L^H or U^H*U",  mirror_test_cpotrf},
    {"cgeqrf", "geqrf", "clapack_fact","Complex QR factorization",   "A = Q*R",             mirror_test_cgeqrf},
    {"cgelqf", "gelqf", "clapack_fact","Complex LQ factorization",   "A = L*Q",             mirror_test_cgelqf},
    {"cgebrd", "gebrd", "clapack_fact","Complex bidiag reduction",   "A = Q*B*P^H",         mirror_test_cgebrd},
    {"hetrf",  nullptr, "clapack_fact","Hermitian indefinite fact",   "A = U*D*U^H or L*D*L^H", mirror_test_hetrf},
    {"hetrd",  nullptr, "clapack_fact","Hermitian tridiag reduction", "A = Q*T*Q^H",         mirror_test_hetrd},
    {"csytrf", "sytrf", "clapack_fact","Complex sym indefinite fact", "A = U*D*U^T or L*D*L^T", mirror_test_csytrf},

    /* ---- LAPACK Solvers (real) ---- */
    {"gesv",   nullptr, "lapack_solve","General linear solve",       "A*X = B",             mirror_test_gesv},
    {"posv",   nullptr, "lapack_solve","Positive definite solve",    "A*X = B (A SPD)",     mirror_test_posv},
    {"sysv",   nullptr, "lapack_solve","Symmetric indefinite solve", "A*X = B (A sym)",     mirror_test_sysv},
    {"gbsv",   nullptr, "lapack_solve","Band linear solve",          "A*X = B (A banded)",  mirror_test_gbsv},
    {"gtsv",   nullptr, "lapack_solve","Tridiagonal solve",          "A*X = B (A tridiag)", mirror_test_gtsv},
    {"gels",   nullptr, "lapack_solve","Least squares solve",        "min||A*X-B||",        mirror_test_gels},
    {"gelsd",  nullptr, "lapack_solve","Least squares via SVD",      "min||A*X-B|| (SVD)",  mirror_test_gelsd},
    /* ---- LAPACK Solvers (complex) ---- */
    {"cgesv",  "gesv",  "clapack_solve","Complex general solve",     "A*X = B",             mirror_test_cgesv},
    {"cposv",  "posv",  "clapack_solve","Complex pos-def solve",     "A*X = B (A HPD)",     mirror_test_cposv},
    {"cgbsv",  "gbsv",  "clapack_solve","Complex band solve",        "A*X = B (A banded)",  mirror_test_cgbsv},
    {"cgtsv",  "gtsv",  "clapack_solve","Complex tridiag solve",     "A*X = B (A tridiag)", mirror_test_cgtsv},
    {"cgels",  "gels",  "clapack_solve","Complex least squares",     "min||A*X-B||",        mirror_test_cgels},
    {"cgelsd", "gelsd", "clapack_solve","Complex LS via SVD",        "min||A*X-B|| (SVD)",  mirror_test_cgelsd},
    {"hesv",   nullptr, "clapack_solve","Hermitian indef solve",     "A*X = B (A herm)",    mirror_test_hesv},
    {"csysv",  "sysv",  "clapack_solve","Complex sym indef solve",   "A*X = B (A csym)",    mirror_test_csysv},

    /* ---- LAPACK Eigenvalues (real) ---- */
    {"syev",   nullptr, "lapack_eig",  "Symmetric eigenvalues (QR)", "A*v = lambda*v",      mirror_test_syev},
    {"syevd",  nullptr, "lapack_eig",  "Symmetric eigenvalues (D&C)","A*v = lambda*v",      mirror_test_syevd},
    {"syevr",  nullptr, "lapack_eig",  "Symmetric eigenvalues (MRRR)","A*v = lambda*v",     mirror_test_syevr},
    {"geev",   nullptr, "lapack_eig",  "General eigenvalues",        "A*v = lambda*v",      mirror_test_geev},
    {"gees",   nullptr, "lapack_eig",  "Schur decomposition",        "A = V*T*V^T",         mirror_test_gees},
    {"gesvd",  nullptr, "lapack_eig",  "SVD (QR)",                   "A = U*S*V^T",         mirror_test_gesvd},
    {"gesdd",  nullptr, "lapack_eig",  "SVD (D&C)",                  "A = U*S*V^T",         mirror_test_gesdd},
    {"sygv",   nullptr, "lapack_eig",  "Gen symmetric eigenvalues",  "A*v = lambda*B*v",    mirror_test_sygv},
    /* ---- LAPACK Eigenvalues (complex) ---- */
    {"heev",   nullptr, "clapack_eig", "Hermitian eigenvalues (QR)", "A*v = lambda*v",      mirror_test_heev},
    {"heevd",  nullptr, "clapack_eig", "Hermitian eigenvalues (D&C)","A*v = lambda*v",      mirror_test_heevd},
    {"heevr",  nullptr, "clapack_eig", "Hermitian eigenvalues (MRRR)","A*v = lambda*v",     mirror_test_heevr},
    {"hegv",   nullptr, "clapack_eig", "Gen hermitian eigenvalues",  "A*v = lambda*B*v",    mirror_test_hegv},
    {"cgesvd", "gesvd", "clapack_eig", "Complex SVD (QR)",           "A = U*S*V^H",         mirror_test_cgesvd},
    {"cgesdd", "gesdd", "clapack_eig", "Complex SVD (D&C)",          "A = U*S*V^H",         mirror_test_cgesdd},
    {"cgeev",  "geev",  "clapack_eig", "Complex general eigenvalues","A*v = lambda*v",      mirror_test_cgeev},
    {"cgees",  "gees",  "clapack_eig", "Complex Schur decomposition","A = V*T*V^H",         mirror_test_cgees},

    /* ---- LAPACK Auxiliary (real) ---- */
    {"getrs",  nullptr, "lapack_aux",  "LU-based solve",             "op(A)*X = B",         mirror_test_getrs},
    {"getri",  nullptr, "lapack_aux",  "LU-based inverse",           "A^-1",                mirror_test_getri},
    {"potrs",  nullptr, "lapack_aux",  "Cholesky-based solve",       "A*X = B",             mirror_test_potrs},
    {"potri",  nullptr, "lapack_aux",  "Cholesky-based inverse",     "A^-1",                mirror_test_potri},
    {"orgqr",  nullptr, "lapack_aux",  "Generate Q from QR",         "Q from QR",           mirror_test_orgqr},
    {"ormqr",  nullptr, "lapack_aux",  "Multiply by Q from QR",      "C = op(Q)*C",         mirror_test_ormqr},
    {"gecon",  nullptr, "lapack_aux",  "Condition number estimate",   "rcond(A)",            mirror_test_gecon},
    {"lange",  nullptr, "lapack_aux",  "General matrix norm",         "||A||",               mirror_test_lange},
    {"lansy",  nullptr, "lapack_aux",  "Symmetric matrix norm",       "||A|| (sym)",         mirror_test_lansy},
    {"lacpy",  nullptr, "lapack_aux",  "Matrix copy",                 "B = A",               mirror_test_lacpy},
    {"laswp",  nullptr, "lapack_aux",  "Row permutation",             "A = P*A",             mirror_test_laswp},
    /* ---- LAPACK Auxiliary (complex) ---- */
    {"cgetrs", "getrs", "clapack_aux", "Complex LU solve",           "op(A)*X = B",         mirror_test_cgetrs},
    {"cgetri", "getri", "clapack_aux", "Complex LU inverse",         "A^-1",                mirror_test_cgetri},
    {"cpotrs", "potrs", "clapack_aux", "Complex Cholesky solve",     "A*X = B",             mirror_test_cpotrs},
    {"cpotri", "potri", "clapack_aux", "Complex Cholesky inverse",   "A^-1",                mirror_test_cpotri},
    {"ungqr",  nullptr, "clapack_aux", "Generate unitary Q from QR", "Q from QR",           mirror_test_ungqr},
    {"unmqr",  nullptr, "clapack_aux", "Multiply by unitary Q",      "C = op(Q)*C",         mirror_test_unmqr},
    {"cgecon", "gecon", "clapack_aux", "Complex condition number",    "rcond(A)",            mirror_test_cgecon},
    {"clange", "lange", "clapack_aux", "Complex matrix norm",         "||A||",               mirror_test_clange},
    {"lanhe",  nullptr, "clapack_aux", "Hermitian matrix norm",       "||A|| (herm)",        mirror_test_lanhe},
    {"clacpy", "lacpy", "clapack_aux", "Complex matrix copy",         "B = A",               mirror_test_clacpy},
    {"claswp", "laswp", "clapack_aux", "Complex row permutation",     "A = P*A",             mirror_test_claswp},
    {"clansy", "lansy", "clapack_aux", "Complex sym matrix norm",     "||A|| (csym)",        mirror_test_clansy},

    /* ---- PBLAS Level 3 (real) ---- */
    {"pgemm",  nullptr, "pblas3",  "Parallel GEMM",             "C = alpha*op(A)*op(B) + beta*C", mirror_test_pgemm},
    {"ptrsm",  nullptr, "pblas3",  "Parallel triangular solve", "op(A)*X = alpha*B",              mirror_test_ptrsm},
    {"psymm",  nullptr, "pblas3",  "Parallel symmetric mult",   "C = alpha*A*B + beta*C",         mirror_test_psymm},
    {"psyrk",  nullptr, "pblas3",  "Parallel sym rank-k",       "C = alpha*A*A^T + beta*C",       mirror_test_psyrk},
    {"psyr2k", nullptr, "pblas3",  "Parallel sym rank-2k",      "C = alpha*A*B^T + ... + beta*C", mirror_test_psyr2k},
    {"ptrmm",  nullptr, "pblas3",  "Parallel tri multiply",     "B = alpha*op(A)*B",              mirror_test_ptrmm},
    {"ptran",  nullptr, "pblas3",  "Parallel transpose",        "C = alpha*A^T + beta*C",         mirror_test_ptran},
    /* ---- PBLAS Level 3 (complex) ---- */
    {"phemm",  nullptr, "cpblas3", "Parallel hermitian mult",   "C = alpha*A*B + beta*C",         mirror_test_phemm},
    {"pherk",  nullptr, "cpblas3", "Parallel herm rank-k",      "C = alpha*A*A^H + beta*C",       mirror_test_pherk},
    {"pher2k", nullptr, "cpblas3", "Parallel herm rank-2k",     "C = alpha*A*B^H + ... + beta*C", mirror_test_pher2k},
    {"ptranc", nullptr, "cpblas3", "Parallel conj transpose",   "C = alpha*A^H + beta*C",         mirror_test_ptranc},
    {"ptranu", nullptr, "cpblas3", "Parallel unconj transpose", "C = alpha*A^T + beta*C",         mirror_test_ptranu},
    /* ---- PBLAS Level 2 (real) ---- */
    {"pgemv",  nullptr, "pblas2",  "Parallel GEMV",             "y = alpha*op(A)*x + beta*y",     mirror_test_pgemv},
    {"psymv",  nullptr, "pblas2",  "Parallel sym mat-vec",      "y = alpha*A*x + beta*y",         mirror_test_psymv},
    {"ptrmv",  nullptr, "pblas2",  "Parallel tri mat-vec",      "x = op(A)*x",                    mirror_test_ptrmv},
    {"ptrsv",  nullptr, "pblas2",  "Parallel tri solve (vec)",  "op(A)*x = b",                    mirror_test_ptrsv},
    {"pger",   nullptr, "pblas2",  "Parallel rank-1 update",    "A = alpha*x*y^T + A",            mirror_test_pger},
    {"psyr",   nullptr, "pblas2",  "Parallel sym rank-1",       "A = alpha*x*x^T + A",            mirror_test_psyr},
    {"psyr2",  nullptr, "pblas2",  "Parallel sym rank-2",       "A = alpha*x*y^T + alpha*y*x^T + A", mirror_test_psyr2},
    /* ---- PBLAS Level 2 (complex) ---- */
    {"phemv",  nullptr, "cpblas2", "Parallel herm mat-vec",     "y = alpha*A*x + beta*y",         mirror_test_phemv},
    {"pgeru",  nullptr, "cpblas2", "Parallel unconj rank-1",    "A = alpha*x*y^T + A",            mirror_test_pgeru},
    {"pgerc",  nullptr, "cpblas2", "Parallel conj rank-1",      "A = alpha*x*y^H + A",            mirror_test_pgerc},
    {"pher",   nullptr, "cpblas2", "Parallel herm rank-1",      "A = alpha*x*x^H + A",            mirror_test_pher},
    {"pher2",  nullptr, "cpblas2", "Parallel herm rank-2",      "A = alpha*x*y^H + conj(alpha)*y*x^H + A", mirror_test_pher2},
    /* ---- PBLAS Level 1 (real) ---- */
    {"pswap",  nullptr, "pblas1",  "Parallel vector swap",      "x <-> y",                        mirror_test_pswap},
    {"pscal",  nullptr, "pblas1",  "Parallel vector scale",     "x = alpha*x",                    mirror_test_pscal},
    {"pcopy",  nullptr, "pblas1",  "Parallel vector copy",      "y = x",                          mirror_test_pcopy},
    {"paxpy",  nullptr, "pblas1",  "Parallel vector axpy",      "y = alpha*x + y",                mirror_test_paxpy},
    {"pdot",   nullptr, "pblas1",  "Parallel dot product",      "result = x^T*y",                 mirror_test_pdot},
    {"pnrm2",  nullptr, "pblas1",  "Parallel 2-norm",           "result = ||x||_2",               mirror_test_pnrm2},
    {"pasum",  nullptr, "pblas1",  "Parallel absolute sum",     "result = sum|x_i|",              mirror_test_pasum},
    {"pamax",  nullptr, "pblas1",  "Parallel max abs value",    "result = max|x_i|",              mirror_test_pamax},
    /* ---- PBLAS Level 1 (complex) ---- */
    {"pdotc",  nullptr, "cpblas1", "Parallel conj dot",         "result = x^H*y",                 mirror_test_pdotc},
    {"pdotu",  nullptr, "cpblas1", "Parallel unconj dot",       "result = x^T*y",                 mirror_test_pdotu},

    /* ---- BLACS ---- */
    {"blacs_setup",  nullptr,   "blacs", "BLACS grid setup",      "grid init/query",    mirror_test_blacs_setup},
    {"blacs_gesd2d", "gesd2d",  "blacs", "General send/recv",     "point-to-point",     mirror_test_blacs_gesd2d},
    {"blacs_trsd2d", "trsd2d",  "blacs", "Triangular send/recv",  "point-to-point tri", mirror_test_blacs_trsd2d},
    {"blacs_gebs2d", "gebs2d",  "blacs", "General broadcast",     "broadcast",          mirror_test_blacs_gebs2d},
    {"blacs_trbs2d", "trbs2d",  "blacs", "Triangular broadcast",  "broadcast tri",      mirror_test_blacs_trbs2d},
    {"blacs_gsum2d", "gsum2d",  "blacs", "Global sum",            "combine sum",        mirror_test_blacs_gsum2d},
    {"blacs_gamx2d", "gamx2d",  "blacs", "Global max",            "combine max",        mirror_test_blacs_gamx2d},
    {"blacs_gamn2d", "gamn2d",  "blacs", "Global min",            "combine min",        mirror_test_blacs_gamn2d},

    /* ---- ScaLAPACK Factorizations (real) ---- */
    {"pgetrf",  nullptr,  "scalapack_fact",  "Distributed LU fact",       "A = P*L*U",          mirror_test_pgetrf},
    {"ppotrf",  nullptr,  "scalapack_fact",  "Distributed Cholesky",      "A = L*L^T",          mirror_test_ppotrf},
    {"pgeqrf",  nullptr,  "scalapack_fact",  "Distributed QR fact",       "A = Q*R",            mirror_test_pgeqrf},
    /* ---- ScaLAPACK Factorizations (complex) ---- */
    {"cpgetrf", "pgetrf", "cscalapack_fact", "Complex distributed LU",    "A = P*L*U",          mirror_test_cpgetrf},
    {"cppotrf", "ppotrf", "cscalapack_fact", "Complex distributed Cholesky","A = L*L^H",        mirror_test_cppotrf},
    {"cpgeqrf", "pgeqrf", "cscalapack_fact", "Complex distributed QR",    "A = Q*R",            mirror_test_cpgeqrf},

    /* ---- ScaLAPACK Solvers (real) ---- */
    {"pgesv",   nullptr,  "scalapack_solve", "Distributed general solve", "A*X = B",            mirror_test_pgesv},
    {"pposv",   nullptr,  "scalapack_solve", "Distributed SPD solve",     "A*X = B (A SPD)",    mirror_test_pposv},
    /* ---- ScaLAPACK Solvers (complex) ---- */
    {"cpgesv",  "pgesv",  "cscalapack_solve","Complex distributed solve", "A*X = B",            mirror_test_cpgesv},
    {"cpposv",  "pposv",  "cscalapack_solve","Complex distributed SPD solve","A*X = B (A HPD)", mirror_test_cpposv},

    /* ---- ScaLAPACK Auxiliary (real) ---- */
    {"pgetrs",  nullptr,  "scalapack_aux",   "Distributed LU solve",      "op(A)*X = B",        mirror_test_pgetrs},
    {"ppotrs",  nullptr,  "scalapack_aux",   "Distributed Cholesky solve", "A*X = B",           mirror_test_ppotrs},
    {"ptrtri",  nullptr,  "scalapack_aux",   "Distributed tri inverse",    "A^-1 (triangular)", mirror_test_ptrtri},
    {"placpy",  nullptr,  "scalapack_aux",   "Distributed matrix copy",    "B = A",             mirror_test_placpy},
    {"plange",  nullptr,  "scalapack_aux",   "Distributed matrix norm",    "||A||",             mirror_test_plange},
    {"plansy",  nullptr,  "scalapack_aux",   "Distributed sym norm",       "||A|| (sym)",       mirror_test_plansy},
    /* ---- ScaLAPACK Auxiliary (complex) ---- */
    {"cpgetrs", "pgetrs", "cscalapack_aux",  "Complex distributed LU solve","op(A)*X = B",      mirror_test_cpgetrs},
    {"cppotrs", "ppotrs", "cscalapack_aux",  "Complex distributed Chol solve","A*X = B",        mirror_test_cppotrs},
    {"cptrtri", "ptrtri", "cscalapack_aux",  "Complex distributed tri inv", "A^-1 (tri)",       mirror_test_cptrtri},
    {"cplacpy", "placpy", "cscalapack_aux",  "Complex distributed copy",    "B = A",            mirror_test_cplacpy},
    {"cplange", "plange", "cscalapack_aux",  "Complex distributed norm",    "||A||",            mirror_test_cplange},
    {"planhe",  nullptr,  "cscalapack_aux",  "Distributed Hermitian norm",  "||A|| (herm)",     mirror_test_planhe},
    {"cplansy", "plansy", "cscalapack_aux",  "Complex distributed sym norm","||A|| (csym)",     mirror_test_cplansy},

    /* ---- ScaLAPACK Eigenvalues (real) ---- */
    {"psyev",   nullptr,  "scalapack_eig",   "Distributed sym eigenvalues", "A*v = lambda*v",   mirror_test_psyev},
    {"psyevd",  nullptr,  "scalapack_eig",   "Distributed sym eig (D&C)",   "A*v = lambda*v",   mirror_test_psyevd},
    {"pgesvd",  nullptr,  "scalapack_eig",   "Distributed SVD",             "A = U*S*V^T",      mirror_test_pgesvd},
    /* ---- ScaLAPACK Eigenvalues (complex) ---- */
    {"pheev",   nullptr,  "cscalapack_eig",  "Distributed Herm eigenvalues","A*v = lambda*v",   mirror_test_pheev},
    {"pheevd",  nullptr,  "cscalapack_eig",  "Distributed Herm eig (D&C)", "A*v = lambda*v",   mirror_test_pheevd},
    {"cpgesvd", "pgesvd", "cscalapack_eig",  "Complex distributed SVD",    "A = U*S*V^H",      mirror_test_cpgesvd},

    /* ---- ScaLAPACK Redistribution ---- */
    {"pgemr2d",  nullptr,  "scalapack_redist","Distributed redistribution", "B = A (reblock)",  mirror_test_pgemr2d},
    {"cpgemr2d", "pgemr2d","cscalapack_redist","Complex redistribution",   "B = A (reblock)",  mirror_test_cpgemr2d},
};

static constexpr int num_routines = sizeof(routines) / sizeof(routines[0]);

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static const MirrorRoutineEntry *find_routine(const char *name) {
    for (int i = 0; i < num_routines; ++i)
        if (std::strcmp(routines[i].name, name) == 0)
            return &routines[i];
    return nullptr;
}

static bool is_batch(const std::string &name) {
    static const char *batches[] = {
        "all",
        "blas", "cblas", "blas1", "blas2", "blas3", "cblas1", "cblas2", "cblas3",
        "lapack", "clapack",
        "lapack_fact", "clapack_fact", "lapack_solve", "clapack_solve",
        "lapack_eig", "clapack_eig", "lapack_aux", "clapack_aux",
        "pblas", "cpblas", "pblas1", "pblas2", "pblas3", "cpblas1", "cpblas2", "cpblas3",
        "blacs",
        "scalapack", "cscalapack",
        "scalapack_fact", "cscalapack_fact", "scalapack_solve", "cscalapack_solve",
        "scalapack_aux", "cscalapack_aux", "scalapack_eig", "cscalapack_eig",
        "scalapack_redist", "cscalapack_redist",
    };
    for (auto b : batches)
        if (name == b) return true;
    return false;
}

static bool category_matches(const char *cat, const std::string &batch) {
    if (batch == "all") return true;
    /* Top-level groups */
    if (batch == "blas")    return std::strncmp(cat, "blas", 4) == 0;
    if (batch == "cblas")   return std::strncmp(cat, "cblas", 5) == 0;
    if (batch == "lapack")  return std::strstr(cat, "lapack") != nullptr;
    if (batch == "clapack") return std::strstr(cat, "clapack") != nullptr;
    if (batch == "pblas")   return std::strstr(cat, "pblas") != nullptr;
    if (batch == "cpblas")  return std::strncmp(cat, "cpblas", 6) == 0;
    if (batch == "blacs")      return std::strcmp(cat, "blacs") == 0;
    if (batch == "scalapack")  return std::strstr(cat, "scalapack") != nullptr;
    if (batch == "cscalapack") return std::strncmp(cat, "cscalapack", 10) == 0;
    /* Exact match */
    return batch == cat;
}

static void run_routine(const MirrorRoutineEntry &r,
                         const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config) {
    std::printf("=== %s (%s) ===\n", r.name, r.description);
    r.test_fn(a, b, params, config);
    std::printf("\n");
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    CLI::App app{"mirror-tester: compare two libraries"};

    bool list_flag = false;
    std::string routine_name;

    /* Side A */
    std::string lib_a_path, conv_lib_a_path, sym_prefix_a, sym_a;
    int typesize_a = 0;
    bool complex_a = false;
    std::string complex_abi_a = "hidden";
    std::vector<std::string> preload_a;

    /* Side B */
    std::string lib_b_path, conv_lib_b_path, sym_prefix_b, sym_b;
    int typesize_b = 0;
    bool complex_b = false;
    std::string complex_abi_b = "hidden";
    std::vector<std::string> preload_b;

    /* Shared */
    int prec = 256;
    unsigned seed = 42;
    int m = 64, n = 64, k = 64;
    int kl = 2, ku = 2;
    int incx = 1, incy = 1;
    int ld_pad = 0;
    int mb = 0, nb_val = 0;
    double threshold = -1.0;
    std::string format = "text";
    std::string reference = "a";

    app.add_flag("--list", list_flag, "List all supported routines and exit");
    app.add_option("--routine", routine_name, "Routine name or category");

    /* Side A options */
    app.add_option("--lib-a", lib_a_path, "Library path for side A");
    app.add_option("--conv-lib-a", conv_lib_a_path, "Conversion library for side A");
    app.add_option("--typesize-a", typesize_a, "Element size in bytes for side A");
    app.add_option("--sym-prefix-a", sym_prefix_a, "Symbol prefix for side A (e.g. s, d)");
    app.add_option("--sym-a", sym_a, "Explicit symbol for side A");
    app.add_flag("--complex-a", complex_a, "Enable complex mode for side A");
    app.add_option("--complex-return-abi-a", complex_abi_a, "Complex return ABI for side A")->default_val("hidden");
    app.add_option("--preload-a", preload_a, "Libraries to preload for side A");

    /* Side B options */
    app.add_option("--lib-b", lib_b_path, "Library path for side B");
    app.add_option("--conv-lib-b", conv_lib_b_path, "Conversion library for side B");
    app.add_option("--typesize-b", typesize_b, "Element size in bytes for side B");
    app.add_option("--sym-prefix-b", sym_prefix_b, "Symbol prefix for side B (e.g. s, d)");
    app.add_option("--sym-b", sym_b, "Explicit symbol for side B");
    app.add_flag("--complex-b", complex_b, "Enable complex mode for side B");
    app.add_option("--complex-return-abi-b", complex_abi_b, "Complex return ABI for side B")->default_val("hidden");
    app.add_option("--preload-b", preload_b, "Libraries to preload for side B");

    /* Shared options */
    app.add_option("--prec", prec, "MPFR precision in bits")->default_val(256);
    app.add_option("--seed", seed, "RNG seed")->default_val(42);
    app.add_option("--m", m, "Matrix rows")->default_val(64);
    app.add_option("--n", n, "Matrix cols")->default_val(64);
    app.add_option("--k", k, "Inner dimension")->default_val(64);
    app.add_option("--kl", kl, "Lower bandwidth")->default_val(2);
    app.add_option("--ku", ku, "Upper bandwidth")->default_val(2);
    app.add_option("--incx", incx, "Increment for x")->default_val(1);
    app.add_option("--incy", incy, "Increment for y")->default_val(1);
    app.add_option("--ld-pad", ld_pad, "Leading dimension padding")->default_val(0);
    app.add_option("--mb", mb, "PBLAS row block size")->default_val(0);
    app.add_option("--nb", nb_val, "PBLAS col block size")->default_val(0);
    app.add_option("--threshold", threshold, "Error threshold (exit nonzero if exceeded)");
    app.add_option("--format", format, "Output format: text/json/csv")->default_val("text");
    app.add_option("--reference", reference, "Which side is reference: a or b")->default_val("a");

    CLI11_PARSE(app, argc, argv);

    /* --list */
    if (list_flag) {
        const char *last_cat = nullptr;
        for (int i = 0; i < num_routines; ++i) {
            if (!last_cat || std::strcmp(routines[i].category, last_cat) != 0) {
                last_cat = routines[i].category;
                std::printf("\n--- %s ---\n", last_cat);
            }
            std::printf("  %-14s %s  [%s]\n",
                        routines[i].name, routines[i].description,
                        routines[i].formula);
        }
        return 0;
    }

    /* Validate required options */
    if (routine_name.empty()) {
        std::fprintf(stderr, "Error: --routine is required\n");
        return 1;
    }
    if (lib_a_path.empty() || lib_b_path.empty()) {
        std::fprintf(stderr, "Error: --lib-a and --lib-b are required\n");
        return 1;
    }
    if (conv_lib_a_path.empty() || conv_lib_b_path.empty()) {
        std::fprintf(stderr, "Error: --conv-lib-a and --conv-lib-b are required\n");
        return 1;
    }
    if (typesize_a == 0 || typesize_b == 0) {
        std::fprintf(stderr, "Error: --typesize-a and --typesize-b are required\n");
        return 1;
    }

    /* Preload libraries */
    auto handles_a = preload_libs(preload_a);
    auto handles_b = preload_libs(preload_b);

    /* Open libraries */
    void *lib_a = dlopen(lib_a_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib_a) {
        std::fprintf(stderr, "dlopen(A): %s\n", dlerror());
        return 1;
    }
    void *lib_b = dlopen(lib_b_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib_b) {
        std::fprintf(stderr, "dlopen(B): %s\n", dlerror());
        return 1;
    }

    void *conv_a = dlopen(conv_lib_a_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!conv_a) {
        std::fprintf(stderr, "dlopen(conv_A): %s\n", dlerror());
        return 1;
    }
    void *conv_b = dlopen(conv_lib_b_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!conv_b) {
        std::fprintf(stderr, "dlopen(conv_B): %s\n", dlerror());
        return 1;
    }

    /* Build TesterCtx for each side */
    auto build_ctx = [](void *conv, int typesize, bool complex_mode,
                         const std::string &abi_str, int p) -> TesterCtx {
        TesterCtx ctx;
        ctx.prec = static_cast<mpfr_prec_t>(p);
        ctx.typesize = static_cast<std::size_t>(typesize);
        ctx.to_mpfr = reinterpret_cast<custom_to_mpfr_fn>(
            load_sym(conv, "custom_to_mpfr"));
        ctx.from_mpfr = reinterpret_cast<mpfr_to_custom_fn>(
            load_sym(conv, "mpfr_to_custom"));
        ctx.complex_mode = complex_mode;
        if (complex_mode) {
            ctx.to_mpfr_complex = reinterpret_cast<custom_to_mpfr_complex_fn>(
                load_sym(conv, "custom_to_mpfr_complex"));
            ctx.from_mpfr_complex = reinterpret_cast<mpfr_to_custom_complex_fn>(
                load_sym(conv, "mpfr_to_custom_complex"));
            ctx.complex_return_abi = (abi_str == "register")
                ? ComplexReturnABI::Register : ComplexReturnABI::Hidden;
        }
        return ctx;
    };

    TesterCtx ctx_a = build_ctx(conv_a, typesize_a, complex_a, complex_abi_a, prec);
    TesterCtx ctx_b = build_ctx(conv_b, typesize_b, complex_b, complex_abi_b, prec);

    /* Build TestParams */
    TestParams params;
    params.m = m; params.n = n; params.k = k;
    params.kl = kl; params.ku = ku;
    params.incx = incx; params.incy = incy;
    params.ld_pad = ld_pad;
    params.mb = mb; params.nb = nb_val;
    params.seed = seed;

    /* Build MirrorConfig */
    MirrorConfig config;
    config.prec = static_cast<mpfr_prec_t>(prec);
    config.reference = reference;
    config.threshold = threshold;
    config.format = format;

    /* Dispatch */
    bool exceeded_threshold = false;

    if (is_batch(routine_name)) {
        bool need_prefix_a = sym_a.empty();
        bool need_prefix_b = sym_b.empty();
        if (need_prefix_a && sym_prefix_a.empty()) {
            std::fprintf(stderr, "Error: batch mode requires --sym-prefix-a or --sym-a\n");
            return 1;
        }
        if (need_prefix_b && sym_prefix_b.empty()) {
            std::fprintf(stderr, "Error: batch mode requires --sym-prefix-b or --sym-b\n");
            return 1;
        }

        for (int i = 0; i < num_routines; ++i) {
            if (!category_matches(routines[i].category, routine_name))
                continue;

            /* Skip complex routines if side doesn't have complex enabled */
            const char *cat = routines[i].category;
            bool is_complex_routine = (std::strncmp(cat, "cblas", 5) == 0 ||
                                        std::strncmp(cat, "clapack", 7) == 0 ||
                                        std::strncmp(cat, "cpblas", 6) == 0 ||
                                        std::strncmp(cat, "cscalapack", 10) == 0);
            if (is_complex_routine && (!complex_a || !complex_b))
                continue;

            MirrorSide side_a;
            side_a.ctx = ctx_a;
            side_a.lib = lib_a;
            side_a.sym = need_prefix_a ? derive_sym(sym_prefix_a, routines[i]) : sym_a;
            side_a.label = "A";

            MirrorSide side_b;
            side_b.ctx = ctx_b;
            side_b.lib = lib_b;
            side_b.sym = need_prefix_b ? derive_sym(sym_prefix_b, routines[i]) : sym_b;
            side_b.label = "B";

            run_routine(routines[i], side_a, side_b, params, config);
        }
    } else {
        /* Single routine */
        const MirrorRoutineEntry *r = find_routine(routine_name.c_str());
        if (!r) {
            std::fprintf(stderr, "Unknown routine: %s\n", routine_name.c_str());
            return 1;
        }

        MirrorSide side_a;
        side_a.ctx = ctx_a;
        side_a.lib = lib_a;
        side_a.sym = !sym_a.empty() ? sym_a :
                     !sym_prefix_a.empty() ? derive_sym(sym_prefix_a, *r) : "";
        side_a.label = "A";
        if (side_a.sym.empty()) {
            std::fprintf(stderr, "Error: --sym-a or --sym-prefix-a required\n");
            return 1;
        }

        MirrorSide side_b;
        side_b.ctx = ctx_b;
        side_b.lib = lib_b;
        side_b.sym = !sym_b.empty() ? sym_b :
                     !sym_prefix_b.empty() ? derive_sym(sym_prefix_b, *r) : "";
        side_b.label = "B";
        if (side_b.sym.empty()) {
            std::fprintf(stderr, "Error: --sym-b or --sym-prefix-b required\n");
            return 1;
        }

        run_routine(*r, side_a, side_b, params, config);
    }

    /* Cleanup */
    dlclose(conv_b);
    dlclose(conv_a);
    dlclose(lib_b);
    dlclose(lib_a);
    close_libs(handles_b);
    close_libs(handles_a);

    return exceeded_threshold ? 1 : 0;
}
