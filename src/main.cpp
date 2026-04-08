#include "core/tester_ctx.h"
#include "core/loader.h"
#include "blas/level3.h"
#include "blas/level2.h"
#include "blas/level1.h"
#include "lapack/factorizations.h"
#include "lapack/solvers.h"
#include "lapack/eigenvalues.h"
#include "lapack/auxiliary.h"
#include "blacs/blacs.h"
#include "pblas/pblas.h"

#include "../third_party/CLI11.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <mpfr.h>

// ---------------------------------------------------------------------------
// Dispatch table
// ---------------------------------------------------------------------------

struct RoutineEntry {
    const char *name;
    const char *fortran_name;  // base name for symbol derivation (NULL = use name)
    const char *category;
    const char *description;
    const char *formula;
    void (*test_fn)(const TesterCtx &, void *, const char *, const TestParams &, const std::string &);
};

static const RoutineEntry routines[] = {
    // Level 3
    {"gemm",  nullptr, "blas3", "General matrix multiply",     "C = alpha*op(A)*op(B) + beta*C", test_gemm},
    {"trsm",  nullptr, "blas3", "Triangular solve",            "op(A)*X = alpha*B",              test_trsm},
    {"symm",  nullptr, "blas3", "Symmetric matrix multiply",   "C = alpha*A*B + beta*C",         test_symm},
    {"syrk",  nullptr, "blas3", "Symmetric rank-k update",     "C = alpha*A*A^T + beta*C",       test_syrk},
    {"syr2k", nullptr, "blas3", "Symmetric rank-2k update",    "C = alpha*A*B^T + alpha*B*A^T + beta*C", test_syr2k},
    {"trmm",  nullptr, "blas3", "Triangular matrix multiply",  "B = alpha*op(A)*B",              test_trmm},
    // Level 2
    {"gemv",  nullptr, "blas2", "General matrix-vector",       "y = alpha*op(A)*x + beta*y",     test_gemv},
    {"symv",  nullptr, "blas2", "Symmetric matrix-vector",     "y = alpha*A*x + beta*y",         test_symv},
    {"trmv",  nullptr, "blas2", "Triangular matrix-vector",    "x = op(A)*x",                    test_trmv},
    {"trsv",  nullptr, "blas2", "Triangular vector solve",     "op(A)*x = b",                    test_trsv},
    {"ger",   nullptr, "blas2", "General rank-1 update",       "A = alpha*x*y^T + A",            test_ger},
    {"syr",   nullptr, "blas2", "Symmetric rank-1 update",     "A = alpha*x*x^T + A",            test_syr},
    {"syr2",  nullptr, "blas2", "Symmetric rank-2 update",     "A = alpha*x*y^T + alpha*y*x^T + A", test_syr2},
    {"gbmv",  nullptr, "blas2", "General banded matrix-vector","y = alpha*op(A)*x + beta*y",     test_gbmv},
    {"sbmv",  nullptr, "blas2", "Symmetric banded MV",         "y = alpha*A*x + beta*y",         test_sbmv},
    {"tbmv",  nullptr, "blas2", "Triangular banded MV",        "x = op(A)*x",                    test_tbmv},
    {"tbsv",  nullptr, "blas2", "Triangular banded solve",     "op(A)*x = b",                    test_tbsv},
    {"spmv",  nullptr, "blas2", "Symmetric packed MV",         "y = alpha*A*x + beta*y",         test_spmv},
    {"tpmv",  nullptr, "blas2", "Triangular packed MV",        "x = op(A)*x",                    test_tpmv},
    {"tpsv",  nullptr, "blas2", "Triangular packed solve",     "op(A)*x = b",                    test_tpsv},
    {"spr",   nullptr, "blas2", "Symmetric packed rank-1",     "A = alpha*x*x^T + A",            test_spr},
    {"spr2",  nullptr, "blas2", "Symmetric packed rank-2",     "A = alpha*x*y^T + alpha*y*x^T + A", test_spr2},
    // Level 1
    {"rotg",  nullptr, "blas1", "Rotation generation",         "Generate Givens rotation",       test_rotg},
    {"rotmg", nullptr, "blas1", "Modified Givens generation",  "Generate modified Givens",       test_rotmg},
    {"rot",   nullptr, "blas1", "Apply rotation",              "[x;y] = [c s;-s c]*[x;y]",      test_rot},
    {"rotm",  nullptr, "blas1", "Apply modified Givens",       "Apply modified Givens rotation", test_rotm},
    {"swap",  nullptr, "blas1", "Swap vectors",                "x <-> y",                        test_swap},
    {"scal",  nullptr, "blas1", "Scale vector",                "x = alpha*x",                    test_scal},
    {"copy",  nullptr, "blas1", "Copy vector",                 "y = x",                          test_copy},
    {"axpy",  nullptr, "blas1", "Vector addition",             "y = alpha*x + y",                test_axpy},
    {"dot",   nullptr, "blas1", "Dot product",                 "x^T * y",                        test_dot},
    {"nrm2",  nullptr, "blas1", "Euclidean norm",              "||x||_2",                        test_nrm2},
    {"asum",  nullptr, "blas1", "Sum of absolute values",      "sum(|x_i|)",                     test_asum},
    {"iamax", nullptr, "blas1", "Index of max absolute value",  "argmax_i(|x_i|)",               test_iamax},
    // Complex-only Level 3
    {"hemm",  nullptr, "cblas3", "Hermitian matrix multiply",   "C = alpha*A*B + beta*C (A Herm)",           test_hemm},
    {"herk",  nullptr, "cblas3", "Hermitian rank-k update",     "C = alpha*A*A^H + beta*C",                  test_herk},
    {"her2k", nullptr, "cblas3", "Hermitian rank-2k update",    "C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C", test_her2k},
    // Complex-only Level 2
    {"hemv",  nullptr, "cblas2", "Hermitian matrix-vector",     "y = alpha*A*x + beta*y (A Herm)",            test_hemv},
    {"hbmv",  nullptr, "cblas2", "Hermitian banded MV",         "y = alpha*A*x + beta*y (A Herm banded)",     test_hbmv},
    {"hpmv",  nullptr, "cblas2", "Hermitian packed MV",         "y = alpha*A*x + beta*y (A Herm packed)",     test_hpmv},
    {"geru",  nullptr, "cblas2", "Unconjugated rank-1 update",  "A = alpha*x*y^T + A",                       test_geru},
    {"gerc",  nullptr, "cblas2", "Conjugated rank-1 update",    "A = alpha*x*conj(y)^T + A",                 test_gerc},
    {"her",   nullptr, "cblas2", "Hermitian rank-1 update",     "A = alpha*x*conj(x)^T + A (alpha real)",    test_her},
    {"hpr",   nullptr, "cblas2", "Hermitian packed rank-1",     "A = alpha*x*conj(x)^T + A (packed)",        test_hpr},
    {"her2",  nullptr, "cblas2", "Hermitian rank-2 update",     "A = alpha*x*conj(y)^T + conj(alpha)*y*conj(x)^T + A", test_her2},
    {"hpr2",  nullptr, "cblas2", "Hermitian packed rank-2",     "A = alpha*x*conj(y)^T + conj(alpha)*y*conj(x)^T + A", test_hpr2},
    // Complex-only Level 1
    {"dotc",   nullptr, "cblas1", "Conjugated dot product",     "conj(x)^T * y",                             test_dotc},
    {"dotu",   nullptr, "cblas1", "Unconjugated dot product",   "x^T * y (no conjugation)",                  test_dotu},
    {"crot",   nullptr, "cblas1", "Real rotation of complex",   "[x;y] = [c s;-s c]*[x;y] (c,s real)",      test_crot},
    {"crscal", nullptr, "cblas1", "Real scale of complex",      "x = alpha*x (alpha real)",                  test_crscal},
    // LAPACK Factorizations
    {"getrf", nullptr, "lapack_fact", "LU factorization",             "PA = LU",                        test_getrf},
    {"potrf", nullptr, "lapack_fact", "Cholesky factorization",       "A = LL^T",                       test_potrf},
    {"sytrf", nullptr, "lapack_fact", "Symmetric LDL^T",              "A = LDL^T (Bunch-Kaufman)",      test_sytrf},
    {"geqrf", nullptr, "lapack_fact", "QR factorization",             "A = QR",                         test_geqrf},
    {"gelqf", nullptr, "lapack_fact", "LQ factorization",             "A = LQ",                         test_gelqf},
    {"gebrd", nullptr, "lapack_fact", "Bidiagonal reduction",         "A = U*B*V^T",                    test_gebrd},
    {"sytrd", nullptr, "lapack_fact", "Tridiagonal reduction",        "A = Q*T*Q^T",                    test_sytrd},
    // LAPACK Solvers
    {"gesv",  nullptr, "lapack_solve","General solve",                "AX = B (LU)",                    test_gesv},
    {"posv",  nullptr, "lapack_solve","SPD solve",                    "AX = B (Cholesky)",              test_posv},
    {"sysv",  nullptr, "lapack_solve","Symmetric solve",              "AX = B (LDL^T)",                 test_sysv},
    {"gbsv",  nullptr, "lapack_solve","Banded solve",                 "AX = B (banded LU)",             test_gbsv},
    {"gtsv",  nullptr, "lapack_solve","Tridiagonal solve",            "AX = B (tridiag)",               test_gtsv},
    {"gels",  nullptr, "lapack_solve","Least squares",                "min||Ax-b||_2",                  test_gels},
    {"gelsd", nullptr, "lapack_solve","Least squares (SVD)",          "min||Ax-b||_2 (SVD)",            test_gelsd},
    // LAPACK Eigenvalue / SVD
    {"syev",  nullptr, "lapack_eig",  "Symmetric eigenvalue",         "A = V*D*V^T",                    test_syev},
    {"syevd", nullptr, "lapack_eig",  "Symmetric eigenvalue (D&C)",   "A = V*D*V^T",                    test_syevd},
    {"syevr", nullptr, "lapack_eig",  "Symmetric eigenvalue (MRRR)",  "A = V*D*V^T",                    test_syevr},
    {"geev",  nullptr, "lapack_eig",  "General eigenvalue",           "A*V = V*diag(w)",                test_geev},
    {"gees",  nullptr, "lapack_eig",  "Schur decomposition",          "A = V*T*V^T",                    test_gees},
    {"gesvd", nullptr, "lapack_eig",  "SVD",                          "A = U*S*V^T",                    test_gesvd},
    {"gesdd", nullptr, "lapack_eig",  "SVD (D&C)",                    "A = U*S*V^T",                    test_gesdd},
    {"sygv",  nullptr, "lapack_eig",  "Gen. symmetric eigenvalue",    "A*x = lambda*B*x",               test_sygv},
    // LAPACK Auxiliary
    {"getrs", nullptr, "lapack_aux",  "Solve from LU factors",        "op(A)*X = B",                    test_getrs},
    {"getri", nullptr, "lapack_aux",  "Inverse from LU",              "A^{-1}",                         test_getri},
    {"potrs", nullptr, "lapack_aux",  "Solve from Cholesky",          "A*X = B",                        test_potrs},
    {"potri", nullptr, "lapack_aux",  "Inverse from Cholesky",        "A^{-1}",                         test_potri},
    {"orgqr", nullptr, "lapack_aux",  "Generate Q from QR",           "Q from reflectors",              test_orgqr},
    {"ormqr", nullptr, "lapack_aux",  "Multiply by Q",                "C = op(Q)*C",                    test_ormqr},
    {"gecon", nullptr, "lapack_aux",  "Condition number estimate",    "rcond(A)",                       test_gecon},
    {"lange", nullptr, "lapack_aux",  "Matrix norm",                  "||A||",                          test_lange},
    {"lansy", nullptr, "lapack_aux",  "Symmetric matrix norm",        "||A||_sym",                      test_lansy},
    {"lacpy", nullptr, "lapack_aux",  "Matrix copy",                  "B = A",                          test_lacpy},
    {"laswp", nullptr, "lapack_aux",  "Row permutations",             "P*A",                            test_laswp},
    // Complex LAPACK Factorizations
    {"cgetrf", "getrf", "clapack_fact", "Complex LU factorization",        "PA = LU",                   test_cgetrf},
    {"cpotrf", "potrf", "clapack_fact", "Complex Cholesky",                "A = LL^H",                  test_cpotrf},
    {"cgeqrf", "geqrf", "clapack_fact", "Complex QR factorization",        "A = QR",                    test_cgeqrf},
    {"cgelqf", "gelqf", "clapack_fact", "Complex LQ factorization",        "A = LQ",                    test_cgelqf},
    {"cgebrd", "gebrd", "clapack_fact", "Complex bidiagonal reduction",    "A = U*B*V^H",               test_cgebrd},
    {"hetrf",  nullptr, "clapack_fact", "Hermitian LDL^H",                 "A = LDL^H",                 test_hetrf},
    {"hetrd",  nullptr, "clapack_fact", "Hermitian tridiag reduction",     "A = Q*T*Q^H",               test_hetrd},
    // Complex LAPACK Solvers
    {"cgesv",  "gesv",  "clapack_solve","Complex general solve",           "AX = B (LU)",               test_cgesv},
    {"cposv",  "posv",  "clapack_solve","HPD solve",                       "AX = B (Cholesky)",         test_cposv},
    {"cgbsv",  "gbsv",  "clapack_solve","Complex banded solve",           "AX = B (banded LU)",        test_cgbsv},
    {"cgtsv",  "gtsv",  "clapack_solve","Complex tridiagonal solve",      "AX = B (tridiag)",          test_cgtsv},
    {"cgels",  "gels",  "clapack_solve","Complex least squares",          "min||Ax-b||_2",             test_cgels},
    {"cgelsd", "gelsd", "clapack_solve","Complex least squares (SVD)",    "min||Ax-b||_2 (SVD)",       test_cgelsd},
    {"hesv",   nullptr, "clapack_solve","Hermitian solve",                "AX = B (LDL^H)",            test_hesv},
    // Complex LAPACK Eigenvalue / SVD
    {"heev",   nullptr, "clapack_eig",  "Hermitian eigenvalue",           "A = V*D*V^H",               test_heev},
    {"heevd",  nullptr, "clapack_eig",  "Hermitian eigenvalue (D&C)",     "A = V*D*V^H",               test_heevd},
    {"heevr",  nullptr, "clapack_eig",  "Hermitian eigenvalue (MRRR)",    "A = V*D*V^H",               test_heevr},
    {"hegv",   nullptr, "clapack_eig",  "Gen. Hermitian eigenvalue",      "A*x = lambda*B*x",          test_hegv},
    {"cgesvd", "gesvd", "clapack_eig",  "Complex SVD",                    "A = U*S*V^H",               test_cgesvd},
    {"cgesdd", "gesdd", "clapack_eig",  "Complex SVD (D&C)",              "A = U*S*V^H",               test_cgesdd},
    {"cgeev",  "geev",  "clapack_eig",  "Complex general eigenvalue",     "A*V = V*diag(w)",           test_cgeev},
    {"cgees",  "gees",  "clapack_eig",  "Complex Schur decomposition",    "A = V*T*V^H",               test_cgees},
    // Complex LAPACK Auxiliary
    {"cgetrs", "getrs", "clapack_aux",  "Complex solve from LU",          "op(A)*X = B",               test_cgetrs},
    {"cgetri", "getri", "clapack_aux",  "Complex inverse from LU",        "A^{-1}",                    test_cgetri},
    {"cpotrs", "potrs", "clapack_aux",  "Complex solve from Cholesky",    "A*X = B",                   test_cpotrs},
    {"cpotri", "potri", "clapack_aux",  "Complex inverse from Cholesky",  "A^{-1}",                    test_cpotri},
    {"ungqr",  nullptr, "clapack_aux",  "Generate unitary Q from QR",     "Q from reflectors",         test_ungqr},
    {"unmqr",  nullptr, "clapack_aux",  "Multiply by unitary Q",          "C = op(Q)*C",               test_unmqr},
    {"cgecon", "gecon", "clapack_aux",  "Complex condition number",       "rcond(A)",                  test_cgecon},
    {"clange", "lange", "clapack_aux",  "Complex matrix norm",            "||A||",                     test_clange},
    {"lanhe",  nullptr, "clapack_aux",  "Hermitian matrix norm",          "||A||_herm",                test_lanhe},
    {"clacpy", "lacpy", "clapack_aux",  "Complex matrix copy",            "B = A",                     test_clacpy},
    {"claswp", "laswp", "clapack_aux",  "Complex row permutations",       "P*A",                       test_claswp},
    // BLACS Context
    {"blacs_setup",  "blacs_pinfo", "blacs_ctx",     "BLACS context lifecycle",  "PINFO/GRIDINIT/GRIDEXIT",  test_blacs_setup},
    // BLACS Point-to-Point
    {"blacs_gesd2d", "gesd2d",      "blacs_p2p",     "BLACS point-to-point",     "xGESD2D/xGERV2D",         test_blacs_gesd2d},
    // BLACS Broadcast
    {"blacs_gebs2d", "gebs2d",      "blacs_bcast",   "BLACS broadcast",          "xGEBS2D/xGEBR2D",         test_blacs_gebs2d},
    // BLACS Combine
    {"blacs_gsum2d", "gsum2d",      "blacs_combine", "BLACS global sum",         "xGSUM2D",                  test_blacs_gsum2d},
    {"blacs_gamx2d", "gamx2d",      "blacs_combine", "BLACS global max",         "xGAMX2D",                  test_blacs_gamx2d},
    {"blacs_gamn2d", "gamn2d",      "blacs_combine", "BLACS global min",         "xGAMN2D",                  test_blacs_gamn2d},
    // PBLAS Level 3
    {"pgemm",  "gemm",  "pblas3", "Parallel matrix multiply",     "C = alpha*op(A)*op(B) + beta*C",              test_pgemm},
    {"ptrsm",  "trsm",  "pblas3", "Parallel triangular solve",    "op(A)*X = alpha*B",                           test_ptrsm},
    {"psymm",  "symm",  "pblas3", "Parallel symmetric multiply",  "C = alpha*A*B + beta*C",                      test_psymm},
    {"psyrk",  "syrk",  "pblas3", "Parallel symmetric rank-k",    "C = alpha*A*A^T + beta*C",                    test_psyrk},
    {"psyr2k", "syr2k", "pblas3", "Parallel symmetric rank-2k",   "C = alpha*AB^T + alpha*BA^T + beta*C",        test_psyr2k},
    {"ptrmm",  "trmm",  "pblas3", "Parallel triangular multiply", "B = alpha*op(A)*B",                           test_ptrmm},
    // PBLAS Level 2
    {"pgemv",  "gemv",  "pblas2", "Parallel matrix-vector",       "y = alpha*op(A)*x + beta*y",                  test_pgemv},
    {"psymv",  "symv",  "pblas2", "Parallel symmetric MV",        "y = alpha*A*x + beta*y",                      test_psymv},
    {"ptrmv",  "trmv",  "pblas2", "Parallel triangular MV",       "x = op(A)*x",                                 test_ptrmv},
    {"ptrsv",  "trsv",  "pblas2", "Parallel triangular solve",    "op(A)*x = b",                                 test_ptrsv},
    {"pger",   "ger",   "pblas2", "Parallel rank-1 update",       "A = alpha*x*y^T + A",                         test_pger},
    {"psyr",   "syr",   "pblas2", "Parallel symmetric rank-1",    "A = alpha*x*x^T + A",                         test_psyr},
    {"psyr2",  "syr2",  "pblas2", "Parallel symmetric rank-2",    "A = alpha*x*y^T + alpha*y*x^T + A",           test_psyr2},
    // PBLAS Level 1
    {"pswap",  "swap",  "pblas1", "Parallel swap vectors",        "x <-> y",                                     test_pswap},
    {"pscal",  "scal",  "pblas1", "Parallel scale vector",        "x = alpha*x",                                 test_pscal},
    {"pcopy",  "copy",  "pblas1", "Parallel copy vector",         "y = x",                                       test_pcopy},
    {"paxpy",  "axpy",  "pblas1", "Parallel vector addition",     "y = alpha*x + y",                             test_paxpy},
    {"pdot",   "dot",   "pblas1", "Parallel dot product",         "x^T * y",                                     test_pdot},
    {"pnrm2",  "nrm2",  "pblas1", "Parallel Euclidean norm",     "||x||_2",                                     test_pnrm2},
    {"pasum",  "asum",  "pblas1", "Parallel absolute sum",        "sum(|x_i|)",                                  test_pasum},
};

static constexpr int num_routines = sizeof(routines) / sizeof(routines[0]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string derive_sym(const std::string &prefix, const RoutineEntry &r) {
    const char *routine_name = r.name;
    const char *base = r.fortran_name ? r.fortran_name : r.name;
    // IAMAX: BLAS convention is i<prefix>amax (e.g., idamax_, isamax_)
    if (std::strcmp(routine_name, "iamax") == 0)
        return "i" + prefix + "amax_";
    // CROT: csrot_/zdrot_ (prefix 'c' -> 'cs', prefix 'z' -> 'zd')
    if (std::strcmp(routine_name, "crot") == 0) {
        if (prefix == "c") return "csrot_";
        if (prefix == "z") return "zdrot_";
        return prefix + "rot_";
    }
    // CRSCAL: csscal_/zdscal_ (prefix 'c' -> 'cs', prefix 'z' -> 'zd')
    if (std::strcmp(routine_name, "crscal") == 0) {
        if (prefix == "c") return "csscal_";
        if (prefix == "z") return "zdscal_";
        return prefix + "scal_";
    }
    // BLACS context routines: no type prefix, just fortran_name + "_"
    if (r.fortran_name && std::strncmp(r.fortran_name, "blacs_", 6) == 0)
        return std::string(r.fortran_name) + "_";
    // PBLAS: p{prefix}{base}_ (e.g., pdgemm_)
    if (std::strncmp(r.category, "pblas", 5) == 0)
        return "p" + prefix + base + "_";
    return prefix + base + "_";
}

static const RoutineEntry *find_routine(const char *name) {
    for (int i = 0; i < num_routines; ++i) {
        if (std::strcmp(routines[i].name, name) == 0)
            return &routines[i];
    }
    return nullptr;
}

static void run_routine(const RoutineEntry &r, const TesterCtx &ctx,
                        void *lib, const std::string &sym,
                        const TestParams &params, const std::string &format) {
    std::printf("=== %s (%s) ===\n", r.name, r.description);
    r.test_fn(ctx, lib, sym.c_str(), params, format);
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    CLI::App app{"linalg-tester: accuracy tester for BLAS/LAPACK"};

    bool list_flag = false;
    bool complex_flag = false;
    std::string routine_name;
    std::string lib_path;
    std::string sym_name;
    std::string sym_prefix;
    std::string conv_lib_path;
    int typesize = 0;
    std::vector<std::string> preload_paths;
    int prec = 256;
    unsigned seed = 42;
    int m = 64, n = 64, k = 64;
    int kl = 2, ku = 2;
    int incx = 1, incy = 1;
    int ld_pad = 0;
    int mb = 0, nb_opt = 0;
    double threshold = -1.0;
    std::string format = "text";
    std::string complex_return_abi_str = "hidden";

    app.add_flag("--list", list_flag, "List all supported routines and exit");
    app.add_flag("--complex", complex_flag, "Enable complex mode");
    app.add_option("--complex-return-abi", complex_return_abi_str, "Complex return ABI for DOTC/DOTU: hidden or register")->default_val("hidden");
    app.add_option("--routine", routine_name, "Routine name (or all/blas1/blas2/blas3/cblas1/cblas2/cblas3/lapack/lapack_fact/lapack_solve/lapack_eig/lapack_aux)");
    app.add_option("--lib", lib_path, "Path to shared library under test");
    app.add_option("--sym", sym_name, "Symbol name to test (e.g. dgemm_)");
    app.add_option("--sym-prefix", sym_prefix, "Symbol prefix for batch mode (e.g. d -> dgemm_)");
    app.add_option("--conv-lib", conv_lib_path, "Path to conversion library");
    app.add_option("--typesize", typesize, "Size of the floating-point type in bytes");
    app.add_option("--preload", preload_paths, "Libraries to preload (repeatable)");
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
    app.add_option("--mb", mb, "PBLAS row block size (0=auto)")->default_val(0);
    app.add_option("--nb", nb_opt, "PBLAS column block size (0=auto)")->default_val(0);
    app.add_option("--threshold", threshold, "Error threshold (exit nonzero if exceeded)");
    app.add_option("--format", format, "Output format (text/json/csv)")->default_val("text");

    CLI11_PARSE(app, argc, argv);

    // --list: print all routines grouped by category
    if (list_flag) {
        const char *cur_cat = "";
        for (int i = 0; i < num_routines; ++i) {
            const auto &r = routines[i];
            if (std::strcmp(r.category, cur_cat) != 0) {
                cur_cat = r.category;
                if (std::strcmp(cur_cat, "blas3") == 0)
                    std::printf("BLAS Level 3:\n");
                else if (std::strcmp(cur_cat, "blas2") == 0)
                    std::printf("BLAS Level 2:\n");
                else if (std::strcmp(cur_cat, "blas1") == 0)
                    std::printf("BLAS Level 1:\n");
                else if (std::strcmp(cur_cat, "lapack_fact") == 0)
                    std::printf("LAPACK Factorizations:\n");
                else if (std::strcmp(cur_cat, "lapack_solve") == 0)
                    std::printf("LAPACK Solvers:\n");
                else if (std::strcmp(cur_cat, "lapack_eig") == 0)
                    std::printf("LAPACK Eigenvalue/SVD:\n");
                else if (std::strcmp(cur_cat, "lapack_aux") == 0)
                    std::printf("LAPACK Auxiliary:\n");
                else if (std::strcmp(cur_cat, "cblas3") == 0)
                    std::printf("Complex BLAS Level 3:\n");
                else if (std::strcmp(cur_cat, "cblas2") == 0)
                    std::printf("Complex BLAS Level 2:\n");
                else if (std::strcmp(cur_cat, "cblas1") == 0)
                    std::printf("Complex BLAS Level 1:\n");
                else if (std::strcmp(cur_cat, "clapack_fact") == 0)
                    std::printf("Complex LAPACK Factorizations:\n");
                else if (std::strcmp(cur_cat, "clapack_solve") == 0)
                    std::printf("Complex LAPACK Solvers:\n");
                else if (std::strcmp(cur_cat, "clapack_eig") == 0)
                    std::printf("Complex LAPACK Eigenvalue/SVD:\n");
                else if (std::strcmp(cur_cat, "clapack_aux") == 0)
                    std::printf("Complex LAPACK Auxiliary:\n");
                else if (std::strcmp(cur_cat, "blacs_ctx") == 0)
                    std::printf("BLACS Context:\n");
                else if (std::strcmp(cur_cat, "blacs_p2p") == 0)
                    std::printf("BLACS Point-to-Point:\n");
                else if (std::strcmp(cur_cat, "blacs_bcast") == 0)
                    std::printf("BLACS Broadcast:\n");
                else if (std::strcmp(cur_cat, "blacs_combine") == 0)
                    std::printf("BLACS Combine:\n");
                else if (std::strcmp(cur_cat, "pblas3") == 0)
                    std::printf("PBLAS Level 3:\n");
                else if (std::strcmp(cur_cat, "pblas2") == 0)
                    std::printf("PBLAS Level 2:\n");
                else if (std::strcmp(cur_cat, "pblas1") == 0)
                    std::printf("PBLAS Level 1:\n");
            }
            std::printf("  %-8s %-35s %s\n", r.name, r.description, r.formula);
        }
        return 0;
    }

    // Validate required options for non-list mode
    if (routine_name.empty()) {
        std::fprintf(stderr, "Error: --routine is required (or use --list)\n");
        return EXIT_FAILURE;
    }
    if (lib_path.empty()) {
        std::fprintf(stderr, "Error: --lib is required\n");
        return EXIT_FAILURE;
    }
    if (conv_lib_path.empty()) {
        std::fprintf(stderr, "Error: --conv-lib is required\n");
        return EXIT_FAILURE;
    }
    if (typesize <= 0) {
        std::fprintf(stderr, "Error: --typesize is required and must be positive\n");
        return EXIT_FAILURE;
    }

    // Preload libraries
    auto preloaded = preload_libs(preload_paths);

    // Load the library under test
    void *lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", lib_path.c_str(), dlerror());
        return EXIT_FAILURE;
    }

    // Load conversion library
    void *conv = dlopen(conv_lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!conv) {
        std::fprintf(stderr, "dlopen(%s): %s\n", conv_lib_path.c_str(), dlerror());
        return EXIT_FAILURE;
    }

    auto to_mpfr_fn = reinterpret_cast<custom_to_mpfr_fn>(load_sym(conv, "custom_to_mpfr"));
    auto from_mpfr_fn = reinterpret_cast<mpfr_to_custom_fn>(load_sym(conv, "mpfr_to_custom"));

    // Build context
    TesterCtx ctx;
    ctx.prec = static_cast<mpfr_prec_t>(prec);
    ctx.typesize = static_cast<std::size_t>(typesize);
    ctx.to_mpfr = to_mpfr_fn;
    ctx.from_mpfr = from_mpfr_fn;

    if (complex_flag) {
        ctx.complex_mode = true;
        ctx.to_mpfr_complex = reinterpret_cast<custom_to_mpfr_complex_fn>(
            load_sym(conv, "custom_to_mpfr_complex"));
        ctx.from_mpfr_complex = reinterpret_cast<mpfr_to_custom_complex_fn>(
            load_sym(conv, "mpfr_to_custom_complex"));
        if (complex_return_abi_str == "register")
            ctx.complex_return_abi = ComplexReturnABI::Register;
        else
            ctx.complex_return_abi = ComplexReturnABI::Hidden;
    }

    // Build params
    TestParams params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.kl = kl;
    params.ku = ku;
    params.incx = incx;
    params.incy = incy;
    params.ld_pad = ld_pad;
    params.mb = mb;
    params.nb = nb_opt;
    params.seed = seed;

    // Determine which routines to run
    bool is_batch = (routine_name == "all" || routine_name == "blas1" ||
                     routine_name == "blas2" || routine_name == "blas3" ||
                     routine_name == "cblas1" || routine_name == "cblas2" ||
                     routine_name == "cblas3" ||
                     routine_name == "lapack" || routine_name == "lapack_fact" ||
                     routine_name == "lapack_solve" || routine_name == "lapack_eig" ||
                     routine_name == "lapack_aux" ||
                     routine_name == "clapack" || routine_name == "clapack_fact" ||
                     routine_name == "clapack_solve" || routine_name == "clapack_eig" ||
                     routine_name == "clapack_aux" ||
                     routine_name == "blacs" || routine_name == "blacs_ctx" ||
                     routine_name == "blacs_p2p" || routine_name == "blacs_bcast" ||
                     routine_name == "blacs_combine" ||
                     routine_name == "pblas" || routine_name == "pblas3" ||
                     routine_name == "pblas2" || routine_name == "pblas1");

    if (is_batch && sym_prefix.empty() && sym_name.empty()) {
        std::fprintf(stderr, "Error: batch mode requires --sym-prefix (or --sym for single routine)\n");
        return EXIT_FAILURE;
    }

    if (is_batch) {
        // Batch mode: run all matching routines
        for (int i = 0; i < num_routines; ++i) {
            const auto &r = routines[i];
            if (routine_name != "all" && routine_name != r.category) {
                /* "lapack" matches all lapack_* categories */
                if (routine_name == "lapack" &&
                    std::strncmp(r.category, "lapack_", 7) == 0)
                    ; /* match */
                /* "clapack" matches all clapack_* categories */
                else if (routine_name == "clapack" &&
                         std::strncmp(r.category, "clapack_", 8) == 0)
                    ; /* match */
                /* "blacs" matches all blacs_* categories */
                else if (routine_name == "blacs" &&
                         std::strncmp(r.category, "blacs_", 6) == 0)
                    ; /* match */
                /* "pblas" matches all pblas* categories */
                else if (routine_name == "pblas" &&
                         std::strncmp(r.category, "pblas", 5) == 0)
                    ; /* match */
                else
                    continue;
            }
            /* Skip complex categories when not in complex mode */
            if (!ctx.complex_mode &&
                (std::strncmp(r.category, "cblas", 5) == 0 ||
                 std::strncmp(r.category, "clapack_", 8) == 0))
                continue;
            /* Skip BLACS categories unless explicitly requested.
               BLACS requires a ScaLAPACK library, not a BLAS/LAPACK library. */
            if (std::strncmp(r.category, "blacs_", 6) == 0 &&
                routine_name != "blacs" &&
                std::strncmp(routine_name.c_str(), "blacs_", 6) != 0)
                continue;
            /* Skip PBLAS categories unless explicitly requested.
               PBLAS requires a ScaLAPACK library. */
            if (std::strncmp(r.category, "pblas", 5) == 0 &&
                routine_name != "pblas" &&
                std::strncmp(routine_name.c_str(), "pblas", 5) != 0)
                continue;
            std::string sym = sym_prefix.empty() ? sym_name : derive_sym(sym_prefix, r);
            run_routine(r, ctx, lib, sym, params, format);
        }
    } else {
        // Single routine mode
        const RoutineEntry *entry = find_routine(routine_name.c_str());
        if (!entry) {
            std::fprintf(stderr, "Error: unknown routine '%s' (use --list to see available routines)\n",
                         routine_name.c_str());
            return EXIT_FAILURE;
        }

        std::string sym = sym_name;
        if (sym.empty() && !sym_prefix.empty()) {
            sym = derive_sym(sym_prefix, *entry);
        }
        if (sym.empty()) {
            std::fprintf(stderr, "Error: --sym or --sym-prefix is required\n");
            return EXIT_FAILURE;
        }

        run_routine(*entry, ctx, lib, sym, params, format);
    }

    // Cleanup
    dlclose(conv);
    dlclose(lib);
    close_libs(preloaded);

    return 0;
}
