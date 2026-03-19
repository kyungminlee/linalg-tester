/* mumps_tester: tests a sparse direct solver (MUMPS-style) by checking
 * residual norms ||b - Ax|| / ||b|| in L1, L2, and Linf.
 *
 * The solver library must export a function with this signature:
 *   int sparse_solve(int n, int nnz, int *irn, int *jcn,
 *                    void *a, void *rhs, int sym);
 * where irn/jcn are 1-based COO indices, rhs is overwritten with the
 * solution, and sym is 0 (unsym), 1 (SPD), or 2 (general symmetric).
 */

#include "reference.h"
#include "tester_utils.h"

#include "../third_party/CLI11.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <mpfr.h>

/* ------------------------------------------------------------------ */
/* Sparse matrix generation (COO, 1-based)                              */
/* ------------------------------------------------------------------ */

struct SparseCOO {
    int n;
    int nnz;
    std::vector<int> irn; /* 1-based row indices */
    std::vector<int> jcn; /* 1-based col indices */
    void *vals;           /* nnz values, each typesize bytes */
};

/* Generate a random sparse matrix in COO format.
 * - If sym==0: general unsymmetric, density controls fill ratio.
 * - If sym==1: SPD — diagonally dominant with A = L + L^T + D.
 * - If sym==2: general symmetric — stores lower triangle only.
 *
 * For sym==1,2 only the lower triangle (including diagonal) is stored. */
static SparseCOO gen_sparse_matrix(int n, double density, int sym,
                                    std::size_t typesize,
                                    mpfr_to_custom_fn from_mpfr,
                                    mpfr_prec_t prec, unsigned *seed)
{
    SparseCOO coo;
    coo.n = n;

    /* First pass: decide which entries exist */
    std::vector<std::pair<int,int>> entries;

    if (sym == 0) {
        /* Unsymmetric: diagonal always present, off-diag with probability density */
        for (int i = 1; i <= n; ++i)
            entries.push_back({i, i}); /* diagonal */
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i != j) {
                    double r = static_cast<double>(rand_r(seed)) /
                               static_cast<double>(RAND_MAX);
                    if (r < density)
                        entries.push_back({i, j});
                }
    } else {
        /* Symmetric (SPD or general): store lower triangle + diagonal */
        for (int i = 1; i <= n; ++i)
            entries.push_back({i, i});
        for (int i = 2; i <= n; ++i)
            for (int j = 1; j < i; ++j) {
                double r = static_cast<double>(rand_r(seed)) /
                           static_cast<double>(RAND_MAX);
                if (r < density)
                    entries.push_back({i, j});
            }
    }

    coo.nnz = static_cast<int>(entries.size());
    coo.irn.resize(coo.nnz);
    coo.jcn.resize(coo.nnz);
    coo.vals = std::malloc(static_cast<std::size_t>(coo.nnz) * typesize);
    if (!coo.vals) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    /* Count off-diagonal entries per row (for diagonal dominance) */
    std::vector<double> row_sum(n + 1, 0.0);

    char *vp = static_cast<char *>(coo.vals);
    for (int k = 0; k < coo.nnz; ++k) {
        coo.irn[k] = entries[k].first;
        coo.jcn[k] = entries[k].second;

        if (coo.irn[k] == coo.jcn[k]) {
            /* Diagonal: fill later for SPD */
            if (sym == 1) {
                mpfr_set_d(val, 0.0, MPFR_RNDN); /* placeholder */
            } else {
                double d = static_cast<double>(rand_r(seed)) /
                           static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                /* Make diagonal dominant for general matrices too */
                d += (d >= 0) ? static_cast<double>(n) : -static_cast<double>(n);
                mpfr_set_d(val, d, MPFR_RNDN);
            }
        } else {
            double d = static_cast<double>(rand_r(seed)) /
                       static_cast<double>(RAND_MAX) * 2.0 - 1.0;
            mpfr_set_d(val, d, MPFR_RNDN);

            double absd = (d >= 0) ? d : -d;
            row_sum[coo.irn[k]] += absd;
            if (sym != 0) /* symmetric: also affects the transposed row */
                row_sum[coo.jcn[k]] += absd;
        }
        from_mpfr(vp + static_cast<std::size_t>(k) * typesize, val, MPFR_RNDN);
    }

    /* For SPD: set diagonal = sum of absolute off-diag + small offset */
    if (sym == 1) {
        for (int k = 0; k < coo.nnz; ++k) {
            if (coo.irn[k] == coo.jcn[k]) {
                double diag_val = row_sum[coo.irn[k]] + 1.0 +
                    static_cast<double>(rand_r(seed)) / static_cast<double>(RAND_MAX);
                mpfr_set_d(val, diag_val, MPFR_RNDN);
                from_mpfr(vp + static_cast<std::size_t>(k) * typesize, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    return coo;
}

/* ------------------------------------------------------------------ */
/* Compute b = A*x using MPFR (for generating the RHS)                  */
/* For symmetric storage (sym!=0), entries (i,j) with i!=j contribute  */
/* to both row i and row j.                                             */
/* ------------------------------------------------------------------ */

static void *compute_rhs(const SparseCOO &coo, const void *x, int sym,
                          std::size_t typesize,
                          custom_to_mpfr_fn to_mpfr,
                          mpfr_to_custom_fn from_mpfr,
                          mpfr_prec_t prec)
{
    int n = coo.n;

    /* Convert x to MPFR */
    auto *x_mpfr = new mpfr_t[n];
    auto *b_mpfr = new mpfr_t[n];
    for (int i = 0; i < n; ++i) {
        mpfr_init2(x_mpfr[i], prec);
        mpfr_init2(b_mpfr[i], prec);
        mpfr_set_d(b_mpfr[i], 0.0, MPFR_RNDN);
    }

    const char *xp = static_cast<const char *>(x);
    for (int i = 0; i < n; ++i)
        to_mpfr(x_mpfr[i], xp + static_cast<std::size_t>(i) * typesize);

    /* SpMV: b = A*x */
    mpfr_t av, prod;
    mpfr_init2(av, prec);
    mpfr_init2(prod, prec);

    const char *vp = static_cast<const char *>(coo.vals);
    for (int k = 0; k < coo.nnz; ++k) {
        int i = coo.irn[k] - 1;
        int j = coo.jcn[k] - 1;
        to_mpfr(av, vp + static_cast<std::size_t>(k) * typesize);

        mpfr_mul(prod, av, x_mpfr[j], MPFR_RNDN);
        mpfr_add(b_mpfr[i], b_mpfr[i], prod, MPFR_RNDN);

        if (sym != 0 && i != j) {
            /* Symmetric: a(i,j) also means a(j,i) */
            mpfr_mul(prod, av, x_mpfr[i], MPFR_RNDN);
            mpfr_add(b_mpfr[j], b_mpfr[j], prod, MPFR_RNDN);
        }
    }

    /* Convert b back to custom type */
    void *b = std::malloc(static_cast<std::size_t>(n) * typesize);
    if (!b) { std::perror("malloc"); std::exit(EXIT_FAILURE); }
    char *bp = static_cast<char *>(b);
    for (int i = 0; i < n; ++i)
        from_mpfr(bp + static_cast<std::size_t>(i) * typesize, b_mpfr[i], MPFR_RNDN);

    mpfr_clear(av);
    mpfr_clear(prod);
    for (int i = 0; i < n; ++i) {
        mpfr_clear(x_mpfr[i]);
        mpfr_clear(b_mpfr[i]);
    }
    delete[] x_mpfr;
    delete[] b_mpfr;

    return b;
}

/* ------------------------------------------------------------------ */
/* Expand symmetric COO to full COO for residual computation            */
/* ------------------------------------------------------------------ */

static SparseCOO expand_symmetric(const SparseCOO &lower, std::size_t typesize)
{
    SparseCOO full;
    full.n = lower.n;

    /* Count: diag entries once, off-diag twice */
    int extra = 0;
    for (int k = 0; k < lower.nnz; ++k)
        if (lower.irn[k] != lower.jcn[k])
            ++extra;

    full.nnz = lower.nnz + extra;
    full.irn.resize(full.nnz);
    full.jcn.resize(full.nnz);
    full.vals = std::malloc(static_cast<std::size_t>(full.nnz) * typesize);
    if (!full.vals) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    const char *lv = static_cast<const char *>(lower.vals);
    char *fv = static_cast<char *>(full.vals);
    int idx = 0;

    for (int k = 0; k < lower.nnz; ++k) {
        full.irn[idx] = lower.irn[k];
        full.jcn[idx] = lower.jcn[k];
        std::memcpy(fv + static_cast<std::size_t>(idx) * typesize,
                    lv + static_cast<std::size_t>(k) * typesize, typesize);
        ++idx;

        if (lower.irn[k] != lower.jcn[k]) {
            full.irn[idx] = lower.jcn[k];
            full.jcn[idx] = lower.irn[k];
            std::memcpy(fv + static_cast<std::size_t>(idx) * typesize,
                        lv + static_cast<std::size_t>(k) * typesize, typesize);
            ++idx;
        }
    }

    return full;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    CLI::App app{"MUMPS sparse solver tester — checks residual norms "
                 "||b-Ax||/||b|| in L1, L2, Linf"};

    std::string lib_path, solve_sym, conv_lib_path;
    std::vector<std::string> preloads;
    std::size_t typesize = 0;
    int n = 64;
    int sym = 0;
    double density = 0.1;
    unsigned seed = 42;
    mpfr_prec_t prec = 256;

    app.add_option("--lib",       lib_path,      "Shared library with sparse_solve")->required();
    app.add_option("--solve-sym", solve_sym,     "Symbol name for solver (e.g. sparse_solve)")->required();
    app.add_option("--conv-lib",  conv_lib_path, "Library exporting custom_to_mpfr/mpfr_to_custom")->required();
    app.add_option("--typesize",  typesize,      "sizeof of the custom scalar type")->required();
    app.add_option("--preload",   preloads,      "Libraries to preload (may be specified multiple times)");
    app.add_option("--n",         n,             "Matrix dimension (default 64)");
    app.add_option("--sym",       sym,           "Symmetry: 0=unsym, 1=SPD, 2=general symmetric (default 0)");
    app.add_option("--density",   density,       "Sparsity density for off-diag (default 0.1)");
    app.add_option("--seed",      seed,          "Random seed (default 42)");
    app.add_option("--prec",      prec,          "MPFR working precision in bits (default 256)");

    CLI11_PARSE(app, argc, argv);

    /* Preload dependencies */
    auto preload_handles = preload_libs(preloads);

    /* Load solver and conversion libraries */
    void *solver_lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!solver_lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", lib_path.c_str(), dlerror());
        return EXIT_FAILURE;
    }
    void *conv_lib = dlopen(conv_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!conv_lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", conv_lib_path.c_str(), dlerror());
        dlclose(solver_lib);
        return EXIT_FAILURE;
    }

    auto *solve_fn = reinterpret_cast<sparse_solve_fn>(
                         load_sym(solver_lib, solve_sym.c_str()));
    auto *to_mpfr  = reinterpret_cast<custom_to_mpfr_fn>(
                         load_sym(conv_lib, "custom_to_mpfr"));
    auto *from_mpfr= reinterpret_cast<mpfr_to_custom_fn>(
                         load_sym(conv_lib, "mpfr_to_custom"));

    TesterCtx ctx{ prec, typesize, to_mpfr, from_mpfr };

    const char *sym_names[] = {"unsymmetric", "SPD", "general symmetric"};
    std::printf("=== Sparse solver test: n=%d, sym=%d (%s), density=%.2f ===\n",
                n, sym, sym_names[sym], density);

    /* Generate sparse matrix */
    unsigned seed_mat = seed;
    SparseCOO coo = gen_sparse_matrix(n, density, sym, typesize,
                                       from_mpfr, prec, &seed_mat);
    std::printf("  Matrix: n=%d, nnz=%d (stored), density=%.4f\n",
                coo.n, coo.nnz,
                static_cast<double>(coo.nnz) / (static_cast<double>(n) * n));

    /* Generate known solution x */
    unsigned seed_x = seed + 100;
    void *x_true = gen_random_array(n, typesize, from_mpfr, prec, &seed_x);

    /* Compute b = A*x using MPFR */
    void *b = compute_rhs(coo, x_true, sym, typesize, to_mpfr, from_mpfr, prec);

    /* Copy b into rhs (solver overwrites it) */
    void *rhs = std::malloc(static_cast<std::size_t>(n) * typesize);
    if (!rhs) { std::perror("malloc"); return EXIT_FAILURE; }
    std::memcpy(rhs, b, static_cast<std::size_t>(n) * typesize);

    /* Call the solver */
    int rc = solve_fn(n, coo.nnz, coo.irn.data(), coo.jcn.data(),
                      coo.vals, rhs, sym);
    if (rc != 0) {
        std::fprintf(stderr, "Solver returned error code %d\n", rc);
        std::free(x_true); std::free(b); std::free(rhs); std::free(coo.vals);
        dlclose(conv_lib); dlclose(solver_lib);
        close_libs(preload_handles);
        return EXIT_FAILURE;
    }

    /* Compute residual.  For symmetric storage we need full COO for
     * the residual computation (reference_sparse_residual uses plain SpMV). */
    ResidualResult res;
    if (sym != 0) {
        SparseCOO full = expand_symmetric(coo, typesize);
        res = reference_sparse_residual(ctx, n, full.nnz,
                                         full.irn.data(), full.jcn.data(),
                                         full.vals, b, rhs);
        std::free(full.vals);
    } else {
        res = reference_sparse_residual(ctx, n, coo.nnz,
                                         coo.irn.data(), coo.jcn.data(),
                                         coo.vals, b, rhs);
    }

    std::printf("  ||b-Ax||_1   / ||b||_1   = %.6e\n", res.l1);
    std::printf("  ||b-Ax||_2   / ||b||_2   = %.6e\n", res.l2);
    std::printf("  ||b-Ax||_inf / ||b||_inf = %.6e\n", res.linf);

    std::free(x_true);
    std::free(b);
    std::free(rhs);
    std::free(coo.vals);
    dlclose(conv_lib);
    dlclose(solver_lib);
    close_libs(preload_handles);
    return EXIT_SUCCESS;
}
