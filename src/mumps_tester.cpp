/* mumps_tester: tests a MUMPS-style sparse direct solver by checking
 * residual norms ||b - Ax|| / ||b|| in L1, L2, and Linf.
 *
 * Calls the MUMPS entry point (e.g. dmumps_c, ddmumps_c, ...) directly
 * through the MUMPS C struct, loaded at runtime via dlopen/dlsym.
 *
 * The struct field offsets are computed at runtime from --typesize so that
 * custom-precision MUMPS builds (dd, qd, ...) work without recompilation.
 * For the standard double case (typesize=8) the computed layout is verified
 * against DMUMPS_STRUC_C at startup.
 *
 * Usage:
 *   mumps_tester --lib <libdmumps_seq.so> --mumps-sym dmumps_c \
 *                --conv-lib <double_conv.so> --typesize 8 \
 *                --n 64 --sym 0 --density 0.1
 */

#include "reference.h"
#include "tester_utils.h"
#include "dmumps_struc.h"   /* for offsetof verification when typesize==8 */

#include "../third_party/CLI11.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <mpfr.h>

/* ------------------------------------------------------------------ */
/* Runtime MUMPS struct layout (MUMPS 5.6.2 field order)                */
/* ------------------------------------------------------------------ */

static std::size_t align_up(std::size_t off, std::size_t a)
{
    return (off + a - 1) & ~(a - 1);
}

/* Offsets of the MUMPS struct fields we actually touch. */
struct MumpsOffsets {
    std::size_t sym, par, job, comm_fortran;
    std::size_t icntl;       /* start of icntl[60] */
    std::size_t n, nz, nnz;
    std::size_t irn, jcn, a; /* pointers */
    std::size_t rhs;         /* pointer  */
    std::size_t nrhs, lrhs;
    std::size_t infog;       /* start of infog[80] */
    std::size_t total;       /* sizeof the whole struct */
};

/* Walk the MUMPS 5.6.2 field list, advancing `off` with alignment,
 * and record offsets of the fields we need. */
static MumpsOffsets compute_mumps_layout(std::size_t real_size,
                                          std::size_t real_align)
{
    MumpsOffsets F{};
    std::size_t off = 0;

    /* helpers — each aligns, records start, advances off */
    auto add_int = [&](std::size_t cnt = 1) {
        off = align_up(off, 4); auto o = off; off += 4 * cnt; return o;
    };
    auto add_int8 = [&](std::size_t cnt = 1) {
        off = align_up(off, 8); auto o = off; off += 8 * cnt; return o;
    };
    auto add_ptr = [&](std::size_t cnt = 1) {
        off = align_up(off, 8); auto o = off; off += 8 * cnt; return o;
    };
    auto add_real = [&](std::size_t cnt = 1) {
        off = align_up(off, real_align); auto o = off;
        off += real_size * cnt; return o;
    };
    auto add_char = [&](std::size_t cnt) {
        auto o = off; off += cnt; return o;
    };

    /* macros that discard the returned offset */
    #define SKIP(...) (void)(__VA_ARGS__)

    /* --- MUMPS 5.6.2 field list, in declaration order --- */

    F.sym            = add_int();           /* sym              */
    F.par            = add_int();           /* par              */
    F.job            = add_int();           /* job              */
    F.comm_fortran   = add_int();           /* comm_fortran     */
    F.icntl          = add_int(60);         /* icntl[60]        */
    SKIP(              add_int(500));        /* keep[500]        */
    SKIP(              add_real(15));        /* cntl[15]         */
    SKIP(              add_real(230));       /* dkeep[230]       */
    SKIP(              add_int8(150));       /* keep8[150]       */
    F.n              = add_int();           /* n                */
    SKIP(              add_int());          /* nblk             */
    SKIP(              add_int());          /* nz_alloc         */
    F.nz             = add_int();           /* nz               */
    F.nnz            = add_int8();          /* nnz              */
    F.irn            = add_ptr();           /* *irn             */
    F.jcn            = add_ptr();           /* *jcn             */
    F.a              = add_ptr();           /* *a               */
    SKIP(              add_int());          /* nz_loc           */
    SKIP(              add_int8());         /* nnz_loc          */
    SKIP(              add_ptr(3));         /* irn_loc,jcn_loc,a_loc */
    SKIP(              add_int());          /* nelt             */
    SKIP(              add_ptr(3));         /* eltptr,eltvar,a_elt   */
    SKIP(              add_ptr(2));         /* blkptr,blkvar    */
    SKIP(              add_ptr());          /* perm_in          */
    SKIP(              add_ptr(2));         /* sym_perm,uns_perm */
    SKIP(              add_ptr(2));         /* colsca,rowsca    */
    SKIP(              add_int(2));         /* colsca_from_mumps,rowsca_from_mumps */
    F.rhs            = add_ptr();           /* *rhs             */
    SKIP(              add_ptr(4));         /* redrhs,rhs_sparse,sol_loc,rhs_loc */
    SKIP(              add_ptr(4));         /* irhs_sparse,irhs_ptr,isol_loc,irhs_loc */
    F.nrhs           = add_int();           /* nrhs             */
    F.lrhs           = add_int();           /* lrhs             */
    SKIP(              add_int(5));         /* lredrhs,nz_rhs,lsol_loc,nloc_rhs,lrhs_loc */
    SKIP(              add_int(7));         /* schur_mloc..npcol */
    SKIP(              add_int(80));        /* info[80]         */
    F.infog          = add_int(80);         /* infog[80]        */
    SKIP(              add_real(40));       /* rinfo[40]        */
    SKIP(              add_real(40));       /* rinfog[40]       */
    SKIP(              add_int());          /* deficiency       */
    SKIP(              add_ptr(2));         /* pivnul_list,mapping */
    SKIP(              add_int());          /* size_schur       */
    SKIP(              add_ptr(2));         /* listvar_schur,schur */
    SKIP(              add_int());          /* instance_number  */
    SKIP(              add_ptr());          /* wk_user          */
    SKIP(              add_char(32));       /* version_number   */
    SKIP(              add_char(256));      /* ooc_tmpdir       */
    SKIP(              add_char(64));       /* ooc_prefix       */
    SKIP(              add_char(256));      /* write_problem    */
    SKIP(              add_int());          /* lwk_user         */
    SKIP(              add_char(256));      /* save_dir         */
    SKIP(              add_char(256));      /* save_prefix      */
    SKIP(              add_int(40));        /* metis_options[40] */

    #undef SKIP

    /* struct alignment = max alignment of any member */
    std::size_t struct_align = 8;
    if (real_align > struct_align) struct_align = real_align;
    F.total = align_up(off, struct_align);

    return F;
}

/* Verify computed layout matches the real DMUMPS_STRUC_C for double. */
static bool verify_layout_double(const MumpsOffsets &F)
{
    bool ok = true;
    #define CHK(field) do { \
        if (F.field != offsetof(DMUMPS_STRUC_C, field)) { \
            std::fprintf(stderr, "  layout mismatch: %s computed=%zu actual=%zu\n", \
                         #field, F.field, offsetof(DMUMPS_STRUC_C, field)); \
            ok = false; \
        } \
    } while (0)

    CHK(sym); CHK(par); CHK(job); CHK(comm_fortran);
    CHK(icntl); CHK(n); CHK(nz); CHK(nnz);
    CHK(irn); CHK(jcn); CHK(a); CHK(rhs);
    CHK(nrhs); CHK(lrhs); CHK(infog);

    if (F.total != sizeof(DMUMPS_STRUC_C)) {
        std::fprintf(stderr, "  layout mismatch: total computed=%zu actual=%zu\n",
                     F.total, sizeof(DMUMPS_STRUC_C));
        ok = false;
    }

    #undef CHK
    return ok;
}

/* ------------------------------------------------------------------ */
/* Byte-level accessors for the MUMPS struct buffer                     */
/* ------------------------------------------------------------------ */

static void wr_int (char *buf, std::size_t off, int     v) { std::memcpy(buf+off, &v, 4); }
static void wr_i64 (char *buf, std::size_t off, int64_t v) { std::memcpy(buf+off, &v, 8); }
static void wr_ptr (char *buf, std::size_t off, void   *v) { std::memcpy(buf+off, &v, 8); }
static int  rd_int (const char *buf, std::size_t off) { int v; std::memcpy(&v, buf+off, 4); return v; }

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

    std::vector<std::pair<int,int>> entries;

    if (sym == 0) {
        for (int i = 1; i <= n; ++i)
            entries.push_back({i, i});
        for (int i = 1; i <= n; ++i)
            for (int j = 1; j <= n; ++j)
                if (i != j) {
                    double r = static_cast<double>(rand_r(seed)) /
                               static_cast<double>(RAND_MAX);
                    if (r < density)
                        entries.push_back({i, j});
                }
    } else {
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
    std::vector<double> row_sum(n + 1, 0.0);

    char *vp = static_cast<char *>(coo.vals);
    for (int k = 0; k < coo.nnz; ++k) {
        coo.irn[k] = entries[k].first;
        coo.jcn[k] = entries[k].second;

        if (coo.irn[k] == coo.jcn[k]) {
            if (sym == 1) {
                mpfr_set_d(val, 0.0, MPFR_RNDN);
            } else {
                double d = static_cast<double>(rand_r(seed)) /
                           static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                d += (d >= 0) ? static_cast<double>(n) : -static_cast<double>(n);
                mpfr_set_d(val, d, MPFR_RNDN);
            }
        } else {
            double d = static_cast<double>(rand_r(seed)) /
                       static_cast<double>(RAND_MAX) * 2.0 - 1.0;
            mpfr_set_d(val, d, MPFR_RNDN);
            double absd = (d >= 0) ? d : -d;
            row_sum[coo.irn[k]] += absd;
            if (sym != 0) row_sum[coo.jcn[k]] += absd;
        }
        from_mpfr(vp + static_cast<std::size_t>(k) * typesize, val, MPFR_RNDN);
    }

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
/* Compute b = A*x using MPFR                                          */
/* ------------------------------------------------------------------ */

static void *compute_rhs(const SparseCOO &coo, const void *x, int sym,
                          std::size_t typesize,
                          custom_to_mpfr_fn to_mpfr,
                          mpfr_to_custom_fn from_mpfr,
                          mpfr_prec_t prec)
{
    int n = coo.n;
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
            mpfr_mul(prod, av, x_mpfr[i], MPFR_RNDN);
            mpfr_add(b_mpfr[j], b_mpfr[j], prod, MPFR_RNDN);
        }
    }

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
    int extra = 0;
    for (int k = 0; k < lower.nnz; ++k)
        if (lower.irn[k] != lower.jcn[k]) ++extra;

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

/* The MUMPS entry point (e.g. dmumps_c) takes a pointer to its struct. */
extern "C" typedef void (*mumps_c_fn)(void *);

int main(int argc, char **argv)
{
    CLI::App app{"MUMPS sparse solver tester — checks residual norms "
                 "||b-Ax||/||b|| in L1, L2, Linf"};

    std::string lib_path, mumps_sym = "dmumps_c", conv_lib_path;
    std::vector<std::string> preloads;
    std::size_t typesize = 0;
    std::size_t real_align = 0;
    int n = 64;
    int sym = 0;
    double density = 0.1;
    unsigned seed = 42;
    mpfr_prec_t prec = 256;

    app.add_option("--lib",        lib_path,      "MUMPS shared library (e.g. libdmumps_seq.so)")->required();
    app.add_option("--mumps-sym",  mumps_sym,     "Symbol name for MUMPS entry point (default: dmumps_c)");
    app.add_option("--conv-lib",   conv_lib_path, "Library exporting custom_to_mpfr/mpfr_to_custom")->required();
    app.add_option("--typesize",   typesize,       "sizeof of the custom scalar type")->required();
    app.add_option("--real-align", real_align,     "Alignment of the real type (default: min(typesize,8))");
    app.add_option("--preload",    preloads,       "Libraries to preload (may be specified multiple times)");
    app.add_option("--n",          n,              "Matrix dimension (default 64)");
    app.add_option("--sym",        sym,            "Symmetry: 0=unsym, 1=SPD, 2=general symmetric (default 0)");
    app.add_option("--density",    density,        "Sparsity density for off-diag (default 0.1)");
    app.add_option("--seed",       seed,           "Random seed (default 42)");
    app.add_option("--prec",       prec,           "MPFR working precision in bits (default 256)");

    CLI11_PARSE(app, argc, argv);

    /* Default real_align: min(typesize, 8), rounded down to power of 2 */
    if (real_align == 0) {
        real_align = 1;
        while (real_align * 2 <= typesize && real_align * 2 <= 8)
            real_align *= 2;
    }

    /* Compute struct layout */
    MumpsOffsets F = compute_mumps_layout(typesize, real_align);

    /* Sanity-check against DMUMPS_STRUC_C when running standard double */
    if (typesize == sizeof(double) && real_align == alignof(double)) {
        if (!verify_layout_double(F)) {
            std::fprintf(stderr, "FATAL: computed layout does not match "
                         "DMUMPS_STRUC_C — struct definition may be stale.\n");
            return EXIT_FAILURE;
        }
    }

    /* Preload dependencies */
    auto preload_handles = preload_libs(preloads);

    /* Load MUMPS library and conversion library */
    void *mumps_lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!mumps_lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", lib_path.c_str(), dlerror());
        return EXIT_FAILURE;
    }
    void *conv_lib = dlopen(conv_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!conv_lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", conv_lib_path.c_str(), dlerror());
        dlclose(mumps_lib);
        return EXIT_FAILURE;
    }

    auto *mumps_fn = reinterpret_cast<mumps_c_fn>(
                         load_sym(mumps_lib, mumps_sym.c_str()));
    auto *to_mpfr  = reinterpret_cast<custom_to_mpfr_fn>(
                         load_sym(conv_lib, "custom_to_mpfr"));
    auto *from_mpfr= reinterpret_cast<mpfr_to_custom_fn>(
                         load_sym(conv_lib, "mpfr_to_custom"));

    TesterCtx ctx{ prec, typesize, to_mpfr, from_mpfr };

    const char *sym_names[] = {"unsymmetric", "SPD", "general symmetric"};
    std::printf("=== MUMPS test: n=%d, sym=%d (%s), density=%.2f ===\n",
                n, sym, sym_names[sym], density);

    /* Generate sparse matrix and known solution in custom type */
    unsigned seed_mat = seed;
    SparseCOO coo = gen_sparse_matrix(n, density, sym, typesize,
                                       from_mpfr, prec, &seed_mat);
    std::printf("  Matrix: n=%d, nnz=%d (stored), density=%.4f\n",
                coo.n, coo.nnz,
                static_cast<double>(coo.nnz) / (static_cast<double>(n) * n));

    unsigned seed_x = seed + 100;
    void *x_true = gen_random_array(n, typesize, from_mpfr, prec, &seed_x);
    void *b = compute_rhs(coo, x_true, sym, typesize, to_mpfr, from_mpfr, prec);

    /* rhs buffer — MUMPS overwrites it with the solution */
    void *rhs = std::malloc(static_cast<std::size_t>(n) * typesize);
    if (!rhs) { std::perror("malloc"); return EXIT_FAILURE; }
    std::memcpy(rhs, b, static_cast<std::size_t>(n) * typesize);

    /* --- Drive MUMPS via byte-level struct access --- */
    char *id = static_cast<char *>(std::calloc(1, F.total));
    if (!id) { std::perror("calloc"); return EXIT_FAILURE; }

    /* Phase 1: Initialise instance */
    wr_int(id, F.par, 1);
    wr_int(id, F.sym, sym);
    wr_int(id, F.comm_fortran, -987654);   /* MUMPS_USE_COMM_WORLD (seq) */
    wr_int(id, F.job, -1);
    mumps_fn(id);

    if (rd_int(id, F.infog) < 0) {
        std::fprintf(stderr, "MUMPS init failed: INFOG(1)=%d, INFOG(2)=%d\n",
                     rd_int(id, F.infog), rd_int(id, F.infog + 4));
        std::free(id);
        goto cleanup;
    }

    /* Suppress MUMPS stdout */
    wr_int(id, F.icntl + 0 * 4, -1);   /* error stream   */
    wr_int(id, F.icntl + 1 * 4, -1);   /* warning stream  */
    wr_int(id, F.icntl + 2 * 4, -1);   /* info stream     */
    wr_int(id, F.icntl + 3 * 4,  0);   /* print level     */

    /* Set assembled matrix — pass custom-type array directly */
    wr_int(id, F.n, n);
    wr_int(id, F.nz, coo.nnz);
    wr_i64(id, F.nnz, static_cast<int64_t>(coo.nnz));
    wr_ptr(id, F.irn, coo.irn.data());
    wr_ptr(id, F.jcn, coo.jcn.data());
    wr_ptr(id, F.a,   coo.vals);

    /* Set RHS — pass custom-type array directly */
    wr_int(id, F.nrhs, 1);
    wr_int(id, F.lrhs, n);
    wr_ptr(id, F.rhs,  rhs);

    /* Phase 2: Analyse + Factorise + Solve */
    wr_int(id, F.job, 6);
    mumps_fn(id);

    {
        int rc = rd_int(id, F.infog);
        if (rc < 0)
            std::fprintf(stderr, "MUMPS solve failed: INFOG(1)=%d, INFOG(2)=%d\n",
                         rc, rd_int(id, F.infog + 4));

        /* Phase 3: Destroy instance */
        wr_int(id, F.job, -2);
        mumps_fn(id);
        std::free(id);
        id = nullptr;

        if (rc < 0) goto cleanup;
    }

    /* rhs now contains the solution in custom type.
     * Compute residual ||b - A*x_solved|| / ||b|| in MPFR. */
    {
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
    }

cleanup:
    std::free(id);
    std::free(x_true);
    std::free(b);
    std::free(rhs);
    std::free(coo.vals);
    dlclose(conv_lib);
    dlclose(mumps_lib);
    close_libs(preload_handles);
    return EXIT_SUCCESS;
}
