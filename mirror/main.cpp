/* main.cpp -- Mirror tester: compare two BLAS libraries */

#include "mirror_ctx.h"
#include "derive_sym.h"
#include "blas/mirror_blas3.h"
#include "blas/mirror_blas2.h"
#include "blas/mirror_blas1.h"
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
    /* BLAS Level 3 (real) */
    {"gemm",   nullptr, "blas3", "General matrix multiply",     "C = alpha*op(A)*op(B) + beta*C", mirror_test_gemm},
    {"trsm",   nullptr, "blas3", "Triangular solve (matrix)",   "op(A)*X = alpha*B",              mirror_test_trsm},
    {"symm",   nullptr, "blas3", "Symmetric multiply",          "C = alpha*A*B + beta*C",         mirror_test_symm},
    {"syrk",   nullptr, "blas3", "Symmetric rank-k update",     "C = alpha*A*A^T + beta*C",       mirror_test_syrk},
    {"syr2k",  nullptr, "blas3", "Symmetric rank-2k update",    "C = alpha*A*B^T + ... + beta*C", mirror_test_syr2k},
    {"trmm",   nullptr, "blas3", "Triangular multiply (matrix)","B = alpha*op(A)*B",              mirror_test_trmm},
    /* BLAS Level 3 (complex-only) */
    {"hemm",   nullptr, "cblas3","Hermitian multiply",          "C = alpha*A*B + beta*C",         mirror_test_hemm},
    {"herk",   nullptr, "cblas3","Hermitian rank-k update",     "C = alpha*A*A^H + beta*C",       mirror_test_herk},
    {"her2k",  nullptr, "cblas3","Hermitian rank-2k update",    "C = alpha*A*B^H + ... + beta*C", mirror_test_her2k},
    /* BLAS Level 2 (real) */
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
    /* BLAS Level 2 (complex-only) */
    {"hemv",   nullptr, "cblas2","Hermitian matrix-vector",     "y = alpha*A*x + beta*y",         mirror_test_hemv},
    {"hbmv",   nullptr, "cblas2","Hermitian band mat-vec",      "y = alpha*A*x + beta*y",         mirror_test_hbmv},
    {"hpmv",   nullptr, "cblas2","Hermitian packed mat-vec",    "y = alpha*A*x + beta*y",         mirror_test_hpmv},
    {"geru",   nullptr, "cblas2","Unconjugated rank-1 update",  "A = alpha*x*y^T + A",            mirror_test_geru},
    {"gerc",   nullptr, "cblas2","Conjugated rank-1 update",    "A = alpha*x*y^H + A",            mirror_test_gerc},
    {"her",    nullptr, "cblas2","Hermitian rank-1 update",     "A = alpha*x*x^H + A",            mirror_test_her},
    {"hpr",    nullptr, "cblas2","Hermitian packed rank-1",     "A = alpha*x*x^H + A",            mirror_test_hpr},
    {"her2",   nullptr, "cblas2","Hermitian rank-2 update",     "A = alpha*x*y^H + conj(alpha)*y*x^H + A", mirror_test_her2},
    {"hpr2",   nullptr, "cblas2","Hermitian packed rank-2",     "A = alpha*x*y^H + conj(alpha)*y*x^H + A", mirror_test_hpr2},
    /* BLAS Level 1 (real) */
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
    /* BLAS Level 1 (complex-only) */
    {"dotc",   nullptr, "cblas1","Conjugated dot product",      "result = x^H*y",                 mirror_test_dotc},
    {"dotu",   nullptr, "cblas1","Unconjugated dot product",    "result = x^T*y",                 mirror_test_dotu},
    {"crot",   nullptr, "cblas1","Complex rotation apply",      "[x,y] = rot(x,y,c,s)",           mirror_test_crot},
    {"crscal", nullptr, "cblas1","Real scale of complex",       "x = alpha*x (alpha real)",        mirror_test_crscal},
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
        "all", "blas", "cblas",
        "blas1", "blas2", "blas3",
        "cblas1", "cblas2", "cblas3",
    };
    for (auto b : batches)
        if (name == b) return true;
    return false;
}

static bool category_matches(const char *cat, const std::string &batch) {
    if (batch == "all") return true;
    if (batch == "blas")  return std::strncmp(cat, "blas", 4) == 0;
    if (batch == "cblas") return std::strncmp(cat, "cblas", 5) == 0;
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
    CLI::App app{"mirror-tester: compare two BLAS libraries"};

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
            std::printf("  %-10s %s  [%s]\n",
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
            bool is_complex_routine = (std::strncmp(routines[i].category, "cblas", 5) == 0);
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
