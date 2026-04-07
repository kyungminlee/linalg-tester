#include "core/tester_ctx.h"
#include "core/loader.h"
#include "blas/level3.h"
#include "blas/level2.h"
#include "blas/level1.h"

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
    const char *category;
    const char *description;
    const char *formula;
    void (*test_fn)(const TesterCtx &, void *, const char *, const TestParams &, const std::string &);
};

static const RoutineEntry routines[] = {
    // Level 3
    {"gemm",  "blas3", "General matrix multiply",     "C = alpha*op(A)*op(B) + beta*C", test_gemm},
    {"trsm",  "blas3", "Triangular solve",            "op(A)*X = alpha*B",              test_trsm},
    {"symm",  "blas3", "Symmetric matrix multiply",   "C = alpha*A*B + beta*C",         test_symm},
    {"syrk",  "blas3", "Symmetric rank-k update",     "C = alpha*A*A^T + beta*C",       test_syrk},
    {"syr2k", "blas3", "Symmetric rank-2k update",    "C = alpha*A*B^T + alpha*B*A^T + beta*C", test_syr2k},
    {"trmm",  "blas3", "Triangular matrix multiply",  "B = alpha*op(A)*B",              test_trmm},
    // Level 2
    {"gemv",  "blas2", "General matrix-vector",       "y = alpha*op(A)*x + beta*y",     test_gemv},
    {"symv",  "blas2", "Symmetric matrix-vector",     "y = alpha*A*x + beta*y",         test_symv},
    {"trmv",  "blas2", "Triangular matrix-vector",    "x = op(A)*x",                    test_trmv},
    {"trsv",  "blas2", "Triangular vector solve",     "op(A)*x = b",                    test_trsv},
    {"ger",   "blas2", "General rank-1 update",       "A = alpha*x*y^T + A",            test_ger},
    {"syr",   "blas2", "Symmetric rank-1 update",     "A = alpha*x*x^T + A",            test_syr},
    {"syr2",  "blas2", "Symmetric rank-2 update",     "A = alpha*x*y^T + alpha*y*x^T + A", test_syr2},
    {"gbmv",  "blas2", "General banded matrix-vector","y = alpha*op(A)*x + beta*y",     test_gbmv},
    {"sbmv",  "blas2", "Symmetric banded MV",         "y = alpha*A*x + beta*y",         test_sbmv},
    {"tbmv",  "blas2", "Triangular banded MV",        "x = op(A)*x",                    test_tbmv},
    {"tbsv",  "blas2", "Triangular banded solve",     "op(A)*x = b",                    test_tbsv},
    {"spmv",  "blas2", "Symmetric packed MV",         "y = alpha*A*x + beta*y",         test_spmv},
    {"tpmv",  "blas2", "Triangular packed MV",        "x = op(A)*x",                    test_tpmv},
    {"tpsv",  "blas2", "Triangular packed solve",     "op(A)*x = b",                    test_tpsv},
    {"spr",   "blas2", "Symmetric packed rank-1",     "A = alpha*x*x^T + A",            test_spr},
    {"spr2",  "blas2", "Symmetric packed rank-2",     "A = alpha*x*y^T + alpha*y*x^T + A", test_spr2},
    // Level 1
    {"rotg",  "blas1", "Rotation generation",         "Generate Givens rotation",       test_rotg},
    {"rotmg", "blas1", "Modified Givens generation",  "Generate modified Givens",       test_rotmg},
    {"rot",   "blas1", "Apply rotation",              "[x;y] = [c s;-s c]*[x;y]",      test_rot},
    {"rotm",  "blas1", "Apply modified Givens",       "Apply modified Givens rotation", test_rotm},
    {"swap",  "blas1", "Swap vectors",                "x <-> y",                        test_swap},
    {"scal",  "blas1", "Scale vector",                "x = alpha*x",                    test_scal},
    {"copy",  "blas1", "Copy vector",                 "y = x",                          test_copy},
    {"axpy",  "blas1", "Vector addition",             "y = alpha*x + y",                test_axpy},
    {"dot",   "blas1", "Dot product",                 "x^T * y",                        test_dot},
    {"nrm2",  "blas1", "Euclidean norm",              "||x||_2",                        test_nrm2},
    {"asum",  "blas1", "Sum of absolute values",      "sum(|x_i|)",                     test_asum},
    {"iamax", "blas1", "Index of max absolute value",  "argmax_i(|x_i|)",               test_iamax},
};

static constexpr int num_routines = sizeof(routines) / sizeof(routines[0]);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string derive_sym(const std::string &prefix, const char *routine_name) {
    // IAMAX: BLAS convention is i<prefix>amax (e.g., idamax_, isamax_)
    if (std::strcmp(routine_name, "iamax") == 0)
        return "i" + prefix + "amax_";
    return prefix + routine_name + "_";
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
    double threshold = -1.0;
    std::string format = "text";

    app.add_flag("--list", list_flag, "List all supported routines and exit");
    app.add_option("--routine", routine_name, "Routine name (or all/blas1/blas2/blas3)");
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
    params.seed = seed;

    // Determine which routines to run
    bool is_batch = (routine_name == "all" || routine_name == "blas1" ||
                     routine_name == "blas2" || routine_name == "blas3");

    if (is_batch && sym_prefix.empty() && sym_name.empty()) {
        std::fprintf(stderr, "Error: batch mode requires --sym-prefix (or --sym for single routine)\n");
        return EXIT_FAILURE;
    }

    if (is_batch) {
        // Batch mode: run all matching routines
        for (int i = 0; i < num_routines; ++i) {
            const auto &r = routines[i];
            if (routine_name != "all" && routine_name != r.category)
                continue;
            std::string sym = sym_prefix.empty() ? sym_name : derive_sym(sym_prefix, r.name);
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
            sym = derive_sym(sym_prefix, entry->name);
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
