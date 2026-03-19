/* trsm_tester: tests all (side, uplo, trans, diag) combinations for a custom TRSM. */

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

int main(int argc, char **argv)
{
    CLI::App app{"TRSM accuracy tester — tests all (side, uplo, trans, diag) combinations"};

    std::string lib_path, trsm_sym, conv_lib_path;
    std::vector<std::string> preloads;
    std::size_t typesize = 0;
    int m = 64, n = 64;
    unsigned seed = 42;
    mpfr_prec_t prec = 256;

    app.add_option("--lib",      lib_path,      "Shared library with custom trsm")->required();
    app.add_option("--trsm-sym", trsm_sym,      "Symbol name for trsm (e.g. mytrsm_)")->required();
    app.add_option("--conv-lib", conv_lib_path, "Library exporting custom_to_mpfr/mpfr_to_custom")->required();
    app.add_option("--typesize", typesize,       "sizeof of the custom scalar type")->required();
    app.add_option("--preload",  preloads,       "Libraries to preload (may be specified multiple times)");
    app.add_option("--m",        m,              "Matrix rows (default 64)");
    app.add_option("--n",        n,              "Matrix columns (default 64)");
    app.add_option("--seed",     seed,           "Random seed (default 42)");
    app.add_option("--prec",     prec,           "MPFR working precision in bits (default 256)");

    CLI11_PARSE(app, argc, argv);

    /* Preload dependencies */
    auto preload_handles = preload_libs(preloads);

    /* Load main library and conversion library */
    void *custom_lib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!custom_lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", lib_path.c_str(), dlerror());
        return EXIT_FAILURE;
    }
    void *conv_lib = dlopen(conv_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!conv_lib) {
        std::fprintf(stderr, "dlopen(%s): %s\n", conv_lib_path.c_str(), dlerror());
        dlclose(custom_lib);
        return EXIT_FAILURE;
    }

    auto *trsm_fn  = reinterpret_cast<trsm_fn_t>(
                         load_sym(custom_lib, trsm_sym.c_str()));
    auto *to_mpfr  = reinterpret_cast<custom_to_mpfr_fn>(
                         load_sym(conv_lib, "custom_to_mpfr"));
    auto *from_mpfr= reinterpret_cast<mpfr_to_custom_fn>(
                         load_sym(conv_lib, "mpfr_to_custom"));

    TesterCtx ctx{ prec, typesize, to_mpfr, from_mpfr };

    /* Test all 16 combinations of (side, uplo, trans, diag) */
    for (char side  : {'L', 'R'}) {
    for (char uplo  : {'U', 'L'}) {
    for (char trans : {'N', 'T'}) {
    for (char diag  : {'N', 'U'}) {
        int ka = (side == 'L') ? m : n;

        unsigned seed_tA  = seed + 10;
        unsigned seed_tB  = seed + 11;
        unsigned seed_tal = seed + 12;

        void *A     = gen_triangular_array(ka, uplo, diag,
                                            typesize, from_mpfr, prec, &seed_tA);
        void *B_in  = gen_random_array(m * n, typesize, from_mpfr, prec, &seed_tB);
        void *alpha = gen_random_array(1,     typesize, from_mpfr, prec, &seed_tal);

        void *X_out = std::malloc(static_cast<std::size_t>(m) * n * typesize);
        if (!X_out) { std::perror("malloc"); return EXIT_FAILURE; }
        std::memcpy(X_out, B_in, static_cast<std::size_t>(m) * n * typesize);

        int lda = ka, ldb = m;
        trsm_fn(&side, &uplo, &trans, &diag,
                &m, &n,
                alpha, A, &lda,
                X_out, &ldb,
                1, 1, 1, 1);

        ErrorResult err = reference_test_trsm(ctx, side, uplo, trans, diag,
                                               m, n, alpha, A, B_in, X_out);
        std::printf("[TRSM side=%c uplo=%c trans=%c diag=%c] "
                    "max_rel=%.6e  normwise=%.6e\n",
                    side, uplo, trans, diag,
                    err.max_relative, err.normwise_relative);

        std::free(A); std::free(B_in); std::free(X_out); std::free(alpha);
    }}}}

    dlclose(conv_lib);
    dlclose(custom_lib);
    close_libs(preload_handles);
    return EXIT_SUCCESS;
}
