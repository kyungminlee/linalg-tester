/* gemm_tester: tests all (transa, transb) combinations for a custom GEMM. */

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
    CLI::App app{"GEMM accuracy tester — tests all (opA, opB) combinations"};

    std::string lib_path, gemm_sym, conv_lib_path;
    std::vector<std::string> preloads;
    std::size_t typesize = 0;
    int m = 64, n = 64, k = 64;
    unsigned seed = 42;
    mpfr_prec_t prec = 256;

    app.add_option("--lib",      lib_path,      "Shared library with custom gemm")->required();
    app.add_option("--gemm-sym", gemm_sym,      "Symbol name for gemm (e.g. mygemm_)")->required();
    app.add_option("--conv-lib", conv_lib_path, "Library exporting custom_to_mpfr/mpfr_to_custom")->required();
    app.add_option("--typesize", typesize,       "sizeof of the custom scalar type")->required();
    app.add_option("--preload",  preloads,       "Libraries to preload (may be specified multiple times)");
    app.add_option("--m",        m,              "Matrix rows (default 64)");
    app.add_option("--n",        n,              "Matrix columns (default 64)");
    app.add_option("--k",        k,              "Inner dimension (default 64)");
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

    auto *gemm_fn  = reinterpret_cast<gemm_fn_t>(
                         load_sym(custom_lib, gemm_sym.c_str()));
    auto *to_mpfr  = reinterpret_cast<custom_to_mpfr_fn>(
                         load_sym(conv_lib, "custom_to_mpfr"));
    auto *from_mpfr= reinterpret_cast<mpfr_to_custom_fn>(
                         load_sym(conv_lib, "mpfr_to_custom"));

    TesterCtx ctx{ prec, typesize, to_mpfr, from_mpfr };

    /* Test all 4 combinations of (transa, transb) */
    for (char ta : {'N', 'T'}) {
        for (char tb : {'N', 'T'}) {
            unsigned seed_A  = seed;
            unsigned seed_B  = seed + 1;
            unsigned seed_C  = seed + 2;
            unsigned seed_ab = seed + 3;

            int rows_A = (ta == 'N') ? m : k;
            int cols_A = (ta == 'N') ? k : m;
            int rows_B = (tb == 'N') ? k : n;
            int cols_B = (tb == 'N') ? n : k;

            void *A     = gen_random_array(rows_A * cols_A, typesize, from_mpfr, prec, &seed_A);
            void *B     = gen_random_array(rows_B * cols_B, typesize, from_mpfr, prec, &seed_B);
            void *C_in  = gen_random_array(m * n,           typesize, from_mpfr, prec, &seed_C);
            void *alpha = gen_random_array(1,               typesize, from_mpfr, prec, &seed_ab);
            void *beta  = gen_random_array(1,               typesize, from_mpfr, prec, &seed_ab);

            void *C_out = std::malloc(static_cast<std::size_t>(m) * n * typesize);
            if (!C_out) { std::perror("malloc"); return EXIT_FAILURE; }
            std::memcpy(C_out, C_in, static_cast<std::size_t>(m) * n * typesize);

            int lda = rows_A, ldb = rows_B, ldc = m;
            gemm_fn(&ta, &tb, &m, &n, &k,
                    alpha, A, &lda, B, &ldb, beta, C_out, &ldc,
                    1, 1);

            ErrorResult err = reference_test_gemm(ctx, ta, tb, m, n, k,
                                                   alpha, A, B, beta, C_in, C_out);
            std::printf("[GEMM transa=%c transb=%c] max_rel=%.6e  normwise=%.6e\n",
                        ta, tb, err.max_relative, err.normwise_relative);

            std::free(A); std::free(B); std::free(C_in);
            std::free(C_out); std::free(alpha); std::free(beta);
        }
    }

    dlclose(conv_lib);
    dlclose(custom_lib);
    close_libs(preload_handles);
    return EXIT_SUCCESS;
}
