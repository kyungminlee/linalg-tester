/* potrs.cpp -- Mirror tester for LAPACK POTRS (Cholesky-based solve) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string replace_suffix(const std::string &sym, const char *from, const char *to) {
    std::string s = sym;
    auto pos = s.find(from);
    if (pos != std::string::npos)
        s.replace(pos, std::strlen(from), to);
    return s;
}

extern "C" typedef void (*potrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t);

extern "C" typedef void (*potrs_fn_t)(
    const char *uplo, const int *n, const int *nrhs,
    const void *A, const int *lda,
    void *B, const int *ldb, int *info, std::size_t);

void mirror_test_potrs(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int nrhs = std::min(n, 4);
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix B_mpfr(n, nrhs, prec);
        gen_mpfr_positive_definite_matrix(A_mpfr, prec, &seed_A);
        gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
            std::string potrf_sym = replace_suffix(side.sym, "potrs", "potrf");
            auto *potrf = reinterpret_cast<potrf_fn_t>(
                load_sym(side.lib, potrf_sym.c_str()));
            auto *potrs = reinterpret_cast<potrs_fn_t>(
                load_sym(side.lib, side.sym.c_str()));

            void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            void *native_B = mpfr_mat_to_native(B_mpfr, ldb, side.ctx);
            int info = 0;

            potrf(&uplo, &n, native_A, &lda, &info, (std::size_t)1);
            info = 0;
            potrs(&uplo, &n, &nrhs, native_A, &lda, native_B, &ldb,
                  &info, (std::size_t)1);

            custom_to_mpfr_mat(result, native_B, ldb, side.ctx);

            std::free(native_A);
            std::free(native_B);
        };

        MpfrMatrix res_a(n, nrhs, prec);
        MpfrMatrix res_b(n, nrhs, prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d nrhs=%d", uplo, n, nrhs);
        mirror_report_result("POTRS", params_str, err,
                              nullptr, nullptr, config);
    }
}
