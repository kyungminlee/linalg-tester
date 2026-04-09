/* cpotri.cpp -- Mirror tester for LAPACK complex POTRI (Cholesky-based inverse) */

#include "../mirror_lapack_common.h"

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

extern "C" typedef void (*cpotrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t);

extern "C" typedef void (*cpotri_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t);

void mirror_test_cpotri(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        gen_mpfr_hermitian_positive_definite(A_mpfr, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
            std::string cpotrf_sym = replace_suffix(side.sym, "potri", "potrf");
            auto *cpotrf = reinterpret_cast<cpotrf_fn_t>(
                load_sym(side.lib, cpotrf_sym.c_str()));
            auto *cpotri = reinterpret_cast<cpotri_fn_t>(
                load_sym(side.lib, side.sym.c_str()));

            void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
            int info = 0;

            cpotrf(&uplo, &n, native_A, &lda, &info, (std::size_t)1);
            info = 0;
            cpotri(&uplo, &n, native_A, &lda, &info, (std::size_t)1);

            custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);

            std::free(native_A);
        };

        MpfrComplexMatrix res_a(n, n, prec);
        MpfrComplexMatrix res_b(n, n, prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        mirror_report_result("CPOTRI", params_str, err,
                              nullptr, nullptr, config);
    }
}
