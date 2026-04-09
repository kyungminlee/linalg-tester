/* trmv.cpp -- Mirror tester for BLAS Level 2 TRMV */

#include "../mirror_blas2.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*trmv_fn_t)(
    const char *uplo, const char *trans, const char *diag,
    const int  *n,
    const void *A,      const int  *lda,
    void       *x,      const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

void mirror_test_trmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        for (char trans : {'N', 'T', 'C'}) {
            for (char diag : {'N', 'U'}) {
                int lda = n + params.ld_pad;

                unsigned seed_A = params.seed;
                unsigned seed_x = params.seed + 1;

                MpfrMatrix A_mpfr(n, n, prec);
                MpfrMatrix x_mpfr(n, 1, prec);

                gen_mpfr_triangular_matrix(A_mpfr, uplo, diag, prec, &seed_A);
                gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

                auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
                    void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
                    void *native_x = mpfr_vec_to_native(x_mpfr, incx, side.ctx);

                    auto *fn = reinterpret_cast<trmv_fn_t>(
                        load_sym(side.lib, side.sym.c_str()));
                    fn(&uplo, &trans, &diag, &n,
                       native_A, &lda, native_x, &incx,
                       (std::size_t)1, (std::size_t)1, (std::size_t)1);

                    custom_to_mpfr_vec(result, native_x, incx, side.ctx);

                    std::free(native_A);
                    std::free(native_x);
                };

                MpfrMatrix res_a(n, 1, prec); run_side(a, res_a);
                MpfrMatrix res_b(n, 1, prec); run_side(b, res_b);

                const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
                const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
                ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

                char params_str[128];
                std::snprintf(params_str, sizeof(params_str),
                              "uplo=%c trans=%c diag=%c n=%d incx=%d",
                              uplo, trans, diag, n, incx);
                mirror_report_result("TRMV", params_str, err,
                                      nullptr, nullptr, config);
            }
        }
    }
}
