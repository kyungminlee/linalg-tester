/* tbsv.cpp -- Mirror tester for BLAS Level 2 TBSV */

#include "../mirror_blas2.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*tbsv_fn_t)(
    const char *uplo, const char *trans, const char *diag,
    const int  *n,    const int  *k,
    const void *AB,   const int  *ldab,
    void       *x,    const int  *incx,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len
);

void mirror_test_tbsv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int k = std::min(params.kl, n - 1);
    int incx = params.incx;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        for (char trans : {'N', 'T', 'C'}) {
            for (char diag : {'N', 'U'}) {
                int ldab = k + 1;

                unsigned seed_AB = params.seed;
                unsigned seed_x  = params.seed + 1;

                MpfrMatrix AB_mpfr(ldab, n, prec);
                MpfrMatrix x_mpfr(n, 1, prec);

                gen_mpfr_triangular_band_matrix(AB_mpfr, n, k, uplo, diag, prec, &seed_AB);
                gen_mpfr_random_vector(x_mpfr, prec, &seed_x);

                auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
                    void *native_AB = mpfr_mat_to_native(AB_mpfr, ldab, side.ctx);
                    void *native_x  = mpfr_vec_to_native(x_mpfr, incx, side.ctx);

                    auto *fn = reinterpret_cast<tbsv_fn_t>(
                        load_sym(side.lib, side.sym.c_str()));
                    fn(&uplo, &trans, &diag, &n, &k,
                       native_AB, &ldab,
                       native_x, &incx,
                       (std::size_t)1, (std::size_t)1, (std::size_t)1);

                    custom_to_mpfr_vec(result, native_x, incx, side.ctx);

                    std::free(native_AB);
                    std::free(native_x);
                };

                MpfrMatrix res_a(n, 1, prec); run_side(a, res_a);
                MpfrMatrix res_b(n, 1, prec); run_side(b, res_b);

                const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
                const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
                ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

                char params_str[128];
                std::snprintf(params_str, sizeof(params_str),
                              "uplo=%c trans=%c diag=%c n=%d k=%d incx=%d",
                              uplo, trans, diag, n, k, incx);
                mirror_report_result("TBSV", params_str, err,
                                      nullptr, nullptr, config);
            }
        }
    }
}
