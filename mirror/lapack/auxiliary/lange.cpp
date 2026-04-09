/* lange.cpp -- Mirror tester for LAPACK LANGE (matrix norm) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void mirror_test_lange(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = m + params.ld_pad;

    for (char norm_type : {'M', '1', 'I', 'F'}) {
        unsigned seed_A = params.seed;

        MpfrMatrix A_mpfr(m, n, prec);
        gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side, MpfrScalar &result) {
            void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            void *work = std::calloc(static_cast<std::size_t>(std::max(1, m)),
                                     side.ctx.typesize);

            std::size_t real_ts = side.ctx.typesize;

            if (real_ts == 4) {
                auto fn = reinterpret_cast<float (*)(
                    const char *, const int *, const int *,
                    const void *, const int *, void *,
                    std::size_t)>(load_sym(side.lib, side.sym.c_str()));
                float r = fn(&norm_type, &m, &n, native_A, &lda, work,
                             (std::size_t)1);
                char buf[sizeof(float)];
                std::memcpy(buf, &r, sizeof(float));
                side.ctx.to_mpfr(result.get(), buf);
            } else if (real_ts == 8) {
                auto fn = reinterpret_cast<double (*)(
                    const char *, const int *, const int *,
                    const void *, const int *, void *,
                    std::size_t)>(load_sym(side.lib, side.sym.c_str()));
                double r = fn(&norm_type, &m, &n, native_A, &lda, work,
                              (std::size_t)1);
                char buf[sizeof(double)];
                std::memcpy(buf, &r, sizeof(double));
                side.ctx.to_mpfr(result.get(), buf);
            } else {
                auto fn = reinterpret_cast<long double (*)(
                    const char *, const int *, const int *,
                    const void *, const int *, void *,
                    std::size_t)>(load_sym(side.lib, side.sym.c_str()));
                long double r = fn(&norm_type, &m, &n, native_A, &lda, work,
                                   (std::size_t)1);
                char buf[sizeof(long double)];
                std::memcpy(buf, &r, sizeof(long double));
                side.ctx.to_mpfr(result.get(), buf);
            }

            std::free(native_A);
            std::free(work);
        };

        MpfrScalar res_a(prec);
        MpfrScalar res_b(prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrScalar &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrScalar &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_scalar(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "norm=%c m=%d n=%d", norm_type, m, n);
        mirror_report_result("LANGE", params_str, err,
                              nullptr, nullptr, config);
    }
}
