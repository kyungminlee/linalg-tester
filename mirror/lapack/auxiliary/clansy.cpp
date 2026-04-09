/* clansy.cpp -- Mirror tester for LAPACK complex LANSY (complex symmetric matrix norm) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void mirror_test_clansy(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        for (char norm_type : {'M', '1', 'I', 'F'}) {
            unsigned seed_A = params.seed;

            MpfrComplexMatrix A_mpfr(n, n, prec);
            gen_mpfr_complex_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);

            auto run_side = [&](const MirrorSide &side, MpfrScalar &result) {
                void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda,
                                                             side.ctx);
                /* work: real array of size max(1,n) for 'I' norm */
                std::size_t real_ts = side.ctx.typesize / 2;
                void *work = std::calloc(
                    static_cast<std::size_t>(std::max(1, n)), real_ts);

                /* Return type is real: dispatch by real_ts */
                if (real_ts == 4) {
                    auto fn = reinterpret_cast<float (*)(
                        const char *, const char *, const int *,
                        const void *, const int *, void *,
                        std::size_t, std::size_t)>(
                        load_sym(side.lib, side.sym.c_str()));
                    float r = fn(&norm_type, &uplo, &n, native_A, &lda, work,
                                 (std::size_t)1, (std::size_t)1);
                    char buf[sizeof(float)];
                    std::memcpy(buf, &r, sizeof(float));
                    side.ctx.to_mpfr(result.get(), buf);
                } else if (real_ts == 8) {
                    auto fn = reinterpret_cast<double (*)(
                        const char *, const char *, const int *,
                        const void *, const int *, void *,
                        std::size_t, std::size_t)>(
                        load_sym(side.lib, side.sym.c_str()));
                    double r = fn(&norm_type, &uplo, &n, native_A, &lda, work,
                                  (std::size_t)1, (std::size_t)1);
                    char buf[sizeof(double)];
                    std::memcpy(buf, &r, sizeof(double));
                    side.ctx.to_mpfr(result.get(), buf);
                } else {
                    auto fn = reinterpret_cast<long double (*)(
                        const char *, const char *, const int *,
                        const void *, const int *, void *,
                        std::size_t, std::size_t)>(
                        load_sym(side.lib, side.sym.c_str()));
                    long double r = fn(&norm_type, &uplo, &n, native_A, &lda,
                                       work, (std::size_t)1, (std::size_t)1);
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
                          "norm=%c uplo=%c n=%d", norm_type, uplo, n);
            mirror_report_result("CLANSY", params_str, err,
                                  nullptr, nullptr, config);
        }
    }
}
