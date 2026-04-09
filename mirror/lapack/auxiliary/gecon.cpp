/* gecon.cpp -- Mirror tester for LAPACK GECON (condition number estimation) */

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

extern "C" typedef void (*getrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info);

extern "C" typedef void (*gecon_fn_t)(
    const char *norm, const int *n, const void *A, const int *lda,
    const void *anorm, void *rcond, void *work, int *iwork, int *info,
    std::size_t);

void mirror_test_gecon(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrMatrix A_mpfr(n, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    /* Compute ||A||_1 in MPFR (max column sum of absolute values) */
    MpfrScalar anorm_mpfr(prec);
    {
        MpfrScalar col_sum(prec), absval(prec);
        mpfr_set_d(anorm_mpfr.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            mpfr_set_d(col_sum.get(), 0.0, MPFR_RNDN);
            for (int i = 0; i < n; ++i) {
                mpfr_abs(absval.get(), A_mpfr.at(i, j), MPFR_RNDN);
                mpfr_add(col_sum.get(), col_sum.get(), absval.get(), MPFR_RNDN);
            }
            if (mpfr_cmp(col_sum.get(), anorm_mpfr.get()) > 0)
                mpfr_set(anorm_mpfr.get(), col_sum.get(), MPFR_RNDN);
        }
    }

    char norm_ch = '1';

    auto run_side = [&](const MirrorSide &side, MpfrScalar &rcond_out) {
        std::string getrf_sym = replace_suffix(side.sym, "gecon", "getrf");
        auto *getrf = reinterpret_cast<getrf_fn_t>(
            load_sym(side.lib, getrf_sym.c_str()));
        auto *gecon = reinterpret_cast<gecon_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
        int *ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));
        int info = 0;

        /* LU factorize */
        getrf(&n, &n, native_A, &lda, ipiv, &info);

        /* Materialize anorm to native */
        void *anorm_native = mpfr_scalar_to_native(anorm_mpfr, side.ctx);

        /* Allocate work and iwork */
        void *work = std::calloc(static_cast<std::size_t>(4 * n),
                                 side.ctx.typesize);
        int *iwork = static_cast<int *>(
            std::calloc(static_cast<std::size_t>(n), sizeof(int)));
        void *rcond_native = std::calloc(1, side.ctx.typesize);

        info = 0;
        gecon(&norm_ch, &n, native_A, &lda, anorm_native, rcond_native,
              work, iwork, &info, (std::size_t)1);

        side.ctx.to_mpfr(rcond_out.get(),
                          static_cast<const char *>(rcond_native));

        std::free(native_A);
        std::free(ipiv);
        std::free(anorm_native);
        std::free(work);
        std::free(iwork);
        std::free(rcond_native);
    };

    MpfrScalar rcond_a(prec);
    MpfrScalar rcond_b(prec);
    run_side(a, rcond_a);
    run_side(b, rcond_b);

    const MpfrScalar &ref = (config.reference == "a") ? rcond_a : rcond_b;
    const MpfrScalar &tst = (config.reference == "a") ? rcond_b : rcond_a;
    ErrorResult err = compute_error_mpfr_scalar(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "norm=1 n=%d", n);
    mirror_report_result("GECON", params_str, err,
                          nullptr, nullptr, config);
}
