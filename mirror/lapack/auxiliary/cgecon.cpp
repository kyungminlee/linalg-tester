/* cgecon.cpp -- Mirror tester for LAPACK complex GECON (condition number estimation) */

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

extern "C" typedef void (*cgetrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info);

extern "C" typedef void (*cgecon_fn_t)(
    const char *norm, const int *n, const void *A, const int *lda,
    const void *anorm, void *rcond, void *work, void *rwork, int *info,
    std::size_t);

void mirror_test_cgecon(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(n, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    /* Compute ||A||_1 in MPFR (max column sum of complex absolute values) */
    MpfrScalar anorm_mpfr(prec);
    {
        MpfrScalar col_sum(prec), abs_re(prec), abs_im(prec), abs_val(prec);
        mpfr_set_d(anorm_mpfr.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            mpfr_set_d(col_sum.get(), 0.0, MPFR_RNDN);
            for (int i = 0; i < n; ++i) {
                /* |z| = sqrt(re^2 + im^2) */
                mpfr_mul(abs_re.get(), A_mpfr.re(i, j), A_mpfr.re(i, j), MPFR_RNDN);
                mpfr_mul(abs_im.get(), A_mpfr.im(i, j), A_mpfr.im(i, j), MPFR_RNDN);
                mpfr_add(abs_val.get(), abs_re.get(), abs_im.get(), MPFR_RNDN);
                mpfr_sqrt(abs_val.get(), abs_val.get(), MPFR_RNDN);
                mpfr_add(col_sum.get(), col_sum.get(), abs_val.get(), MPFR_RNDN);
            }
            if (mpfr_cmp(col_sum.get(), anorm_mpfr.get()) > 0)
                mpfr_set(anorm_mpfr.get(), col_sum.get(), MPFR_RNDN);
        }
    }

    char norm_ch = '1';

    auto run_side = [&](const MirrorSide &side, MpfrScalar &rcond_out) {
        std::string cgetrf_sym = replace_suffix(side.sym, "gecon", "getrf");
        auto *cgetrf = reinterpret_cast<cgetrf_fn_t>(
            load_sym(side.lib, cgetrf_sym.c_str()));
        auto *cgecon = reinterpret_cast<cgecon_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
        int *ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));
        int info = 0;

        /* LU factorize */
        cgetrf(&n, &n, native_A, &lda, ipiv, &info);

        /* anorm and rcond are real scalars (typesize/2) */
        std::size_t real_ts = side.ctx.typesize / 2;

        /* Materialize anorm to native real */
        void *anorm_native = std::calloc(1, real_ts);
        side.ctx.from_mpfr(anorm_native, anorm_mpfr.get(), MPFR_RNDN);

        /* rcond is real */
        void *rcond_native = std::calloc(1, real_ts);

        /* work: complex array of size 2*n */
        void *work = std::calloc(static_cast<std::size_t>(2 * n),
                                 side.ctx.typesize);
        /* rwork: real array of size 2*n */
        void *rwork = std::calloc(static_cast<std::size_t>(2 * n), real_ts);

        info = 0;
        cgecon(&norm_ch, &n, native_A, &lda, anorm_native, rcond_native,
               work, rwork, &info, (std::size_t)1);

        side.ctx.to_mpfr(rcond_out.get(),
                          static_cast<const char *>(rcond_native));

        std::free(native_A);
        std::free(ipiv);
        std::free(anorm_native);
        std::free(rcond_native);
        std::free(work);
        std::free(rwork);
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
    mirror_report_result("CGECON", params_str, err,
                          nullptr, nullptr, config);
}
