/* pher2.cpp -- Mirror tester for PBLAS Level 2 PHER2 (complex hermitian rank-2) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pher2_fn_t)(
    const char *uplo,
    const int *n,
    const void *alpha,
    const void *x, const int *ix, const int *jx, const int *descx,
    const int *incx,
    const void *y, const int *iy, const int *jy, const int *descy,
    const int *incy,
    void *A, const int *ia, const int *ja, const int *desca,
    std::size_t);

void mirror_test_pher2(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A  = params.seed;
        unsigned seed_x  = params.seed + 1;
        unsigned seed_y  = params.seed + 2;
        unsigned seed_al = params.seed + 3;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        MpfrComplexMatrix x_mpfr(n, 1, prec);
        MpfrComplexMatrix y_mpfr(n, 1, prec);
        MpfrComplexScalar alpha_mpfr(prec);  /* alpha is COMPLEX for PHER2 */

        gen_mpfr_hermitian_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_complex_vector(x_mpfr, prec, &seed_x);
        gen_mpfr_random_complex_vector(y_mpfr, prec, &seed_y);
        gen_mpfr_random_complex_scalar(alpha_mpfr, prec, &seed_al);

        auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
            MirrorPblasCtx mpc;
            if (!mpc.init(s, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             s.label.c_str());
                return;
            }

            int loc_rA = mpc.local_rows(n);
            int loc_rx = mpc.local_rows(n);
            int loc_ry = mpc.local_rows(n);
            int lld_a = std::max(1, loc_rA);
            int lld_x = std::max(1, loc_rx);
            int lld_y = std::max(1, loc_ry);

            void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, n, n,
                                                         lld_a, mpc, s.ctx);
            void *x_loc = scatter_mpfr_complex_to_local(x_mpfr, n, 1,
                                                         lld_x, mpc, s.ctx);
            void *y_loc = scatter_mpfr_complex_to_local(y_mpfr, n, 1,
                                                         lld_y, mpc, s.ctx);
            void *alpha_n = mpfr_complex_scalar_to_native(alpha_mpfr, s.ctx);

            int desc_a[9], desc_x[9], desc_y[9];
            mpc.make_desc(desc_a, n, n, lld_a);
            mpc.make_desc(desc_x, n, 1, lld_x);
            mpc.make_desc(desc_y, n, 1, lld_y);

            int one = 1;
            auto *fn = reinterpret_cast<pher2_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&uplo, &n,
               alpha_n,
               x_loc, &one, &one, desc_x, &one,
               y_loc, &one, &one, desc_y, &one,
               A_loc, &one, &one, desc_a,
               (std::size_t)1);

            gather_local_complex_to_mpfr(result, A_loc, lld_a, n, n,
                                          mpc, s.ctx);

            std::free(A_loc);
            std::free(x_loc);
            std::free(y_loc);
            std::free(alpha_n);
            mpc.finalize();
        };

        MpfrComplexMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrComplexMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c n=%d", uplo, n);
        mirror_report_result("PHER2", ps, err, nullptr, nullptr, config);
    }
}
