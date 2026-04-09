/* gelsd.cpp -- Mirror tester for LAPACK GELSD (least squares via SVD) */

#include "../mirror_lapack_common.h"

#include <algorithm>

extern "C" typedef void (*gelsd_fn_t)(
    const int *m, const int *n, const int *nrhs,
    void *A, const int *lda,
    void *B, const int *ldb,
    void *S, const void *rcond, int *rank,
    void *work, const int *lwork, int *iwork, int *info
);

void mirror_test_gelsd(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int nrhs = (std::min(m, n) < 4) ? std::min(m, n) : 4;
    int mn_min = std::min(m, n);
    mpfr_prec_t prec = config.prec;

    int lda = m + params.ld_pad;
    int mn_max = std::max(m, n);
    int ldb = mn_max + params.ld_pad;

    unsigned seed_A = params.seed;
    unsigned seed_B = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrMatrix A_mpfr(m, n, prec);
    MpfrMatrix B_mpfr(mn_max, nrhs, prec);

    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);

    /* rcond = -1.0 (use machine precision) */
    MpfrScalar rcond_mpfr(prec);
    mpfr_set_d(rcond_mpfr.get(), -1.0, MPFR_RNDN);

    /* Run one side */
    auto run_side = [&](const MirrorSide &s, MpfrMatrix &result_B,
                         MpfrMatrix &result_S) {
        auto *fn = reinterpret_cast<gelsd_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        void *native_rcond = mpfr_scalar_to_native(rcond_mpfr, s.ctx);

        /* Workspace query */
        int lwork = -1;
        int info = 0;
        int rank = 0;
        std::vector<char> work_query(s.ctx.typesize);
        int iwork_query = 0;
        {
            void *tmp_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            void *tmp_B = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);
            std::vector<char> tmp_S(static_cast<std::size_t>(mn_min) * s.ctx.typesize);

            fn(&m, &n, &nrhs, tmp_A, &lda, tmp_B, &ldb,
               tmp_S.data(), native_rcond, &rank,
               work_query.data(), &lwork, &iwork_query, &info);

            lwork = mirror_query_lwork(work_query.data(), s.ctx);

            std::free(tmp_A);
            std::free(tmp_B);
        }

        /* Re-materialize and solve */
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
        void *native_B = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);
        std::vector<char> native_S(static_cast<std::size_t>(mn_min) * s.ctx.typesize);
        std::vector<char> work(static_cast<std::size_t>(lwork) * s.ctx.typesize);
        /* iwork size: at least 1; GELSD documentation says query returns it in iwork[0] */
        int iwork_size = (iwork_query > 0) ? iwork_query : 1;
        /* Safe estimate: 3*min(m,n)*nlvl + 11*min(m,n) where nlvl ~ log2(min(m,n)/25)+1 */
        int nlvl = 1;
        { int tmp = mn_min; while (tmp > 25) { tmp /= 2; ++nlvl; } }
        int iwork_est = 3 * mn_min * nlvl + 11 * mn_min;
        if (iwork_est > iwork_size) iwork_size = iwork_est;
        std::vector<int> iwork(iwork_size, 0);
        info = 0;
        rank = 0;

        fn(&m, &n, &nrhs, native_A, &lda, native_B, &ldb,
           native_S.data(), native_rcond, &rank,
           work.data(), &lwork, iwork.data(), &info);

        custom_to_mpfr_mat(result_B, native_B, ldb, s.ctx);
        native_array_to_mpfr(result_S, native_S.data(), mn_min, s.ctx);

        std::free(native_A);
        std::free(native_B);
        std::free(native_rcond);
    };

    MpfrMatrix res_B_a(mn_max, nrhs, prec), res_S_a(mn_min, 1, prec);
    MpfrMatrix res_B_b(mn_max, nrhs, prec), res_S_b(mn_min, 1, prec);
    run_side(a, res_B_a, res_S_a);
    run_side(b, res_B_b, res_S_b);

    const MpfrMatrix &ref_B = (config.reference == "a") ? res_B_a : res_B_b;
    const MpfrMatrix &tst_B = (config.reference == "a") ? res_B_b : res_B_a;
    ErrorResult err_B = compute_error_mpfr_matrix(ref_B, tst_B, prec);

    const MpfrMatrix &ref_S = (config.reference == "a") ? res_S_a : res_S_b;
    const MpfrMatrix &tst_S = (config.reference == "a") ? res_S_b : res_S_a;
    ErrorResult err_S = compute_error_mpfr_vector(ref_S, tst_S, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d nrhs=%d", m, n, nrhs);
    mirror_report_result("GELSD(B)", params_str, err_B,
                          nullptr, nullptr, config);
    mirror_report_result("GELSD(S)", params_str, err_S,
                          nullptr, nullptr, config);
}
