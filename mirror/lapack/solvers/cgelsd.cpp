/* cgelsd.cpp -- Mirror tester for LAPACK CGELSD/ZGELSD (complex least squares via SVD) */

#include "../mirror_lapack_common.h"

#include <algorithm>

extern "C" typedef void (*cgelsd_fn_t)(
    const int *m, const int *n, const int *nrhs,
    void *A, const int *lda,
    void *B, const int *ldb,
    void *S, const void *rcond, int *rank,
    void *work, const int *lwork, void *rwork, int *iwork, int *info
);

void mirror_test_cgelsd(const MirrorSide &a, const MirrorSide &b,
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
    MpfrComplexMatrix A_mpfr(m, n, prec);
    MpfrComplexMatrix B_mpfr(mn_max, nrhs, prec);

    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);

    /* rcond = -1.0 (use machine precision); S and rcond are real, use typesize/2 */
    MpfrScalar rcond_mpfr(prec);
    mpfr_set_d(rcond_mpfr.get(), -1.0, MPFR_RNDN);

    /* Run one side */
    auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result_B,
                         MpfrMatrix &result_S) {
        auto *fn = reinterpret_cast<cgelsd_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        std::size_t real_ts = s.ctx.typesize / 2;

        /* Materialize rcond as real scalar (typesize/2) */
        void *native_rcond = static_cast<void *>(std::malloc(real_ts));
        {
            MpfrScalar tmp(s.ctx.prec);
            mpfr_set(tmp.get(), rcond_mpfr.get(), MPFR_RNDN);
            s.ctx.from_mpfr(native_rcond, tmp.get(), MPFR_RNDN);
        }

        /* Workspace query */
        int lwork = -1;
        int info = 0;
        int rank = 0;
        std::vector<char> work_query(s.ctx.typesize);
        int iwork_query = 0;
        {
            void *tmp_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
            void *tmp_B = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
            std::vector<char> tmp_S(static_cast<std::size_t>(mn_min) * real_ts);
            std::vector<char> tmp_rwork(real_ts); /* minimal for query */

            fn(&m, &n, &nrhs, tmp_A, &lda, tmp_B, &ldb,
               tmp_S.data(), native_rcond, &rank,
               work_query.data(), &lwork, tmp_rwork.data(),
               &iwork_query, &info);

            lwork = mirror_query_lwork_complex(work_query.data(), s.ctx);

            std::free(tmp_A);
            std::free(tmp_B);
        }

        /* Compute iwork and rwork sizes */
        int nlvl = 1;
        { int tmp = mn_min; while (tmp > 25) { tmp /= 2; ++nlvl; } }
        int iwork_size = 3 * mn_min * nlvl + 11 * mn_min;
        if (iwork_size < 1) iwork_size = 1;

        /* rwork size for complex GELSD:
           10*mn_min + 2*mn_min*SMLSIZ + 8*nlvl*mn_min + 3*SMLSIZ*nrhs + max((SMLSIZ+1)^2, ...)
           Use generous estimate */
        int smlsiz = 25;
        long rwork_size = 10L * mn_min + 2L * mn_min * smlsiz +
                          8L * nlvl * mn_min + 3L * smlsiz * nrhs +
                          static_cast<long>((smlsiz + 1) * (smlsiz + 1));
        if (rwork_size < 1) rwork_size = 1;

        /* Re-materialize and solve */
        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
        void *native_B = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
        std::vector<char> native_S(static_cast<std::size_t>(mn_min) * real_ts);
        std::vector<char> work(static_cast<std::size_t>(lwork) * s.ctx.typesize);
        std::vector<char> rwork(static_cast<std::size_t>(rwork_size) * real_ts);
        std::vector<int> iwork(iwork_size, 0);
        info = 0;
        rank = 0;

        fn(&m, &n, &nrhs, native_A, &lda, native_B, &ldb,
           native_S.data(), native_rcond, &rank,
           work.data(), &lwork, rwork.data(), iwork.data(), &info);

        custom_to_mpfr_complex_mat(result_B, native_B, ldb, s.ctx);
        native_real_array_to_mpfr(result_S, native_S.data(), mn_min, s.ctx);

        std::free(native_A);
        std::free(native_B);
        std::free(native_rcond);
    };

    MpfrComplexMatrix res_B_a(mn_max, nrhs, prec);
    MpfrMatrix res_S_a(mn_min, 1, prec);
    MpfrComplexMatrix res_B_b(mn_max, nrhs, prec);
    MpfrMatrix res_S_b(mn_min, 1, prec);
    run_side(a, res_B_a, res_S_a);
    run_side(b, res_B_b, res_S_b);

    const MpfrComplexMatrix &ref_B = (config.reference == "a") ? res_B_a : res_B_b;
    const MpfrComplexMatrix &tst_B = (config.reference == "a") ? res_B_b : res_B_a;
    ErrorResult err_B = compute_error_mpfr_complex_matrix(ref_B, tst_B, prec);

    const MpfrMatrix &ref_S = (config.reference == "a") ? res_S_a : res_S_b;
    const MpfrMatrix &tst_S = (config.reference == "a") ? res_S_b : res_S_a;
    ErrorResult err_S = compute_error_mpfr_vector(ref_S, tst_S, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d nrhs=%d", m, n, nrhs);
    mirror_report_result("CGELSD(B)", params_str, err_B,
                          nullptr, nullptr, config);
    mirror_report_result("CGELSD(S)", params_str, err_S,
                          nullptr, nullptr, config);
}
