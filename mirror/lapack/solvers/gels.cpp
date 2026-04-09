/* gels.cpp -- Mirror tester for LAPACK GELS (least squares solve) */

#include "../mirror_lapack_common.h"

#include <algorithm>

extern "C" typedef void (*gels_fn_t)(
    const char *trans, const int *m, const int *n, const int *nrhs,
    void *A, const int *lda,
    void *B, const int *ldb,
    void *work, const int *lwork, int *info,
    std::size_t trans_len
);

void mirror_test_gels(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int nrhs = (std::min(m, n) < 4) ? std::min(m, n) : 4;
    mpfr_prec_t prec = config.prec;

    int lda = m + params.ld_pad;
    int mn_max = std::max(m, n);
    int ldb = mn_max + params.ld_pad;

    char trans = 'N';

    unsigned seed_A = params.seed;
    unsigned seed_B = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrMatrix A_mpfr(m, n, prec);
    MpfrMatrix B_mpfr(mn_max, nrhs, prec);

    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);

    /* Run one side */
    auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
        auto *fn = reinterpret_cast<gels_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        /* Workspace query */
        int lwork = -1;
        int info = 0;
        std::vector<char> work_query(s.ctx.typesize);
        {
            void *tmp_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            void *tmp_B = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);

            fn(&trans, &m, &n, &nrhs, tmp_A, &lda, tmp_B, &ldb,
               work_query.data(), &lwork, &info, (std::size_t)1);

            lwork = mirror_query_lwork(work_query.data(), s.ctx);

            std::free(tmp_A);
            std::free(tmp_B);
        }

        /* Re-materialize and solve */
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
        void *native_B = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);
        std::vector<char> work(static_cast<std::size_t>(lwork) * s.ctx.typesize);
        info = 0;

        fn(&trans, &m, &n, &nrhs, native_A, &lda, native_B, &ldb,
           work.data(), &lwork, &info, (std::size_t)1);

        custom_to_mpfr_mat(result, native_B, ldb, s.ctx);

        std::free(native_A);
        std::free(native_B);
    };

    MpfrMatrix res_a(mn_max, nrhs, prec);
    MpfrMatrix res_b(mn_max, nrhs, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "trans=%c m=%d n=%d nrhs=%d", trans, m, n, nrhs);
    mirror_report_result("GELS", params_str, err,
                          nullptr, nullptr, config);
}
