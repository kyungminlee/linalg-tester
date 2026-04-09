/* cgesvd.cpp -- Mirror tester for LAPACK CGESVD (complex SVD, QR) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*cgesvd_fn_t)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n, void *A, const int *lda,
    void *S,
    void *U, const int *ldu, void *VT, const int *ldvt,
    void *work, const int *lwork, void *rwork, int *info,
    std::size_t jobu_len, std::size_t jobvt_len
);

void mirror_test_cgesvd(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int m = params.m;
    int n = params.n;
    int minmn = (m < n) ? m : n;
    mpfr_prec_t prec = config.prec;
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &S_out) {
        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
        std::size_t real_ts = s.ctx.typesize / 2;
        void *S = std::calloc(minmn, real_ts);
        char u_dummy, vt_dummy;
        int ldu = 1, ldvt = 1;
        void *rwork = std::calloc(5 * minmn, real_ts);
        int info = 0;

        auto *fn = reinterpret_cast<cgesvd_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        /* Workspace query */
        char work_buf[256];
        int lwork_query = -1;
        fn(&"N"[0], &"N"[0], &m, &n, native_A, &lda, S,
           &u_dummy, &ldu, &vt_dummy, &ldvt,
           work_buf, &lwork_query, rwork, &info,
           (std::size_t)1, (std::size_t)1);
        int lwork = mirror_query_lwork_complex(work_buf, s.ctx);
        void *work = std::calloc(lwork, s.ctx.typesize);

        /* Re-materialize A */
        std::free(native_A);
        native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);

        info = 0;
        fn(&"N"[0], &"N"[0], &m, &n, native_A, &lda, S,
           &u_dummy, &ldu, &vt_dummy, &ldvt,
           work, &lwork, rwork, &info,
           (std::size_t)1, (std::size_t)1);

        /* S is real (typesize/2 per element) */
        native_real_array_to_mpfr(S_out, S, minmn, s.ctx);

        std::free(native_A);
        std::free(S);
        std::free(rwork);
        std::free(work);
    };

    MpfrMatrix S_a(minmn, 1, prec);
    MpfrMatrix S_b(minmn, 1, prec);
    run_side(a, S_a);
    run_side(b, S_b);

    const MpfrMatrix &ref = (config.reference == "a") ? S_a : S_b;
    const MpfrMatrix &tst = (config.reference == "a") ? S_b : S_a;
    ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    mirror_report_result("CGESVD", params_str, err,
                          nullptr, nullptr, config);
}
