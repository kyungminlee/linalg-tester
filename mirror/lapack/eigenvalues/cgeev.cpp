/* cgeev.cpp -- Mirror tester for LAPACK CGEEV (complex general eigenvalues) */

#include "../mirror_lapack_common.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

extern "C" typedef void (*cgeev_fn_t)(
    const char *jobvl, const char *jobvr, const int *n,
    void *A, const int *lda, void *W,
    void *VL, const int *ldvl, void *VR, const int *ldvr,
    void *work, const int *lwork, void *rwork, int *info,
    std::size_t jobvl_len, std::size_t jobvr_len
);

/* Sort complex eigenvalues by magnitude */
static void sort_complex_eigenvalues(MpfrComplexMatrix &W, int n)
{
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::vector<double> mag(n);
    for (int i = 0; i < n; ++i) {
        double re = mpfr_get_d(W.re(i, 0), MPFR_RNDN);
        double im = mpfr_get_d(W.im(i, 0), MPFR_RNDN);
        mag[i] = std::sqrt(re * re + im * im);
    }

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return mag[a] < mag[b]; });

    mpfr_prec_t prec = mpfr_get_prec(W.re(0, 0));
    MpfrComplexMatrix tmp(n, 1, prec);
    for (int i = 0; i < n; ++i) {
        mpfr_set(tmp.re(i, 0), W.re(idx[i], 0), MPFR_RNDN);
        mpfr_set(tmp.im(i, 0), W.im(idx[i], 0), MPFR_RNDN);
    }
    for (int i = 0; i < n; ++i) {
        mpfr_set(W.re(i, 0), tmp.re(i, 0), MPFR_RNDN);
        mpfr_set(W.im(i, 0), tmp.im(i, 0), MPFR_RNDN);
    }
}

void mirror_test_cgeev(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(n, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &W_out) {
        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
        void *W = std::calloc(n, s.ctx.typesize);
        std::size_t real_ts = s.ctx.typesize / 2;
        char vl_dummy, vr_dummy;
        int ldvl = 1, ldvr = 1;
        void *rwork = std::calloc(2 * n, real_ts);
        int info = 0;

        auto *fn = reinterpret_cast<cgeev_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        /* Workspace query */
        char work_buf[256];
        int lwork_query = -1;
        fn(&"N"[0], &"N"[0], &n, native_A, &lda, W,
           &vl_dummy, &ldvl, &vr_dummy, &ldvr,
           work_buf, &lwork_query, rwork, &info,
           (std::size_t)1, (std::size_t)1);
        int lwork = mirror_query_lwork_complex(work_buf, s.ctx);
        void *work = std::calloc(lwork, s.ctx.typesize);

        /* Re-materialize A */
        std::free(native_A);
        native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);

        info = 0;
        fn(&"N"[0], &"N"[0], &n, native_A, &lda, W,
           &vl_dummy, &ldvl, &vr_dummy, &ldvr,
           work, &lwork, rwork, &info,
           (std::size_t)1, (std::size_t)1);

        /* W is complex (full typesize per element) */
        native_complex_array_to_mpfr(W_out, W, n, s.ctx);

        std::free(native_A);
        std::free(W);
        std::free(rwork);
        std::free(work);
    };

    MpfrComplexMatrix W_a(n, 1, prec);
    MpfrComplexMatrix W_b(n, 1, prec);
    run_side(a, W_a);
    run_side(b, W_b);

    /* Sort eigenvalues by magnitude before comparing */
    sort_complex_eigenvalues(W_a, n);
    sort_complex_eigenvalues(W_b, n);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? W_a : W_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? W_b : W_a;
    ErrorResult err = compute_error_mpfr_complex_vector(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "n=%d", n);
    mirror_report_result("CGEEV", params_str, err,
                          nullptr, nullptr, config);
}
