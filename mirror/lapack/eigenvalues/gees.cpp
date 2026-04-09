/* gees.cpp -- Mirror tester for LAPACK GEES (Schur decomposition) */

#include "../mirror_lapack_common.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

extern "C" typedef void (*gees_fn_t)(
    const char *jobvs, const char *sort, void *select_fn,
    const int *n, void *A, const int *lda, int *sdim,
    void *wr, void *wi,
    void *VS, const int *ldvs,
    void *work, const int *lwork, int *bwork, int *info,
    std::size_t jobvs_len, std::size_t sort_len
);

static void sort_eigenvalues_by_magnitude(MpfrMatrix &wr, MpfrMatrix &wi, int n)
{
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::vector<double> mag(n);
    for (int i = 0; i < n; ++i) {
        double re = mpfr_get_d(wr.at(i, 0), MPFR_RNDN);
        double im = mpfr_get_d(wi.at(i, 0), MPFR_RNDN);
        mag[i] = std::sqrt(re * re + im * im);
    }

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return mag[a] < mag[b]; });

    mpfr_prec_t prec = mpfr_get_prec(wr.at(0, 0));
    MpfrMatrix wr_tmp(n, 1, prec);
    MpfrMatrix wi_tmp(n, 1, prec);
    for (int i = 0; i < n; ++i) {
        mpfr_set(wr_tmp.at(i, 0), wr.at(idx[i], 0), MPFR_RNDN);
        mpfr_set(wi_tmp.at(i, 0), wi.at(idx[i], 0), MPFR_RNDN);
    }
    for (int i = 0; i < n; ++i) {
        mpfr_set(wr.at(i, 0), wr_tmp.at(i, 0), MPFR_RNDN);
        mpfr_set(wi.at(i, 0), wi_tmp.at(i, 0), MPFR_RNDN);
    }
}

void mirror_test_gees(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrMatrix A_mpfr(n, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &wr_out, MpfrMatrix &wi_out) {
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
        void *wr = std::calloc(n, s.ctx.typesize);
        void *wi = std::calloc(n, s.ctx.typesize);
        int sdim = 0;
        char vs_dummy;
        int ldvs = 1;
        int *bwork = static_cast<int *>(std::calloc(n, sizeof(int)));
        int info = 0;

        auto *fn = reinterpret_cast<gees_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        /* Workspace query */
        char work_buf[256];
        int lwork_query = -1;
        fn(&"N"[0], &"N"[0], nullptr,
           &n, native_A, &lda, &sdim,
           wr, wi, &vs_dummy, &ldvs,
           work_buf, &lwork_query, bwork, &info,
           (std::size_t)1, (std::size_t)1);
        int lwork = mirror_query_lwork(work_buf, s.ctx);
        void *work = std::calloc(lwork, s.ctx.typesize);

        /* Re-materialize A */
        std::free(native_A);
        native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);

        info = 0;
        sdim = 0;
        fn(&"N"[0], &"N"[0], nullptr,
           &n, native_A, &lda, &sdim,
           wr, wi, &vs_dummy, &ldvs,
           work, &lwork, bwork, &info,
           (std::size_t)1, (std::size_t)1);

        native_array_to_mpfr(wr_out, wr, n, s.ctx);
        native_array_to_mpfr(wi_out, wi, n, s.ctx);

        std::free(native_A);
        std::free(wr);
        std::free(wi);
        std::free(bwork);
        std::free(work);
    };

    MpfrMatrix wr_a(n, 1, prec), wi_a(n, 1, prec);
    MpfrMatrix wr_b(n, 1, prec), wi_b(n, 1, prec);
    run_side(a, wr_a, wi_a);
    run_side(b, wr_b, wi_b);

    sort_eigenvalues_by_magnitude(wr_a, wi_a, n);
    sort_eigenvalues_by_magnitude(wr_b, wi_b, n);

    const MpfrMatrix &wr_ref = (config.reference == "a") ? wr_a : wr_b;
    const MpfrMatrix &wr_tst = (config.reference == "a") ? wr_b : wr_a;
    const MpfrMatrix &wi_ref = (config.reference == "a") ? wi_a : wi_b;
    const MpfrMatrix &wi_tst = (config.reference == "a") ? wi_b : wi_a;

    ErrorResult err_wr = compute_error_mpfr_vector(wr_ref, wr_tst, prec);
    ErrorResult err_wi = compute_error_mpfr_vector(wi_ref, wi_tst, prec);

    ErrorResult err;
    err.max_relative = std::fmax(err_wr.max_relative, err_wi.max_relative);
    err.normwise_relative = std::fmax(err_wr.normwise_relative, err_wi.normwise_relative);
    err.max_absolute_at_zero = std::fmax(err_wr.max_absolute_at_zero, err_wi.max_absolute_at_zero);
    err.nan_inf_mismatches = err_wr.nan_inf_mismatches + err_wi.nan_inf_mismatches;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "n=%d", n);
    mirror_report_result("GEES", params_str, err,
                          nullptr, nullptr, config);
}
