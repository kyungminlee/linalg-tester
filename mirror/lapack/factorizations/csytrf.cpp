/* csytrf.cpp -- Mirror tester for LAPACK complex SYTRF (complex symmetric indefinite factorization) */

#include "../mirror_lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*csytrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *ipiv, void *work, const int *lwork, int *info,
    std::size_t uplo_len);

void mirror_test_csytrf(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        gen_mpfr_diag_dominant_complex_symmetric(A_mpfr, uplo, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side,
                            MpfrComplexMatrix &result, int *ipiv_out) {
            void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
            std::size_t ts = side.ctx.typesize;
            int info = 0;

            auto *fn = reinterpret_cast<csytrf_fn_t>(
                load_sym(side.lib, side.sym.c_str()));

            /* Workspace query */
            int lwork_query = -1;
            void *work_q = std::malloc(ts);
            fn(&uplo, &n, native_A, &lda, ipiv_out, work_q, &lwork_query,
               &info, (std::size_t)1);
            int lwork = mirror_query_lwork_complex(work_q, side.ctx);
            std::free(work_q);

            /* Re-materialize A */
            std::free(native_A);
            native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);

            void *work = std::malloc(static_cast<std::size_t>(lwork) * ts);
            info = 0;
            fn(&uplo, &n, native_A, &lda, ipiv_out, work, &lwork,
               &info, (std::size_t)1);

            custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);
            std::free(native_A);
            std::free(work);
        };

        MpfrComplexMatrix res_a(n, n, prec);
        MpfrComplexMatrix res_b(n, n, prec);
        int *ipiv_a = new int[n];
        int *ipiv_b = new int[n];
        run_side(a, res_a, ipiv_a);
        run_side(b, res_b, ipiv_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        bool ipiv_match = true;
        for (int i = 0; i < n; ++i) {
            if (ipiv_a[i] != ipiv_b[i]) { ipiv_match = false; break; }
        }

        char params_str[256];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c ipiv_match=%s", uplo,
                      ipiv_match ? "yes" : "NO");
        mirror_report_result("CSYTRF", params_str, err,
                              nullptr, nullptr, config);

        delete[] ipiv_a;
        delete[] ipiv_b;
    }
}
