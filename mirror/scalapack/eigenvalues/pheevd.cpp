/* pheevd.cpp -- Mirror tester for ScaLAPACK PHEEVD (distributed Hermitian eigenvalues, D&C) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*pheevd_fn_t)(
    const char *jobz, const char *uplo, const int *n,
    void *A, const int *ia, const int *ja, const int *desca,
    void *W,
    void *Z, const int *iz, const int *jz, const int *descz,
    void *work, const int *lwork, void *rwork, const int *lrwork,
    int *iwork, const int *liwork,
    int *info,
    std::size_t jobz_len, std::size_t uplo_len
);

void mirror_test_pheevd(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        gen_mpfr_hermitian_matrix(A_mpfr, uplo, prec, &seed_A);

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &W_out) {
            MirrorPblasCtx mpc;
            if (!mpc.init(s, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             s.label.c_str());
                return;
            }

            std::size_t real_ts = s.ctx.typesize / 2;

            int loc_r = mpc.local_rows(n);
            int lld_a = std::max(1, loc_r);

            void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, n, n,
                                                          lld_a, mpc, s.ctx);

            void *W = std::calloc(n, real_ts);
            int loc_c = mpc.local_cols(n);
            int lld_z = std::max(1, loc_r);
            void *Z_loc = std::calloc(static_cast<std::size_t>(lld_z) * std::max(1, loc_c), s.ctx.typesize);

            int desc_a[9], desc_z[9];
            mpc.make_desc(desc_a, n, n, lld_a);
            mpc.make_desc(desc_z, n, n, lld_a);

            int one = 1;
            int info = 0;

            auto *fn = reinterpret_cast<pheevd_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Triple workspace query: lwork=-1, lrwork=-1, liwork=-1 */
            char work_buf[256];
            char rwork_buf[256];
            int iwork_buf;
            int lwork_query = -1;
            int lrwork_query = -1;
            int liwork_query = -1;
            fn(&"V"[0], &uplo, &n,
               A_loc, &one, &one, desc_a,
               W,
               Z_loc, &one, &one, desc_z,
               work_buf, &lwork_query, rwork_buf, &lrwork_query,
               &iwork_buf, &liwork_query,
               &info,
               (std::size_t)1, (std::size_t)1);
            int lwork = mirror_query_lwork_complex(work_buf, s.ctx);
            MpfrScalar rwork_tmp(prec);
            s.ctx.to_mpfr(rwork_tmp.get(), rwork_buf);
            int lrwork = static_cast<int>(mpfr_get_d(rwork_tmp.get(), MPFR_RNDN));
            int liwork = iwork_buf;

            void *work = std::calloc(lwork, s.ctx.typesize);
            void *rwork = std::calloc(lrwork, real_ts);
            int *iwork = static_cast<int *>(std::calloc(liwork, sizeof(int)));

            /* Re-scatter A */
            std::free(A_loc);
            A_loc = scatter_mpfr_complex_to_local(A_mpfr, n, n,
                                                    lld_a, mpc, s.ctx);

            info = 0;
            fn(&"V"[0], &uplo, &n,
               A_loc, &one, &one, desc_a,
               W,
               Z_loc, &one, &one, desc_z,
               work, &lwork, rwork, &lrwork,
               iwork, &liwork,
               &info,
               (std::size_t)1, (std::size_t)1);

            native_real_array_to_mpfr(W_out, W, n, s.ctx);

            std::free(A_loc);
            std::free(Z_loc);
            std::free(W);
            std::free(work);
            std::free(rwork);
            std::free(iwork);
            mpc.finalize();
        };

        MpfrMatrix W_a(n, 1, prec);
        MpfrMatrix W_b(n, 1, prec);
        run_side(a, W_a);
        run_side(b, W_b);

        const MpfrMatrix &ref = (config.reference == "a") ? W_a : W_b;
        const MpfrMatrix &tst = (config.reference == "a") ? W_b : W_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        mirror_report_result("PHEEVD", params_str, err,
                              nullptr, nullptr, config);
    }
}
