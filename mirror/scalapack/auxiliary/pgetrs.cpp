/* pgetrs.cpp -- Mirror tester for ScaLAPACK PGETRS (distributed LU solve) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*pgetrf_fn_t)(
    const int *m, const int *n, void *A, const int *ia, const int *ja,
    const int *desca, int *ipiv, int *info);

extern "C" typedef void (*pgetrs_fn_t)(
    const char *trans, const int *n, const int *nrhs,
    const void *A, const int *ia, const int *ja, const int *desca,
    const int *ipiv,
    void *B, const int *ib, const int *jb, const int *descb,
    int *info, std::size_t trans_len);

static std::string replace_suffix(const std::string &sym, const char *from, const char *to) {
    std::string s = sym;
    auto pos = s.find(from);
    if (pos != std::string::npos)
        s.replace(pos, std::strlen(from), to);
    return s;
}

void mirror_test_pgetrs(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n, nrhs = params.k;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char trans : {'N', 'T'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix B_mpfr(n, nrhs, prec);
        gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
        gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
            MirrorPblasCtx mpc;
            if (!mpc.init(side, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             side.label.c_str());
                return;
            }

            int loc_rA = mpc.local_rows(n);
            int loc_rB = mpc.local_rows(n);
            int lld_a = std::max(1, loc_rA);
            int lld_b = std::max(1, loc_rB);

            void *A_loc = scatter_mpfr_to_local(A_mpfr, n, n, lld_a, mpc, side.ctx);
            void *B_loc = scatter_mpfr_to_local(B_mpfr, n, nrhs, lld_b, mpc, side.ctx);

            int desc_a[9], desc_b[9];
            mpc.make_desc(desc_a, n, n, lld_a);
            mpc.make_desc(desc_b, n, nrhs, lld_b);

            int one = 1;
            int loc_cA = mpc.local_cols(n);
            int ipiv_len = loc_rA + mb;
            int *ipiv = static_cast<int *>(std::calloc(ipiv_len, sizeof(int)));
            int info = 0;

            /* PGETRF */
            std::string rf_sym = replace_suffix(side.sym, "getrs", "getrf");
            auto *rf_fn = reinterpret_cast<pgetrf_fn_t>(
                load_sym(side.lib, rf_sym.c_str()));
            rf_fn(&n, &n, A_loc, &one, &one, desc_a, ipiv, &info);

            /* PGETRS */
            auto *fn = reinterpret_cast<pgetrs_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&trans, &n, &nrhs,
               A_loc, &one, &one, desc_a, ipiv,
               B_loc, &one, &one, desc_b,
               &info, (std::size_t)1);

            gather_local_to_mpfr(result, B_loc, lld_b, n, nrhs, mpc, side.ctx);

            std::free(A_loc);
            std::free(B_loc);
            std::free(ipiv);
            mpc.finalize();
        };

        MpfrMatrix res_a(n, nrhs, prec); run_side(a, res_a);
        MpfrMatrix res_b(n, nrhs, prec); run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "trans=%c", trans);
        mirror_report_result("PGETRS", ps, err, nullptr, nullptr, config);
    }
}
