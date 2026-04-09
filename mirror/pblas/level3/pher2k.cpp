/* pher2k.cpp -- Mirror tester for PBLAS Level 3 PHER2K (complex-only) */

#include "../mirror_pblas_common.h"

extern "C" typedef void (*pher2k_fn_t)(
    const char *uplo, const char *trans,
    const int *n, const int *k,
    const void *alpha,
    const void *A, const int *ia, const int *ja, const int *desca,
    const void *B, const int *ib, const int *jb, const int *descb,
    const void *beta,
    void *C, const int *ic, const int *jc, const int *descc,
    std::size_t, std::size_t);

/* Helper: materialize an MpfrScalar as a real native value.
   In complex mode, ctx.typesize is the complex element size (e.g. 16),
   but beta for PHER2K is a real scalar of size typesize/2. */
static void *mpfr_real_scalar_to_native(const MpfrScalar &src,
                                         const TesterCtx &ctx)
{
    std::size_t real_ts = ctx.typesize / 2;
    char *arr = static_cast<char *>(std::malloc(real_ts));
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }
    ctx.from_mpfr(arr, const_cast<mpfr_t &>(src.get()), MPFR_RNDN);
    return static_cast<void *>(arr);
}

void mirror_test_pher2k(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int n = params.n, k = params.k;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'C'}) {
        int rows_AB = (trans == 'N') ? n : k;
        int cols_AB = (trans == 'N') ? k : n;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_al = params.seed + 3;
        unsigned seed_be = params.seed + 4;

        MpfrComplexMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix C_mpfr(n, n, prec);
        MpfrComplexScalar alpha_mpfr(prec);
        MpfrScalar beta_mpfr(prec);

        gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);
        gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);
        gen_mpfr_hermitian_matrix(C_mpfr, uplo, prec, &seed_C);
        gen_mpfr_random_complex_scalar(alpha_mpfr, prec, &seed_al);
        gen_mpfr_random_scalar(beta_mpfr, prec, &seed_be);

        auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
            MirrorPblasCtx mpc;
            if (!mpc.init(s, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             s.label.c_str());
                return;
            }

            int loc_rAB = mpc.local_rows(rows_AB);
            int loc_rC  = mpc.local_rows(n);
            int lld_ab = std::max(1, loc_rAB);
            int lld_c  = std::max(1, loc_rC);

            void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, rows_AB, cols_AB,
                                                         lld_ab, mpc, s.ctx);
            void *B_loc = scatter_mpfr_complex_to_local(B_mpfr, rows_AB, cols_AB,
                                                         lld_ab, mpc, s.ctx);
            void *C_loc = scatter_mpfr_complex_to_local(C_mpfr, n, n,
                                                         lld_c, mpc, s.ctx);
            void *alpha_n = mpfr_complex_scalar_to_native(alpha_mpfr, s.ctx);
            void *beta_n  = mpfr_real_scalar_to_native(beta_mpfr, s.ctx);

            int desc_a[9], desc_b[9], desc_c[9];
            mpc.make_desc(desc_a, rows_AB, cols_AB, lld_ab);
            mpc.make_desc(desc_b, rows_AB, cols_AB, lld_ab);
            mpc.make_desc(desc_c, n, n, lld_c);

            int one = 1;
            auto *fn = reinterpret_cast<pher2k_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&uplo, &trans, &n, &k,
               alpha_n,
               A_loc, &one, &one, desc_a,
               B_loc, &one, &one, desc_b,
               beta_n,
               C_loc, &one, &one, desc_c,
               (std::size_t)1, (std::size_t)1);

            gather_local_complex_to_mpfr(result, C_loc, lld_c, n, n,
                                          mpc, s.ctx);

            std::free(A_loc);
            std::free(B_loc);
            std::free(C_loc);
            std::free(alpha_n);
            std::free(beta_n);
            mpc.finalize();
        };

        MpfrComplexMatrix res_a(n, n, prec); run_side(a, res_a);
        MpfrComplexMatrix res_b(n, n, prec); run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char ps[128];
        std::snprintf(ps, sizeof(ps), "uplo=%c trans=%c", uplo, trans);
        mirror_report_result("PHER2K", ps, err, nullptr, nullptr, config);
    }}
}
