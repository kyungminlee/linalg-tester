/* cgetri.cpp -- Mirror tester for LAPACK complex GETRI (LU-based inverse) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string replace_suffix(const std::string &sym, const char *from, const char *to) {
    std::string s = sym;
    auto pos = s.find(from);
    if (pos != std::string::npos)
        s.replace(pos, std::strlen(from), to);
    return s;
}

extern "C" typedef void (*cgetrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info);

extern "C" typedef void (*cgetri_fn_t)(
    const int *n, void *A, const int *lda, const int *ipiv,
    void *work, const int *lwork, int *info);

void mirror_test_cgetri(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(n, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result) {
        std::string cgetrf_sym = replace_suffix(side.sym, "getri", "getrf");
        auto *cgetrf = reinterpret_cast<cgetrf_fn_t>(
            load_sym(side.lib, cgetrf_sym.c_str()));
        auto *cgetri = reinterpret_cast<cgetri_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
        int *ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));
        int info = 0;

        /* LU factorize */
        cgetrf(&n, &n, native_A, &lda, ipiv, &info);

        /* Workspace query for GETRI */
        int lwork_query = -1;
        void *work_q = std::malloc(side.ctx.typesize);
        info = 0;
        cgetri(&n, native_A, &lda, ipiv, work_q, &lwork_query, &info);
        int lwork = mirror_query_lwork_complex(work_q, side.ctx);
        std::free(work_q);

        /* Re-factorize A */
        std::free(native_A);
        native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
        info = 0;
        cgetrf(&n, &n, native_A, &lda, ipiv, &info);

        /* Compute inverse */
        void *work = std::malloc(
            static_cast<std::size_t>(lwork) * side.ctx.typesize);
        info = 0;
        cgetri(&n, native_A, &lda, ipiv, work, &lwork, &info);

        custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);

        std::free(native_A);
        std::free(ipiv);
        std::free(work);
    };

    MpfrComplexMatrix res_a(n, n, prec);
    MpfrComplexMatrix res_b(n, n, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "n=%d", n);
    mirror_report_result("CGETRI", params_str, err,
                          nullptr, nullptr, config);
}
