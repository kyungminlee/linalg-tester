/* planhe.cpp -- ScaLAPACK PLANHE (PZLANHE Hermitian matrix norm) accuracy tester */

#include "../scalapack.h"
#include "../scalapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* PZLANHE returns double (norm is always real) */
extern "C" typedef double (*pzlanhe_fn_t)(const char *norm, const char *uplo,
                                           const int *n,
                                           const void *a, const int *ia, const int *ja,
                                           const int *desca, void *work,
                                           std::size_t, std::size_t);

/* ------------------------------------------------------------------ */
/* Build full Hermitian matrix from stored triangle (MPFR complex)     */
/* ------------------------------------------------------------------ */

static void mpfr_hermitianize(MpfrComplexMatrix &A, char uplo)
{
    int n = A.rows();
    if (uplo == 'U') {
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < n; ++i) {
                /* A(i,j) = conj(A(j,i)) */
                mpfr_set(A.re(i, j), A.re(j, i), MPFR_RNDN);
                mpfr_neg(A.im(i, j), A.im(j, i), MPFR_RNDN);
            }
    } else {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < j; ++i) {
                /* A(i,j) = conj(A(j,i)) */
                mpfr_set(A.re(i, j), A.re(j, i), MPFR_RNDN);
                mpfr_neg(A.im(i, j), A.im(j, i), MPFR_RNDN);
            }
    }
    /* Ensure diagonal is real (zero out imaginary part) */
    for (int i = 0; i < n; ++i)
        mpfr_set_d(A.im(i, i), 0.0, MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Complex max absolute element norm (M-norm) for Hermitian matrix     */
/* ------------------------------------------------------------------ */

static double mpfr_complex_hermat_normM(const MpfrComplexMatrix &A)
{
    int n = A.rows();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar abs_val(prec), max_val(prec);
    mpfr_set_d(max_val.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            mpfr_complex_abs(abs_val.get(), A.re(i, j), A.im(i, j), MPFR_RNDN);
            if (mpfr_cmp(abs_val.get(), max_val.get()) > 0)
                mpfr_set(max_val.get(), abs_val.get(), MPFR_RNDN);
        }

    return mpfr_get_d(max_val.get(), MPFR_RNDN);
}

void test_planhe(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PLANHE", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pzlanhe_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PLANHE", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        /* Generate Hermitian matrix */
        void *A_g = gen_hermitian_array(n, uplo, ctx.typesize,
                                         ctx.from_mpfr_complex, prec, &seed_A);

        /* Local dimensions */
        int loc_m = pc.local_rows(n);
        int loc_n = pc.local_cols(n);
        int lld_a = std::max(1, loc_m);

        /* Allocate local */
        void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n), ctx.typesize);

        /* Scatter */
        scatter_global_to_local(A_loc, lld_a, A_g, n,
                                n, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                                ctx.typesize);

        /* Descriptor */
        int desc_a[9];
        pc.make_desc(desc_a, n, n, lld_a);

        /* Work array */
        int work_size = std::max(loc_m, loc_n) + 1;
        void *work = std::calloc(work_size, ctx.typesize);

        /* MPFR reference: build full Hermitian matrix */
        MpfrComplexMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_complex_mat(A_mpfr, A_g, n, ctx);
        mpfr_hermitianize(A_mpfr, uplo);

        int one = 1;

        /* Test each norm type */
        for (char norm_type : {'M', '1', 'I', 'F'}) {
            double result = fn(&norm_type, &uplo, &n, A_loc, &one, &one, desc_a,
                               work, (std::size_t)1, (std::size_t)1);

            /* MPFR reference norm on full Hermitian matrix */
            double ref = 0.0;
            switch (norm_type) {
                case 'M': ref = mpfr_complex_hermat_normM(A_mpfr); break;
                case '1': ref = mpfr_hermat_norm1(A_mpfr, uplo); break;
                case 'I': ref = mpfr_hermat_norm1(A_mpfr, uplo); break; /* 1-norm = I-norm for Hermitian */
                case 'F': ref = mpfr_complex_mat_normF(A_mpfr); break;
            }

            double rel_err = std::fabs(result - ref) / std::max(std::fabs(ref), 1e-300);

            if (pc.bc.is_root()) {
                char params_str[128];
                std::snprintf(params_str, sizeof(params_str),
                              "norm=%c uplo=%c n=%d grid=%dx%d",
                              norm_type, uplo, n,
                              pc.bc.nprow, pc.bc.npcol);
                LapackResult lr = {rel_err / eps, -1.0, 0};
                report_lapack_result("PLANHE", params_str, lr, format);
            }
        }

        std::free(A_g);
        std::free(A_loc);
        std::free(work);
    }

    pc.finalize();
}
