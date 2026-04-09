/* cplange.cpp -- ScaLAPACK CPLANGE (complex PLANGE) accuracy tester */

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

/* PZLANGE returns double (norm is always real) */
extern "C" typedef double (*plange_fn_t)(const char *norm, const int *m, const int *n,
                                          const void *a, const int *ia, const int *ja,
                                          const int *desca, void *work,
                                          std::size_t norm_len);

/* ------------------------------------------------------------------ */
/* Complex max absolute element norm (M-norm)                          */
/* ------------------------------------------------------------------ */

static double mpfr_complex_mat_normM(const MpfrComplexMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar abs_val(prec), max_val(prec);
    mpfr_set_d(max_val.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            mpfr_complex_abs(abs_val.get(), A.re(i, j), A.im(i, j), MPFR_RNDN);
            if (mpfr_cmp(abs_val.get(), max_val.get()) > 0)
                mpfr_set(max_val.get(), abs_val.get(), MPFR_RNDN);
        }

    return mpfr_get_d(max_val.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Complex infinity norm (max row sum of |a_ij|)                       */
/* ------------------------------------------------------------------ */

static double mpfr_complex_mat_normI(const MpfrComplexMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar abs_val(prec), row_sum(prec), max_row(prec);
    mpfr_set_d(max_row.get(), 0.0, MPFR_RNDN);

    for (int i = 0; i < m; ++i) {
        mpfr_set_d(row_sum.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            mpfr_complex_abs(abs_val.get(), A.re(i, j), A.im(i, j), MPFR_RNDN);
            mpfr_add(row_sum.get(), row_sum.get(), abs_val.get(), MPFR_RNDN);
        }
        if (mpfr_cmp(row_sum.get(), max_row.get()) > 0)
            mpfr_set(max_row.get(), row_sum.get(), MPFR_RNDN);
    }

    return mpfr_get_d(max_row.get(), MPFR_RNDN);
}

void test_cplange(const TesterCtx &ctx, void *lib, const char *sym,
                  const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPLANGE", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<plange_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPLANGE", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    unsigned seed_A = params.seed;

    /* Generate random complex matrix A */
    void *A_g = gen_random_complex_array(m * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);

    /* Local dimensions */
    int loc_m = pc.local_rows(m);
    int loc_n = pc.local_cols(n);
    int lld_a = std::max(1, loc_m);

    /* Allocate local */
    void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n), ctx.typesize);

    /* Scatter */
    scatter_global_to_local(A_loc, lld_a, A_g, m,
                            m, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Descriptor */
    int desc_a[9];
    pc.make_desc(desc_a, m, n, lld_a);

    /* Work array (real-valued for PZLANGE) */
    int work_size = std::max(loc_m, loc_n) + 1;
    void *work = std::calloc(work_size, ctx.typesize);

    /* MPFR reference: build full complex matrix */
    MpfrComplexMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_g, m, ctx);

    int one = 1;

    /* Test each norm type */
    for (char norm_type : {'M', '1', 'I', 'F'}) {
        double result = fn(&norm_type, &m, &n, A_loc, &one, &one, desc_a,
                           work, (std::size_t)1);

        /* MPFR reference norm */
        double ref = 0.0;
        switch (norm_type) {
            case 'M': ref = mpfr_complex_mat_normM(A_mpfr); break;
            case '1': ref = mpfr_complex_mat_norm1(A_mpfr); break;
            case 'I': ref = mpfr_complex_mat_normI(A_mpfr); break;
            case 'F': ref = mpfr_complex_mat_normF(A_mpfr); break;
        }

        double rel_err = std::fabs(result - ref) / std::max(std::fabs(ref), 1e-300);

        if (pc.bc.is_root()) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "norm=%c m=%d n=%d grid=%dx%d", norm_type, m, n,
                          pc.bc.nprow, pc.bc.npcol);
            LapackResult lr = {rel_err / eps, -1.0, 0};
            report_lapack_result("CPLANGE", params_str, lr, format);
        }
    }

    std::free(A_g);
    std::free(A_loc);
    std::free(work);

    pc.finalize();
}
