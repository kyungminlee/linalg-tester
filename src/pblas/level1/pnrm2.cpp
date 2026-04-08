/* pnrm2.cpp -- PBLAS Level 1 PNRM2 accuracy tester */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"
#include "../../core/sentinel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PDNRM2 (no hidden char lengths) */
extern "C" typedef void (*pnrm2_fn_t)(
    const int  *n,
    void       *norm2,
    const void *x,  const int *ix, const int *jx, const int *descx, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pnrm2(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PNRM2", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pnrm2_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PNRM2", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;  /* vector length */
    mpfr_prec_t prec = ctx.prec;

    /* Generate global vector */
    unsigned seed_x = params.seed;
    void *x_g = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

    /* Local dimensions for n x 1 distributed matrix */
    int loc_m = pc.local_rows(n);
    int loc_n = pc.local_cols(1);
    int lld_x = std::max(1, loc_m);

    /* Allocate local buffer */
    void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_n), ctx.typesize);

    /* Scatter global to local */
    scatter_global_to_local(x_loc, lld_x, x_g, n,
                            n, 1, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Create descriptor */
    int desc_x[9];
    pc.make_desc(desc_x, n, 1, lld_x);

    int one = 1, incx = 1;

    /* Result buffer for scalar output */
    char result_buf[64] = {};

    /* Call PBLAS: norm2 = ||x||_2 */
    fn(&n, result_buf, x_loc, &one, &one, desc_x, &incx);

    /* MPFR reference: nrm2_ref = sqrt(sum(x[i]^2)) */
    MpfrMatrix x_mpfr(n, 1, prec);
    custom_to_mpfr_mat(x_mpfr, x_g, n, ctx);

    MpfrScalar sum_sq(prec), tmp(prec), nrm2_ref(prec);
    mpfr_set_d(sum_sq.get(), 0.0, MPFR_RNDN);
    for (int i = 0; i < n; ++i) {
        mpfr_mul(tmp.get(), x_mpfr.at(i, 0), x_mpfr.at(i, 0), MPFR_RNDN);
        mpfr_add(sum_sq.get(), sum_sq.get(), tmp.get(), MPFR_RNDN);
    }
    mpfr_sqrt(nrm2_ref.get(), sum_sq.get(), MPFR_RNDN);

    /* Compare scalar result */
    ErrorResult err = compute_error_scalar(nrm2_ref, result_buf, ctx);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d", n, pc.bc.nprow, pc.bc.npcol);
        report_result("PNRM2", params_str, err, format);
    }

    std::free(x_g);
    std::free(x_loc);
    pc.finalize();
}
