/* pdot.cpp -- PBLAS Level 1 PDOT accuracy tester */

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

/* Fortran-ABI function pointer for PDDOT (no hidden char lengths) */
extern "C" typedef void (*pdot_fn_t)(
    const int  *n,
    void       *dot,
    const void *x,  const int *ix, const int *jx, const int *descx, const int *incx,
    const void *y,  const int *iy, const int *jy, const int *descy, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pdot(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PDOT", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pdot_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PDOT", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;  /* vector length */
    mpfr_prec_t prec = ctx.prec;

    /* Generate global vectors */
    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;
    void *x_g = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *y_g = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_y);

    /* Local dimensions for n x 1 distributed matrix */
    int loc_m = pc.local_rows(n);
    int loc_n = pc.local_cols(1);
    int lld_x = std::max(1, loc_m);
    int lld_y = std::max(1, loc_m);

    /* Allocate local buffers */
    void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_n), ctx.typesize);
    void *y_loc = std::calloc(static_cast<std::size_t>(lld_y) * std::max(1, loc_n), ctx.typesize);

    /* Scatter global to local */
    scatter_global_to_local(x_loc, lld_x, x_g, n,
                            n, 1, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);
    scatter_global_to_local(y_loc, lld_y, y_g, n,
                            n, 1, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Create descriptors */
    int desc_x[9], desc_y[9];
    pc.make_desc(desc_x, n, 1, lld_x);
    pc.make_desc(desc_y, n, 1, lld_y);

    int one = 1, incx = 1, incy = 1;

    /* Result buffer for scalar output */
    char result_buf[64] = {};

    /* Call PBLAS: dot = x^T * y */
    fn(&n, result_buf,
       x_loc, &one, &one, desc_x, &incx,
       y_loc, &one, &one, desc_y, &incy);

    /* MPFR reference: dot_ref = sum of x[i]*y[i] using FMA */
    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrMatrix y_mpfr(n, 1, prec);
    custom_to_mpfr_mat(x_mpfr, x_g, n, ctx);
    custom_to_mpfr_mat(y_mpfr, y_g, n, ctx);

    MpfrScalar dot_ref(prec);
    mpfr_set_d(dot_ref.get(), 0.0, MPFR_RNDN);
    for (int i = 0; i < n; ++i)
        mpfr_fma(dot_ref.get(), x_mpfr.at(i, 0), y_mpfr.at(i, 0),
                 dot_ref.get(), MPFR_RNDN);

    /* Compare scalar result */
    ErrorResult err = compute_error_scalar(dot_ref, result_buf, ctx);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d", n, pc.bc.nprow, pc.bc.npcol);
        report_result("PDOT", params_str, err, format);
    }

    std::free(x_g);
    std::free(y_g);
    std::free(x_loc);
    std::free(y_loc);
    pc.finalize();
}
