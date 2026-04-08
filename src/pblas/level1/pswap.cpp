/* pswap.cpp -- PBLAS Level 1 PSWAP accuracy tester */

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

/* Fortran-ABI function pointer for PDSWAP (no hidden char lengths) */
extern "C" typedef void (*pswap_fn_t)(
    const int  *n,
    void       *x,  const int *ix, const int *jx, const int *descx, const int *incx,
    void       *y,  const int *iy, const int *jy, const int *descy, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pswap(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSWAP", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pswap_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSWAP", "error=symbol_not_found", br, format);
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

    /* Allocate local with sentinels */
    unsigned sentinel_x = 0xFBAD0014u;
    unsigned sentinel_y = 0xFBAD0015u;
    void *x_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_x) * std::max(1, loc_n),
                                       ctx.typesize, sentinel_x);
    void *y_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_y) * std::max(1, loc_n),
                                       ctx.typesize, sentinel_y);

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

    /* Call PBLAS */
    fn(&n, x_loc, &one, &one, desc_x, &incx,
           y_loc, &one, &one, desc_y, &incy);

    /* MPFR reference: after swap, x should contain original y and vice versa */
    MpfrMatrix x_ref(n, 1, prec);
    MpfrMatrix y_ref(n, 1, prec);
    custom_to_mpfr_mat(x_ref, y_g, n, ctx);  /* x_ref = original y */
    custom_to_mpfr_mat(y_ref, x_g, n, ctx);  /* y_ref = original x */

    /* Extract local references and compare */
    MpfrMatrix x_local_ref(loc_m, std::max(1, loc_n), prec);
    MpfrMatrix y_local_ref(loc_m, std::max(1, loc_n), prec);
    extract_local_mpfr(x_local_ref, x_ref, n, 1, pc.mb, pc.nb,
                       pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);
    extract_local_mpfr(y_local_ref, y_ref, n, 1, pc.mb, pc.nb,
                       pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

    ErrorResult err_x = compute_error_matrix(x_local_ref, x_loc, lld_x, ctx);
    ErrorResult err_y = compute_error_matrix(y_local_ref, y_loc, lld_y, ctx);

    SentinelResult sr_x = check_matrix_sentinels(x_loc, loc_m, std::max(1, loc_n), lld_x,
                                                  ctx.typesize, sentinel_x);
    SentinelResult sr_y = check_matrix_sentinels(y_loc, loc_m, std::max(1, loc_n), lld_y,
                                                  ctx.typesize, sentinel_y);

    /* Combine: take max error from both vectors */
    ErrorResult err;
    err.max_relative = std::max(err_x.max_relative, err_y.max_relative);
    err.normwise_relative = std::max(err_x.normwise_relative, err_y.normwise_relative);
    err.max_absolute_at_zero = std::max(err_x.max_absolute_at_zero, err_y.max_absolute_at_zero);
    err.nan_inf_mismatches = err_x.nan_inf_mismatches + err_y.nan_inf_mismatches;

    SentinelResult sr;
    sr.passed = sr_x.passed && sr_y.passed;
    sr.corrupted_count = sr_x.corrupted_count + sr_y.corrupted_count;
    sr.first_offset = sr_x.passed ? sr_y.first_offset : sr_x.first_offset;

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d", n, pc.bc.nprow, pc.bc.npcol);
        report_result("PSWAP", params_str, err, &sr, format);
    }

    std::free(x_g);
    std::free(y_g);
    std::free(x_loc);
    std::free(y_loc);
    pc.finalize();
}
