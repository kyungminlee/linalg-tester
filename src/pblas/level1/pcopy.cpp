/* pcopy.cpp -- PBLAS Level 1 PCOPY accuracy tester */

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

/* Fortran-ABI function pointer for PDCOPY (no hidden char lengths) */
extern "C" typedef void (*pcopy_fn_t)(
    const int  *n,
    const void *x,  const int *ix, const int *jx, const int *descx, const int *incx,
    void       *y,  const int *iy, const int *jy, const int *descy, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pcopy(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PCOPY", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pcopy_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PCOPY", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;  /* vector length */
    mpfr_prec_t prec = ctx.prec;

    /* Generate global source vector */
    unsigned seed_x = params.seed;
    void *x_g = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);

    /* Local dimensions for n x 1 distributed matrix */
    int loc_m = pc.local_rows(n);
    int loc_n = pc.local_cols(1);
    int lld_x = std::max(1, loc_m);
    int lld_y = std::max(1, loc_m);

    /* Allocate local buffers */
    void *x_loc = std::calloc(static_cast<std::size_t>(lld_x) * std::max(1, loc_n), ctx.typesize);

    unsigned sentinel_seed = 0xFBAD0016u;
    void *y_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_y) * std::max(1, loc_n),
                                       ctx.typesize, sentinel_seed);

    /* Scatter source to local */
    scatter_global_to_local(x_loc, lld_x, x_g, n,
                            n, 1, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Create descriptors */
    int desc_x[9], desc_y[9];
    pc.make_desc(desc_x, n, 1, lld_x);
    pc.make_desc(desc_y, n, 1, lld_y);

    int one = 1, incx = 1, incy = 1;

    /* Call PBLAS: y = x */
    fn(&n, x_loc, &one, &one, desc_x, &incx,
           y_loc, &one, &one, desc_y, &incy);

    /* MPFR reference: y_ref = x_in (exact copy) */
    MpfrMatrix y_ref(n, 1, prec);
    custom_to_mpfr_mat(y_ref, x_g, n, ctx);

    /* Extract local reference and compare */
    MpfrMatrix y_local_ref(loc_m, std::max(1, loc_n), prec);
    extract_local_mpfr(y_local_ref, y_ref, n, 1, pc.mb, pc.nb,
                       pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

    ErrorResult err = compute_error_matrix(y_local_ref, y_loc, lld_y, ctx);
    SentinelResult sr = check_matrix_sentinels(y_loc, loc_m, std::max(1, loc_n), lld_y,
                                                ctx.typesize, sentinel_seed);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d", n, pc.bc.nprow, pc.bc.npcol);
        report_result("PCOPY", params_str, err, &sr, format);
    }

    std::free(x_g);
    std::free(x_loc);
    std::free(y_loc);
    pc.finalize();
}
