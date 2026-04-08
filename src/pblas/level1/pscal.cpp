/* pscal.cpp -- PBLAS Level 1 PSCAL accuracy tester */

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

/* Fortran-ABI function pointer for PDSCAL (no hidden char lengths) */
extern "C" typedef void (*pscal_fn_t)(
    const int  *n,
    const void *alpha,
    void       *x,  const int *ix, const int *jx, const int *descx, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pscal(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSCAL", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pscal_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PSCAL", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;  /* vector length */
    mpfr_prec_t prec = ctx.prec;

    /* Generate global vector and alpha */
    unsigned seed_x = params.seed;
    unsigned seed_a = params.seed + 1;
    void *x_g   = gen_random_array(n, ctx.typesize, ctx.from_mpfr, prec, &seed_x);
    void *alpha  = gen_random_array(1, ctx.typesize, ctx.from_mpfr, prec, &seed_a);

    /* Local dimensions for n x 1 distributed matrix */
    int loc_m = pc.local_rows(n);
    int loc_n = pc.local_cols(1);
    int lld_x = std::max(1, loc_m);

    /* Allocate local with sentinel */
    unsigned sentinel_seed = 0xFBAD0015u;
    void *x_loc = alloc_with_sentinel(static_cast<std::size_t>(lld_x) * std::max(1, loc_n),
                                       ctx.typesize, sentinel_seed);

    /* Scatter global to local */
    scatter_global_to_local(x_loc, lld_x, x_g, n,
                            n, 1, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Create descriptor */
    int desc_x[9];
    pc.make_desc(desc_x, n, 1, lld_x);

    int one = 1, incx = 1;

    /* Call PBLAS */
    fn(&n, alpha, x_loc, &one, &one, desc_x, &incx);

    /* MPFR reference: x_ref[i] = alpha * x_in[i] */
    MpfrScalar mpfr_alpha(prec);
    ctx.to_mpfr(mpfr_alpha.get(), alpha);

    MpfrMatrix x_mpfr(n, 1, prec);
    MpfrMatrix x_ref(n, 1, prec);
    custom_to_mpfr_mat(x_mpfr, x_g, n, ctx);

    for (int i = 0; i < n; ++i)
        mpfr_mul(x_ref.at(i, 0), mpfr_alpha.get(), x_mpfr.at(i, 0), MPFR_RNDN);

    /* Extract local reference and compare */
    MpfrMatrix x_local_ref(loc_m, std::max(1, loc_n), prec);
    extract_local_mpfr(x_local_ref, x_ref, n, 1, pc.mb, pc.nb,
                       pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

    ErrorResult err = compute_error_matrix(x_local_ref, x_loc, lld_x, ctx);
    SentinelResult sr = check_matrix_sentinels(x_loc, loc_m, std::max(1, loc_n), lld_x,
                                                ctx.typesize, sentinel_seed);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d", n, pc.bc.nprow, pc.bc.npcol);
        report_result("PSCAL", params_str, err, &sr, format);
    }

    std::free(x_g);
    std::free(alpha);
    std::free(x_loc);
    pc.finalize();
}
