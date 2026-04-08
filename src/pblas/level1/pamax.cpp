/* pamax.cpp -- PBLAS Level 1 PAMAX accuracy tester */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PDAMAX */
extern "C" typedef void (*pamax_fn_t)(
    const int  *n,
    void       *amax,
    int        *indx,
    const void *x,  const int *ix, const int *jx, const int *descx, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pamax(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PAMAX", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pamax_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PAMAX", "error=symbol_not_found", br, format);
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

    /* Result buffers */
    char amax_buf[64] = {};
    int indx_out = 0;

    /* Call PBLAS: find max |x[i]| and its 1-based index */
    fn(&n, amax_buf, &indx_out, x_loc, &one, &one, desc_x, &incx);

    /* MPFR reference: find max absolute value and its 1-based index */
    MpfrMatrix x_mpfr(n, 1, prec);
    custom_to_mpfr_mat(x_mpfr, x_g, n, ctx);

    MpfrScalar amax_ref(prec), abs_val(prec), max_abs(prec);
    mpfr_set_d(max_abs.get(), 0.0, MPFR_RNDN);
    int indx_ref = 1;  /* 1-based */

    for (int i = 0; i < n; ++i) {
        mpfr_abs(abs_val.get(), x_mpfr.at(i, 0), MPFR_RNDN);
        if (mpfr_cmp(abs_val.get(), max_abs.get()) > 0) {
            mpfr_set(max_abs.get(), abs_val.get(), MPFR_RNDN);
            indx_ref = i + 1;  /* 1-based global index */
        }
    }
    /* PDAMAX returns the actual value at the max index, not its absolute value */
    mpfr_set(amax_ref.get(), x_mpfr.at(indx_ref - 1, 0), MPFR_RNDN);

    /* Compare scalar amax value */
    ErrorResult err = compute_error_scalar(amax_ref, amax_buf, ctx);

    /* Compare index (exact match) */
    bool indx_match = (indx_out == indx_ref);

    /* If index mismatch, mark as failure by inflating error */
    if (!indx_match) {
        err.nan_inf_mismatches += 1;
    }

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d indx_ref=%d indx_out=%d",
                      n, pc.bc.nprow, pc.bc.npcol, indx_ref, indx_out);
        report_result("PAMAX", params_str, err, format);
    }

    std::free(x_g);
    std::free(x_loc);
    pc.finalize();
}
