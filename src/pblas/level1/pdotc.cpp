/* pdotc.cpp -- PBLAS Level 1 PDOTC accuracy tester (complex conjugated dot) */

#include "../pblas.h"
#include "../pblas_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer for PZDOTC */
extern "C" typedef void (*pdotc_fn_t)(
    const int  *n,
    void       *dotc,
    const void *x,  const int *ix, const int *jx, const int *descx, const int *incx,
    const void *y,  const int *iy, const int *jy, const int *descy, const int *incy
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_pdotc(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    if (!ctx.complex_mode) {
        std::fprintf(stderr, "PDOTC requires --complex\n");
        return;
    }

    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PDOTC", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<pdotc_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("PDOTC", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int n = params.m;  /* vector length */
    mpfr_prec_t prec = ctx.prec;

    /* Generate global complex vectors */
    unsigned seed_x = params.seed;
    unsigned seed_y = params.seed + 1;
    void *x_g = gen_random_complex_array(n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_x);
    void *y_g = gen_random_complex_array(n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_y);

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

    /* Result buffer for complex scalar output */
    char result_buf[64] = {};

    /* Call PBLAS: dotc = conj(x)^T * y */
    fn(&n, result_buf,
       x_loc, &one, &one, desc_x, &incx,
       y_loc, &one, &one, desc_y, &incy);

    /* MPFR reference: dotc_ref = sum of conj(x[i]) * y[i] */
    MpfrComplexMatrix x_mpfr(n, 1, prec);
    MpfrComplexMatrix y_mpfr(n, 1, prec);
    custom_to_mpfr_complex_mat(x_mpfr, x_g, n, ctx);
    custom_to_mpfr_complex_mat(y_mpfr, y_g, n, ctx);

    MpfrComplexScalar dotc_ref(prec);
    {
        MpfrComplexScalar conj_x(prec);
        mpfr_set_d(dotc_ref.re(), 0.0, MPFR_RNDN);
        mpfr_set_d(dotc_ref.im(), 0.0, MPFR_RNDN);
        for (int i = 0; i < n; ++i) {
            mpfr_complex_conj(conj_x.re(), conj_x.im(),
                              x_mpfr.re(i, 0), x_mpfr.im(i, 0), MPFR_RNDN);
            mpfr_complex_fma(dotc_ref.re(), dotc_ref.im(),
                             conj_x.re(), conj_x.im(),
                             y_mpfr.re(i, 0), y_mpfr.im(i, 0), MPFR_RNDN);
        }
    }

    /* Compare complex scalar result */
    ErrorResult err = compute_error_complex_scalar(dotc_ref, result_buf, ctx);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "n=%d grid=%dx%d", n, pc.bc.nprow, pc.bc.npcol);
        report_result("PDOTC", params_str, err, format);
    }

    std::free(x_g);
    std::free(y_g);
    std::free(x_loc);
    std::free(y_loc);
    pc.finalize();
}
