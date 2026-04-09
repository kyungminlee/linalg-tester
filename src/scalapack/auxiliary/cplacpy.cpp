/* cplacpy.cpp -- ScaLAPACK CPLACPY (complex PLACPY) accuracy tester */

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

extern "C" typedef void (*placpy_fn_t)(const char *, const int *, const int *,
                                        const void *, const int *, const int *, const int *,
                                        void *, const int *, const int *, const int *,
                                        std::size_t);

void test_cplacpy(const TesterCtx &ctx, void *lib, const char *sym,
                  const TestParams &params, const std::string &format)
{
    int mb = params.mb > 0 ? params.mb : params.m;
    int nb = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb, nb)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPLACPY", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<placpy_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPLACPY", "error=symbol_not_found", br, format);
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
    int lld_b = std::max(1, loc_m);

    /* Allocate local A and B (B zeroed) */
    void *A_loc = std::calloc(static_cast<std::size_t>(lld_a) * std::max(1, loc_n), ctx.typesize);
    void *B_loc = std::calloc(static_cast<std::size_t>(lld_b) * std::max(1, loc_n), ctx.typesize);

    /* Scatter A */
    scatter_global_to_local(A_loc, lld_a, A_g, m,
                            m, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Descriptors */
    int desc_a[9], desc_b[9];
    pc.make_desc(desc_a, m, n, lld_a);
    pc.make_desc(desc_b, m, n, lld_b);

    /* Call PZLACPY with uplo='A' (copy all) */
    char uplo = 'A';
    int one = 1;
    fn(&uplo, &m, &n, A_loc, &one, &one, desc_a,
       B_loc, &one, &one, desc_b, (std::size_t)1);

    /* Reference: build MPFR from global A, extract local, compare with B_loc */
    MpfrComplexMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_g, m, ctx);

    MpfrComplexMatrix A_local_ref(loc_m, loc_n, prec);
    extract_local_mpfr_complex(A_local_ref, A_mpfr, m, n, pc.mb, pc.nb,
                                pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol);

    ErrorResult err = compute_error_complex_matrix(A_local_ref, B_loc, lld_b, ctx);

    if (pc.bc.is_root()) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=A m=%d n=%d grid=%dx%d", m, n,
                      pc.bc.nprow, pc.bc.npcol);
        LapackResult lr = {err.max_relative / eps, -1.0, 0};
        report_lapack_result("CPLACPY", params_str, lr, format);
    }

    std::free(A_g);
    std::free(A_loc); std::free(B_loc);

    pc.finalize();
}
