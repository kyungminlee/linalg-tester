/* cpgemr2d.cpp -- ScaLAPACK PZGEMR2D accuracy tester (complex matrix redistribution) */

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

extern "C" typedef void (*cpgemr2d_fn_t)(
    const int *m, const int *n,
    const void *a, const int *ia, const int *ja, const int *desca,
    void *b, const int *ib, const int *jb, const int *descb,
    const int *ictxt);

void test_cpgemr2d(const TesterCtx &ctx, void *lib, const char *sym,
                   const TestParams &params, const std::string &format)
{
    int mb1 = params.mb > 0 ? params.mb : params.m;
    int nb1 = params.nb > 0 ? params.nb : params.n;

    PblasCtx pc;
    if (!pc.init(lib, mb1, nb1)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGEMR2D", "error=init_failed", br, format);
        return;
    }

    auto *fn = reinterpret_cast<cpgemr2d_fn_t>(try_load_sym(lib, sym));
    if (!fn) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("CPGEMR2D", "error=symbol_not_found", br, format);
        pc.finalize();
        return;
    }

    int m = params.m, n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    unsigned seed_A = params.seed;

    /* Generate random complex A (m x n) */
    void *A_g = gen_random_complex_array(m * n, ctx.typesize,
                    ctx.from_mpfr_complex, prec, &seed_A);

    /* Source layout: (mb1, nb1) */
    int loc_m1 = pc.local_rows(m);
    int loc_n1 = pc.local_cols(n);
    int lld1 = std::max(1, loc_m1);

    void *A_loc = std::calloc(static_cast<std::size_t>(lld1) * std::max(1, loc_n1),
                              ctx.typesize);

    scatter_global_to_local(A_loc, lld1, A_g, m,
                            m, n, pc.mb, pc.nb,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    int descA[9];
    pc.make_desc(descA, m, n, lld1);

    /* Destination layout: different block sizes within the same BLACS context */
    int mb2 = std::max(1, mb1 / 2);
    int nb2 = std::max(1, nb1 / 2);
    /* Ensure different from source if possible */
    if (mb2 == mb1 && mb1 > 1) mb2 = mb1 - 1;
    if (nb2 == nb1 && nb1 > 1) nb2 = nb1 - 1;

    int loc_m2 = numroc(m, mb2, pc.bc.myrow, 0, pc.bc.nprow);
    int loc_n2 = numroc(n, nb2, pc.bc.mycol, 0, pc.bc.npcol);
    int lld2 = std::max(1, loc_m2);

    void *B_loc = std::calloc(static_cast<std::size_t>(lld2) * std::max(1, loc_n2),
                              ctx.typesize);

    /* Build descriptor for B with (mb2, nb2) */
    int descB[9];
    descB[0] = 1;             /* DTYPE_ */
    descB[1] = pc.bc.ictxt;   /* CTXT_  */
    descB[2] = m;             /* M_     */
    descB[3] = n;             /* N_     */
    descB[4] = mb2;           /* MB_    */
    descB[5] = nb2;           /* NB_    */
    descB[6] = 0;             /* RSRC_  */
    descB[7] = 0;             /* CSRC_  */
    descB[8] = lld2;          /* LLD_   */

    int one = 1;

    /* Call PZGEMR2D: redistribute from (mb1, nb1) to (mb2, nb2) */
    fn(&m, &n,
       A_loc, &one, &one, descA,
       B_loc, &one, &one, descB,
       &pc.bc.ictxt);

    /* Verification: scatter original A into the new layout and compare */
    void *B_ref = std::calloc(static_cast<std::size_t>(lld2) * std::max(1, loc_n2),
                              ctx.typesize);
    scatter_global_to_local(B_ref, lld2, A_g, m,
                            m, n, mb2, nb2,
                            pc.bc.myrow, pc.bc.mycol, pc.bc.nprow, pc.bc.npcol,
                            ctx.typesize);

    /* Compare B_loc vs B_ref element-by-element using complex error metric */
    double max_rel = 0.0;
    if (loc_m2 > 0 && loc_n2 > 0) {
        MpfrComplexMatrix ref_mpfr(loc_m2, loc_n2, prec);
        custom_to_mpfr_complex_mat(ref_mpfr, B_ref, lld2, ctx);

        ErrorResult err = compute_error_complex_matrix(ref_mpfr, B_loc, lld2, ctx);
        max_rel = err.max_relative;
    }

    if (pc.bc.is_root()) {
        char ps[128];
        std::snprintf(ps, sizeof(ps), "m=%d n=%d mb1=%d nb1=%d mb2=%d nb2=%d grid=%dx%d",
                      m, n, mb1, nb1, mb2, nb2, pc.bc.nprow, pc.bc.npcol);
        LapackResult lr;
        lr.residual = max_rel / eps;
        lr.orthogonality = -1.0;
        lr.info = 0;
        report_lapack_result("CPGEMR2D", ps, lr, format);
    }

    /* Cleanup */
    std::free(A_g);
    std::free(A_loc);
    std::free(B_loc);
    std::free(B_ref);

    pc.finalize();
}
