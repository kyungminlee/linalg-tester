/* blacs_gamn2d.cpp -- BLACS global min accuracy test */
/* Tests xGAMN2D: global element-wise minimum across process grid */
/* For single process, the result should equal the input. */

#include "../blacs.h"
#include "../blacs_common.h"
#include "../../core/generators.h"
#include "../../core/error_metrics.h"
#include "../../core/mpfr_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void test_blacs_gamn2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format)
{
    BlacsCtx bc;
    if (!bc.load(lib)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("BLACS_GAMN2D", "error=symbols_not_found", br, format);
        return;
    }

    /* Load typed gamn2d symbol (same signature as gamx2d) */
    auto *fn_gamn = reinterpret_cast<void (*)(
        const int *, const char *, const char *,
        const int *, const int *, void *, const int *,
        int *, int *, const int *,
        const int *, const int *,
        std::size_t, std::size_t)>(load_sym(lib, sym));

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;

    bc.init_grid(1, 1, 'R');
    if (!bc.in_grid()) {
        bc.finalize();
        return;
    }

    /* Generate random m-by-n matrix */
    unsigned seed_A = params.seed;
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, ctx.prec, &seed_A);

    /* Save original in MPFR */
    MpfrMatrix A_ref(m, n, ctx.prec);
    {
        const char *ap = static_cast<const char *>(A);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                ctx.to_mpfr(A_ref.at(i, j),
                            ap + (static_cast<std::size_t>(j) * lda + i) * ctx.typesize);
    }

    /* RA, CA: index arrays (-1 to skip tracking) */
    int ldia = -1;
    int rdest = 0, cdest = 0;
    char scope = 'A';
    char top = ' ';

    fn_gamn(&bc.ictxt, &scope, &top, &m, &n, A, &lda,
            nullptr, nullptr, &ldia, &rdest, &cdest,
            (std::size_t)1, (std::size_t)1);

    /* For single process: min of 1 element = itself */
    ErrorResult err = compute_error_matrix(A_ref, A, lda, ctx);

    bc.finalize();

    if (bc.mypnum == 0 || bc.nprocs <= 1) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "m=%d n=%d scope=A nprocs=%d", m, n, bc.nprocs);
        BlacsResult br;
        br.passed = (err.max_relative == 0.0 && err.nan_inf_mismatches == 0);
        br.max_error = err.max_relative;
        br.data_mismatches = err.nan_inf_mismatches;
        report_blacs_result("BLACS_GAMN2D", params_str, br, format);
    }

    std::free(A);
}
