/* blacs_gamn2d.cpp -- Mirror tester for BLACS global min */
/* Compares xGAMN2D across two library implementations.
   On single process: result = input (min of 1 element).
   Compares results between sides. */

#include "../mirror_blacs.h"
#include "../../mirror_ctx.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"
#include "../../../src/blacs/blacs_common.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

/* Fortran ABI for xGAMN2D */
extern "C" typedef void (*gamn2d_fn_t)(
    const int *ictxt, const char *scope, const char *top,
    const int *m, const int *n, void *a, const int *lda,
    int *ra, int *ca, const int *ldia,
    const int *rdest, const int *cdest,
    std::size_t scope_len, std::size_t top_len
);

void mirror_test_blacs_gamn2d(const MirrorSide &a, const MirrorSide &b,
                               const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    mpfr_prec_t prec = config.prec;

    /* Generate canonical MPFR data */
    unsigned seed_A = params.seed;
    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    char scope = 'A';
    char top = ' ';
    int ldia = -1;  /* skip index tracking */
    int rdest = 0, cdest = 0;

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        /* Load BLACS context symbols */
        BlacsCtx bc;
        if (!bc.load(side.lib)) {
            std::fprintf(stderr, "mirror_blacs_gamn2d: failed to load BLACS symbols from side %s\n",
                         side.label.c_str());
            return;
        }

        /* Load gamn2d symbol */
        auto *fn_gamn = reinterpret_cast<gamn2d_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        /* Initialize 1x1 grid */
        bc.init_grid(1, 1, 'R');
        if (!bc.in_grid()) {
            bc.finalize();
            return;
        }

        /* Materialize MPFR data to native format */
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);

        /* Call GAMN2D -- modifies A in place, ra/ca = nullptr (ldia = -1) */
        fn_gamn(&bc.ictxt, &scope, &top, &m, &n, native_A, &lda,
                nullptr, nullptr, &ldia, &rdest, &cdest,
                (std::size_t)1, (std::size_t)1);

        /* Convert result back to MPFR */
        custom_to_mpfr_mat(result, native_A, lda, side.ctx);

        bc.finalize();
        std::free(native_A);
    };

    MpfrMatrix res_a(m, n, prec);
    MpfrMatrix res_b(m, n, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d scope=A", m, n);
    mirror_report_result("BLACS_GAMN2D", params_str, err, nullptr, nullptr, config);
}
