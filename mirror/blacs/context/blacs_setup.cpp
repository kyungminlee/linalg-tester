/* blacs_setup.cpp -- Mirror tester for BLACS grid initialization */
/* Compares that both implementations initialize grids correctly and
   agree on grid topology parameters (nprow, npcol, myrow, mycol). */

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

void mirror_test_blacs_setup(const MirrorSide &a, const MirrorSide &b,
                              const TestParams &params, const MirrorConfig &config)
{
    /* Load BLACS context symbols from each side */
    BlacsCtx bc_a, bc_b;
    if (!bc_a.load(a.lib)) {
        std::fprintf(stderr, "mirror_blacs_setup: failed to load BLACS symbols from side A\n");
        return;
    }
    if (!bc_b.load(b.lib)) {
        std::fprintf(stderr, "mirror_blacs_setup: failed to load BLACS symbols from side B\n");
        return;
    }

    bool all_passed = true;

    for (char order : {'R', 'C'}) {
        /* Initialize 1x1 grid on each side */
        int nprow_req = 1, npcol_req = 1;

        /* Side A */
        int ictxt_a;
        {
            int mypnum_a, nprocs_a;
            bc_a.fn_pinfo(&mypnum_a, &nprocs_a);
            int neg1 = -1, zero = 0;
            bc_a.fn_get(&neg1, &zero, &ictxt_a);
            bc_a.fn_gridinit(&ictxt_a, &order, &nprow_req, &npcol_req, (std::size_t)1);
        }

        int nprow_a, npcol_a, myrow_a, mycol_a;
        bc_a.fn_gridinfo(&ictxt_a, &nprow_a, &npcol_a, &myrow_a, &mycol_a);

        /* Side B */
        int ictxt_b;
        {
            int mypnum_b, nprocs_b;
            bc_b.fn_pinfo(&mypnum_b, &nprocs_b);
            int neg1 = -1, zero = 0;
            bc_b.fn_get(&neg1, &zero, &ictxt_b);
            bc_b.fn_gridinit(&ictxt_b, &order, &nprow_req, &npcol_req, (std::size_t)1);
        }

        int nprow_b, npcol_b, myrow_b, mycol_b;
        bc_b.fn_gridinfo(&ictxt_b, &nprow_b, &npcol_b, &myrow_b, &mycol_b);

        /* Compare grid parameters */
        bool ok = true;
        if (nprow_a != nprow_b || npcol_a != npcol_b) ok = false;
        if (myrow_a != myrow_b || mycol_a != mycol_b) ok = false;

        /* Verify expected values for 1x1 grid */
        if (nprow_a != 1 || npcol_a != 1) ok = false;
        if (myrow_a != 0 || mycol_a != 0) ok = false;

        if (!ok)
            all_passed = false;

        /* Finalize grids */
        bc_a.fn_gridexit(&ictxt_a);
        bc_b.fn_gridexit(&ictxt_b);
    }

    /* Report using MPFR error infrastructure: encode pass/fail as a scalar.
       We use a trivial ErrorResult since this is a topology comparison. */
    ErrorResult err;
    err.max_relative = all_passed ? 0.0 : 1.0;
    err.normwise_relative = err.max_relative;
    err.max_absolute_at_zero = -1.0;
    err.nan_inf_mismatches = all_passed ? 0 : 1;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "grid=1x1 orders=R,C");
    mirror_report_result("BLACS_SETUP", params_str, err, nullptr, nullptr, config);
}
