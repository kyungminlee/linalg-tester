/* blacs_setup.cpp -- BLACS context lifecycle test */
/* Tests BLACS_PINFO, BLACS_GET, BLACS_GRIDINIT, BLACS_GRIDINFO, BLACS_GRIDEXIT */

#include "../blacs.h"
#include "../blacs_common.h"

#include <cstdio>
#include <cmath>
#include <vector>

/* ------------------------------------------------------------------ */
/* Compute valid grid factorizations for nprocs                        */
/* Returns a list of (nprow, npcol) pairs where nprow*npcol <= nprocs  */
/* ------------------------------------------------------------------ */

static std::vector<std::pair<int,int>> grid_topologies(int nprocs)
{
    std::vector<std::pair<int,int>> grids;
    grids.push_back({1, 1});
    if (nprocs > 1) {
        grids.push_back({1, nprocs});       /* 1D column */
        grids.push_back({nprocs, 1});       /* 1D row */
        /* Closest-to-square factorization */
        int sq = static_cast<int>(std::sqrt(static_cast<double>(nprocs)));
        for (int r = sq; r >= 1; --r) {
            if (nprocs % r == 0) {
                int c = nprocs / r;
                if (r != 1 && c != 1 && r != nprocs && c != nprocs)
                    grids.push_back({r, c});
                break;
            }
        }
    }
    return grids;
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_blacs_setup(const TesterCtx &ctx, void *lib, const char * /*sym*/,
                      const TestParams & /*params*/, const std::string &format)
{
    BlacsCtx bc;
    if (!bc.load(lib)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("BLACS_SETUP", "error=symbols_not_found", br, format);
        return;
    }

    /* Query process info */
    bc.fn_pinfo(&bc.mypnum, &bc.nprocs);

    bool all_passed = true;
    auto topologies = grid_topologies(bc.nprocs);

    for (char order : {'R', 'C'}) {
        for (auto &[nr, nc] : topologies) {
            /* Initialize grid */
            int ictxt;
            int neg1 = -1, zero = 0;
            bc.fn_get(&neg1, &zero, &ictxt);
            bc.fn_gridinit(&ictxt, &order, &nr, &nc, (std::size_t)1);

            int nprow_out, npcol_out, myrow_out, mycol_out;
            bc.fn_gridinfo(&ictxt, &nprow_out, &npcol_out, &myrow_out, &mycol_out);

            bool ok = true;
            if (nprow_out != nr || npcol_out != nc)
                ok = false;

            /* Processes in the grid should have valid coordinates */
            if (myrow_out >= 0 && mycol_out >= 0) {
                if (myrow_out >= nr || mycol_out >= nc)
                    ok = false;
            }
            /* Processes outside the grid should have myrow=-1, mycol=-1 */
            if (bc.mypnum >= nr * nc) {
                if (myrow_out != -1 || mycol_out != -1)
                    ok = false;
            }

            bc.fn_gridexit(&ictxt);

            if (!ok)
                all_passed = false;
        }
    }

    /* Only root reports in multi-process mode */
    if (bc.mypnum == 0) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "nprocs=%d topologies=%d",
                      bc.nprocs, static_cast<int>(topologies.size()) * 2);
        BlacsResult br = {all_passed, -1.0, 0};
        report_blacs_result("BLACS_SETUP", params_str, br, format);
    }
}
