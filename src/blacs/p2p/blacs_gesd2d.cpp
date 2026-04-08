/* blacs_gesd2d.cpp -- BLACS point-to-point send/receive accuracy test */
/* Tests xGESD2D (send) and xGERV2D (receive) for data integrity */

#include "../blacs.h"
#include "../blacs_common.h"
#include "../../core/generators.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

void test_blacs_gesd2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format)
{
    BlacsCtx bc;
    if (!bc.load(lib)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("BLACS_GESD2D", "error=symbols_not_found", br, format);
        return;
    }

    /* Load typed send/receive symbols.
       sym is e.g. "dgesd2d_". Derive the receive symbol by replacing "gesd" with "gerv". */
    auto *fn_send = reinterpret_cast<void (*)(
        const int *, const int *, const int *, const void *, const int *,
        const int *, const int *)>(load_sym(lib, sym));

    /* Derive receive symbol: replace "gesd2d" with "gerv2d" in sym */
    std::string recv_sym(sym);
    auto pos = recv_sym.find("gesd2d");
    if (pos != std::string::npos)
        recv_sym.replace(pos, 6, "gerv2d");
    auto *fn_recv = reinterpret_cast<void (*)(
        const int *, const int *, const int *, void *, const int *,
        const int *, const int *)>(load_sym(lib, recv_sym.c_str()));

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;

    /* Initialize 1x1 grid for single process (or NxM for multi) */
    int nprow = 1, npcol = 1;
    if (bc.nprocs <= 0) {
        bc.fn_pinfo(&bc.mypnum, &bc.nprocs);
    }
    bc.init_grid(nprow, npcol, 'R');

    if (!bc.in_grid()) {
        bc.finalize();
        return;
    }

    bool passed = true;
    int mismatches = 0;

    /* Generate random m-by-n matrix */
    unsigned seed_A = params.seed;
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, ctx.prec, &seed_A);

    /* Allocate receive buffer with sentinel protection.
       alloc_with_sentinel fills the entire buffer with sentinel bytes.
       GERV2D will write only the active m rows per column, leaving
       padding rows as sentinels for out-of-bounds write detection. */
    unsigned sentinel_seed = 0xBCAD0001;
    void *B = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);

    /* Send to self (0,0) -> (0,0) */
    int rdest = 0, cdest = 0;
    fn_send(&bc.ictxt, &m, &n, A, &lda, &rdest, &cdest);

    int rsrc = 0, csrc = 0;
    fn_recv(&bc.ictxt, &m, &n, B, &lda, &rsrc, &csrc);

    /* Verify: compare active region of A and B */
    for (int j = 0; j < n; ++j) {
        const char *a_col = static_cast<const char *>(A) +
            static_cast<std::size_t>(j) * lda * ctx.typesize;
        const char *b_col = static_cast<const char *>(B) +
            static_cast<std::size_t>(j) * lda * ctx.typesize;
        for (int i = 0; i < m; ++i) {
            if (std::memcmp(a_col + static_cast<std::size_t>(i) * ctx.typesize,
                            b_col + static_cast<std::size_t>(i) * ctx.typesize,
                            ctx.typesize) != 0) {
                ++mismatches;
            }
        }
    }

    /* Check sentinel bytes */
    SentinelResult sr = check_matrix_sentinels(B, m, n, lda, ctx.typesize, sentinel_seed);
    if (!sr.passed)
        passed = false;
    if (mismatches > 0)
        passed = false;

    bc.finalize();

    if (bc.mypnum == 0 || bc.nprocs <= 1) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "m=%d n=%d lda=%d nprocs=%d", m, n, lda, bc.nprocs);
        BlacsResult br = {passed, -1.0, mismatches};
        report_blacs_result("BLACS_GESD2D", params_str, br, format);
    }

    std::free(A);
    std::free(B);
}
