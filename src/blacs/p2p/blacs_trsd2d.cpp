/* blacs_trsd2d.cpp -- BLACS triangular point-to-point send/receive accuracy test */
/* Tests xTRSD2D (send) and xTRRV2D (receive) for data integrity */

#include "../blacs.h"
#include "../blacs_common.h"
#include "../../core/generators.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

void test_blacs_trsd2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format)
{
    BlacsCtx bc;
    if (!bc.load(lib)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("BLACS_TRSD2D", "error=symbols_not_found", br, format);
        return;
    }

    /* Load typed send symbol.
       sym is e.g. "dtrsd2d_". Function signature includes uplo, diag and
       their hidden character lengths. */
    auto *fn_send = reinterpret_cast<void (*)(
        const int *, const char *, const char *,
        const int *, const int *, const void *, const int *,
        const int *, const int *,
        std::size_t, std::size_t)>(load_sym(lib, sym));

    /* Derive receive symbol: replace "trsd2d" with "trrv2d" in sym */
    std::string recv_sym(sym);
    auto pos = recv_sym.find("trsd2d");
    if (pos != std::string::npos)
        recv_sym.replace(pos, 6, "trrv2d");
    auto *fn_recv = reinterpret_cast<void (*)(
        const int *, const char *, const char *,
        const int *, const int *, void *, const int *,
        const int *, const int *,
        std::size_t, std::size_t)>(load_sym(lib, recv_sym.c_str()));

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;

    /* Initialize 1x1 grid for single process */
    int nprow = 1, npcol = 1;
    if (bc.nprocs <= 0) {
        bc.fn_pinfo(&bc.mypnum, &bc.nprocs);
    }
    bc.init_grid(nprow, npcol, 'R');

    if (!bc.in_grid()) {
        bc.finalize();
        return;
    }

    bool overall_passed = true;
    int overall_mismatches = 0;

    /* Loop over uplo and diag combinations */
    const char uplo_vals[] = {'U', 'L'};
    const char diag_vals[] = {'N', 'U'};

    for (char uplo : uplo_vals) {
        for (char diag : diag_vals) {
            /* Generate a triangular matrix */
            unsigned seed_A = params.seed;
            void *A = gen_triangular_array(std::min(m, n), uplo, diag,
                                            ctx.typesize, ctx.from_mpfr,
                                            ctx.prec, &seed_A);

            /* For triangular send, the matrix is m-by-n stored in lda.
               gen_triangular_array generates a k-by-k matrix (k = min(m,n)).
               We need an lda-by-n buffer. Allocate and copy. */
            int k = std::min(m, n);
            std::size_t buf_size = static_cast<std::size_t>(lda) * n * ctx.typesize;
            void *A_buf = std::calloc(1, buf_size);
            /* Copy generated k-by-k triangular into the lda-by-n buffer */
            for (int j = 0; j < k; ++j) {
                std::memcpy(
                    static_cast<char *>(A_buf) +
                        static_cast<std::size_t>(j) * lda * ctx.typesize,
                    static_cast<const char *>(A) +
                        static_cast<std::size_t>(j) * k * ctx.typesize,
                    static_cast<std::size_t>(k) * ctx.typesize);
            }

            /* Allocate receive buffer with sentinel protection */
            unsigned sentinel_seed = 0xBCAD0002;
            void *B_buf = alloc_with_sentinel(lda * n, ctx.typesize, sentinel_seed);

            /* Send to self (0,0) -> (0,0) */
            int rdest = 0, cdest = 0;
            fn_send(&bc.ictxt, &uplo, &diag, &m, &n, A_buf, &lda,
                    &rdest, &cdest, (std::size_t)1, (std::size_t)1);

            int rsrc = 0, csrc = 0;
            fn_recv(&bc.ictxt, &uplo, &diag, &m, &n, B_buf, &lda,
                    &rsrc, &csrc, (std::size_t)1, (std::size_t)1);

            /* Verify: compare only the active triangle region */
            int mismatches = 0;
            for (int j = 0; j < n; ++j) {
                const char *a_col = static_cast<const char *>(A_buf) +
                    static_cast<std::size_t>(j) * lda * ctx.typesize;
                const char *b_col = static_cast<const char *>(B_buf) +
                    static_cast<std::size_t>(j) * lda * ctx.typesize;
                for (int i = 0; i < m; ++i) {
                    bool in_triangle = false;
                    if (uplo == 'U') {
                        in_triangle = (i <= j);
                    } else {
                        in_triangle = (i >= j);
                    }
                    if (!in_triangle) continue;
                    /* When diag='U', the diagonal is not transmitted */
                    if (diag == 'U' && i == j) continue;

                    if (std::memcmp(a_col + static_cast<std::size_t>(i) * ctx.typesize,
                                    b_col + static_cast<std::size_t>(i) * ctx.typesize,
                                    ctx.typesize) != 0) {
                        ++mismatches;
                    }
                }
            }

            if (mismatches > 0) {
                overall_passed = false;
                overall_mismatches += mismatches;
            }

            std::free(A);
            std::free(A_buf);
            std::free(B_buf);
        }
    }

    bc.finalize();

    if (bc.mypnum == 0 || bc.nprocs <= 1) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "m=%d n=%d lda=%d nprocs=%d", m, n, lda, bc.nprocs);
        BlacsResult br = {overall_passed, -1.0, overall_mismatches};
        report_blacs_result("BLACS_TRSD2D", params_str, br, format);
    }
}
