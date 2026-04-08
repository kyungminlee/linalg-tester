/* blacs_trbs2d.cpp -- BLACS triangular broadcast send/receive accuracy test */
/* Tests xTRBS2D (broadcast send) and xTRBR2D (broadcast receive) */

#include "../blacs.h"
#include "../blacs_common.h"
#include "../../core/generators.h"
#include "../../core/sentinel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

void test_blacs_trbs2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format)
{
    BlacsCtx bc;
    if (!bc.load(lib)) {
        BlacsResult br = {false, -1.0, 0};
        report_blacs_result("BLACS_TRBS2D", "error=symbols_not_found", br, format);
        return;
    }

    /* Load typed broadcast send symbol.
       sym is e.g. "dtrbs2d_". Signature includes scope, top, uplo, diag
       and their hidden character lengths. */
    auto *fn_bsend = reinterpret_cast<void (*)(
        const int *, const char *, const char *,
        const char *, const char *,
        const int *, const int *, const void *, const int *,
        std::size_t, std::size_t, std::size_t, std::size_t)>(load_sym(lib, sym));

    /* Derive receive symbol: replace "trbs2d" with "trbr2d" in sym */
    std::string recv_sym(sym);
    auto pos = recv_sym.find("trbs2d");
    if (pos != std::string::npos)
        recv_sym.replace(pos, 6, "trbr2d");
    auto *fn_brecv = reinterpret_cast<void (*)(
        const int *, const char *, const char *,
        const char *, const char *,
        const int *, const int *, void *, const int *,
        const int *, const int *,
        std::size_t, std::size_t, std::size_t, std::size_t)>(load_sym(lib, recv_sym.c_str()));

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;

    /* Initialize 1x1 grid */
    bc.init_grid(1, 1, 'R');
    if (!bc.in_grid()) {
        bc.finalize();
        return;
    }

    bool overall_passed = true;
    int overall_mismatches = 0;

    char scope = 'A';
    char top = ' ';  /* default topology */

    /* Loop over uplo combinations */
    const char uplo_vals[] = {'U', 'L'};
    const char diag_val = 'N';  /* representative diag for broadcast test */

    for (char uplo : uplo_vals) {
        /* Generate a triangular matrix */
        unsigned seed_A = params.seed;
        void *A = gen_triangular_array(std::min(m, n), uplo, diag_val,
                                        ctx.typesize, ctx.from_mpfr,
                                        ctx.prec, &seed_A);

        /* Copy into lda-by-n buffer */
        int k = std::min(m, n);
        std::size_t buf_size = static_cast<std::size_t>(lda) * n * ctx.typesize;
        void *A_buf = std::calloc(1, buf_size);
        for (int j = 0; j < k; ++j) {
            std::memcpy(
                static_cast<char *>(A_buf) +
                    static_cast<std::size_t>(j) * lda * ctx.typesize,
                static_cast<const char *>(A) +
                    static_cast<std::size_t>(j) * k * ctx.typesize,
                static_cast<std::size_t>(k) * ctx.typesize);
        }

        /* Save a copy for verification */
        void *A_copy = std::malloc(buf_size);
        std::memcpy(A_copy, A_buf, buf_size);

        if (bc.nprocs <= 1) {
            /* Single process: root broadcasts to self, verify buffer unchanged. */
            fn_bsend(&bc.ictxt, &scope, &top, &uplo, &diag_val,
                     &m, &n, A_buf, &lda,
                     (std::size_t)1, (std::size_t)1, (std::size_t)1, (std::size_t)1);

            /* Verify A_buf is unchanged after broadcast (triangle region) */
            int mismatches = 0;
            for (int j = 0; j < n; ++j) {
                const char *a_col = static_cast<const char *>(A_buf) +
                    static_cast<std::size_t>(j) * lda * ctx.typesize;
                const char *c_col = static_cast<const char *>(A_copy) +
                    static_cast<std::size_t>(j) * lda * ctx.typesize;
                for (int i = 0; i < m; ++i) {
                    bool in_triangle = (uplo == 'U') ? (i <= j) : (i >= j);
                    if (!in_triangle) continue;

                    if (std::memcmp(a_col + static_cast<std::size_t>(i) * ctx.typesize,
                                    c_col + static_cast<std::size_t>(i) * ctx.typesize,
                                    ctx.typesize) != 0) {
                        ++mismatches;
                    }
                }
            }
            if (mismatches > 0) {
                overall_passed = false;
                overall_mismatches += mismatches;
            }
        } else {
            /* Multi-process: root sends, non-root receives */
            if (bc.is_root()) {
                fn_bsend(&bc.ictxt, &scope, &top, &uplo, &diag_val,
                         &m, &n, A_buf, &lda,
                         (std::size_t)1, (std::size_t)1, (std::size_t)1, (std::size_t)1);
            } else {
                int rsrc = 0, csrc = 0;
                fn_brecv(&bc.ictxt, &scope, &top, &uplo, &diag_val,
                         &m, &n, A_buf, &lda, &rsrc, &csrc,
                         (std::size_t)1, (std::size_t)1, (std::size_t)1, (std::size_t)1);
            }
        }

        std::free(A);
        std::free(A_buf);
        std::free(A_copy);
    }

    bc.finalize();

    if (bc.mypnum == 0 || bc.nprocs <= 1) {
        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "m=%d n=%d scope=A nprocs=%d", m, n, bc.nprocs);
        BlacsResult br = {overall_passed, -1.0, overall_mismatches};
        report_blacs_result("BLACS_TRBS2D", params_str, br, format);
    }
}
