/* blacs_gebs2d.cpp -- Mirror tester for BLACS general broadcast */
/* Compares xGEBS2D/xGEBR2D across two library implementations.
   On single process: root broadcasts to self, compare data. */

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

/* Fortran ABI for xGEBS2D (broadcast send) */
extern "C" typedef void (*gebs2d_fn_t)(
    const int *ictxt, const char *scope, const char *top,
    const int *m, const int *n, const void *a, const int *lda,
    std::size_t scope_len, std::size_t top_len
);

/* Fortran ABI for xGEBR2D (broadcast receive) */
extern "C" typedef void (*gebr2d_fn_t)(
    const int *ictxt, const char *scope, const char *top,
    const int *m, const int *n, void *a, const int *lda,
    const int *rsrc, const int *csrc,
    std::size_t scope_len, std::size_t top_len
);

void mirror_test_blacs_gebs2d(const MirrorSide &a, const MirrorSide &b,
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

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        /* Load BLACS context symbols */
        BlacsCtx bc;
        if (!bc.load(side.lib)) {
            std::fprintf(stderr, "mirror_blacs_gebs2d: failed to load BLACS symbols from side %s\n",
                         side.label.c_str());
            return;
        }

        /* Load broadcast send symbol */
        auto *fn_bsend = reinterpret_cast<gebs2d_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        /* Derive receive symbol: replace "gebs2d" with "gebr2d" */
        std::string recv_sym(side.sym);
        auto pos = recv_sym.find("gebs2d");
        if (pos != std::string::npos)
            recv_sym.replace(pos, 6, "gebr2d");
        auto *fn_brecv = reinterpret_cast<gebr2d_fn_t>(
            load_sym(side.lib, recv_sym.c_str()));

        /* Initialize 1x1 grid */
        bc.init_grid(1, 1, 'R');
        if (!bc.in_grid()) {
            bc.finalize();
            return;
        }

        /* Materialize MPFR data to native format */
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);

        /* Allocate receive buffer (zeroed) */
        std::size_t buf_size = static_cast<std::size_t>(lda) * n * side.ctx.typesize;
        void *native_B = std::calloc(1, buf_size);

        /* Root broadcasts */
        fn_bsend(&bc.ictxt, &scope, &top, &m, &n, native_A, &lda,
                  (std::size_t)1, (std::size_t)1);

        /* Root receives from self (single process) */
        int rsrc = 0, csrc = 0;
        fn_brecv(&bc.ictxt, &scope, &top, &m, &n, native_B, &lda,
                  &rsrc, &csrc, (std::size_t)1, (std::size_t)1);

        /* Convert received data back to MPFR */
        custom_to_mpfr_mat(result, native_B, lda, side.ctx);

        bc.finalize();
        std::free(native_A);
        std::free(native_B);
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
    mirror_report_result("BLACS_GEBS2D", params_str, err, nullptr, nullptr, config);
}
