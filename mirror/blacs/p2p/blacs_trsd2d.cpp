/* blacs_trsd2d.cpp -- Mirror tester for BLACS triangular send/receive */
/* Compares xTRSD2D/xTRRV2D across two library implementations.
   On single process: send from (0,0) to (0,0), compare received data.
   Loops over uplo={'U','L'}, diag={'N','U'}. */

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

/* Fortran ABI for xTRSD2D (send) */
extern "C" typedef void (*trsd2d_fn_t)(
    const int *ictxt, const char *uplo, const char *diag,
    const int *m, const int *n, const void *a, const int *lda,
    const int *rdest, const int *cdest,
    std::size_t uplo_len, std::size_t diag_len
);

/* Fortran ABI for xTRRV2D (receive) */
extern "C" typedef void (*trrv2d_fn_t)(
    const int *ictxt, const char *uplo, const char *diag,
    const int *m, const int *n, void *a, const int *lda,
    const int *rsrc, const int *csrc,
    std::size_t uplo_len, std::size_t diag_len
);

void mirror_test_blacs_trsd2d(const MirrorSide &a, const MirrorSide &b,
                               const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        for (char diag : {'N', 'U'}) {
            unsigned seed_A = params.seed;

            /* Generate canonical MPFR triangular matrix */
            MpfrMatrix A_mpfr(m, n, prec);
            gen_mpfr_triangular_matrix(A_mpfr, uplo, diag, prec, &seed_A);

            auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
                /* Load BLACS context symbols */
                BlacsCtx bc;
                if (!bc.load(side.lib)) {
                    std::fprintf(stderr, "mirror_blacs_trsd2d: failed to load BLACS symbols from side %s\n",
                                 side.label.c_str());
                    return;
                }

                /* Load send symbol */
                auto *fn_send = reinterpret_cast<trsd2d_fn_t>(
                    load_sym(side.lib, side.sym.c_str()));

                /* Derive receive symbol: replace "trsd2d" with "trrv2d" */
                std::string recv_sym(side.sym);
                auto pos = recv_sym.find("trsd2d");
                if (pos != std::string::npos)
                    recv_sym.replace(pos, 6, "trrv2d");
                auto *fn_recv = reinterpret_cast<trrv2d_fn_t>(
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

                /* Send to self (0,0) -> (0,0) */
                int rdest = 0, cdest = 0;
                fn_send(&bc.ictxt, &uplo, &diag, &m, &n, native_A, &lda,
                        &rdest, &cdest, (std::size_t)1, (std::size_t)1);

                int rsrc = 0, csrc = 0;
                fn_recv(&bc.ictxt, &uplo, &diag, &m, &n, native_B, &lda,
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
                          "uplo=%c diag=%c m=%d n=%d lda=%d", uplo, diag, m, n, lda);
            mirror_report_result("BLACS_TRSD2D", params_str, err,
                                  nullptr, nullptr, config);
        }
    }
}
