/* potri.cpp -- LAPACK POTRI accuracy tester (inverse from Cholesky) */

#include "../auxiliary.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*potrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t uplo_len);

extern "C" typedef void (*potri_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t uplo_len);

void test_potri(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    std::string sym_str(sym);
    std::string routine = "potri";
    std::size_t pos = sym_str.find(routine);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine.size());
    std::string potrf_sym = prefix + "potrf" + suffix;

    auto *potrf_fn = reinterpret_cast<potrf_fn_t>(load_sym(lib, potrf_sym.c_str()));
    auto *potri_fn = reinterpret_cast<potri_fn_t>(load_sym(lib, sym));

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        void *A_spd = gen_positive_definite_array(n, ctx.typesize, ctx.from_mpfr,
                                                   ctx.to_mpfr, prec, &seed_A);
        void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        if (lda == n) {
            std::memcpy(A_buf, A_spd, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(A_buf, 0, static_cast<std::size_t>(lda) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_spd) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        MpfrMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_mat(A_mpfr, A_buf, lda, ctx);

        /* POTRF */
        int info = 0;
        potrf_fn(&uplo, &n, A_buf, &lda, &info, (std::size_t)1);
        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c POTRF failed", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("POTRI", ps, lr, format);
            std::free(A_buf); std::free(A_spd); continue;
        }

        /* POTRI */
        potri_fn(&uplo, &n, A_buf, &lda, &info, (std::size_t)1);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("POTRI", ps, lr, format);
        } else {
            /* POTRI returns the inverse in the specified triangle.
               Symmetrize to get the full inverse. */
            MpfrMatrix Ainv_mpfr(n, n, prec);
            custom_to_mpfr_mat(Ainv_mpfr, A_buf, lda, ctx);
            /* Mirror the triangle */
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) {
                    if (uplo == 'U')
                        mpfr_set(Ainv_mpfr.at(j, i), Ainv_mpfr.at(i, j), MPFR_RNDN);
                    else
                        mpfr_set(Ainv_mpfr.at(i, j), Ainv_mpfr.at(j, i), MPFR_RNDN);
                }

            MpfrMatrix prod(n, n, prec);
            mpfr_mat_mul_simple(prod, A_mpfr, Ainv_mpfr);

            MpfrMatrix I_mat(n, n, prec);
            mpfr_mat_set_identity(I_mat);

            MpfrMatrix R(n, n, prec);
            mpfr_mat_sub(R, prod, I_mat);

            double norm_R = mpfr_mat_norm1(R);
            double norm_A = mpfr_mat_norm1(A_mpfr);
            double norm_Ainv = mpfr_mat_norm1(Ainv_mpfr);

            double residual = 0.0;
            if (norm_A > 0.0 && norm_Ainv > 0.0)
                residual = norm_R / (norm_A * norm_Ainv * n * eps);

            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("POTRI", ps, lr, format);
        }

        std::free(A_buf); std::free(A_spd);
    }
}
