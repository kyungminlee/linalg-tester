/* cpotri.cpp -- LAPACK CPOTRI/ZPOTRI accuracy tester (complex inverse from Cholesky) */

#include "../auxiliary.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

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

void test_cpotri(const TesterCtx &ctx, void *lib, const char *sym,
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

        void *A_hpd = gen_hermitian_positive_definite_array(n, ctx.typesize,
                          ctx.from_mpfr_complex, ctx.to_mpfr_complex, prec, &seed_A);

        void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        if (lda == n) {
            std::memcpy(A_buf, A_hpd, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(A_buf, 0, static_cast<std::size_t>(lda) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_hpd) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        MpfrComplexMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_complex_mat(A_mpfr, A_buf, lda, ctx);

        /* POTRF */
        int info = 0;
        potrf_fn(&uplo, &n, A_buf, &lda, &info, (std::size_t)1);
        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c POTRF failed", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("CPOTRI", ps, lr, format);
            std::free(A_buf); std::free(A_hpd); continue;
        }

        /* POTRI */
        potri_fn(&uplo, &n, A_buf, &lda, &info, (std::size_t)1);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("CPOTRI", ps, lr, format);
        } else {
            /* POTRI returns the inverse in the specified triangle.
               Hermitianize to get the full inverse: A_inv(i,j) = conj(A_inv(j,i)) */
            MpfrComplexMatrix Ainv_mpfr(n, n, prec);
            custom_to_mpfr_complex_mat(Ainv_mpfr, A_buf, lda, ctx);
            /* Mirror the triangle with conjugation */
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) {
                    if (uplo == 'U') {
                        /* Upper stored: copy conj(upper) to lower */
                        mpfr_complex_conj(Ainv_mpfr.re(j, i), Ainv_mpfr.im(j, i),
                                          Ainv_mpfr.re(i, j), Ainv_mpfr.im(i, j),
                                          MPFR_RNDN);
                    } else {
                        /* Lower stored: copy conj(lower) to upper */
                        mpfr_complex_conj(Ainv_mpfr.re(i, j), Ainv_mpfr.im(i, j),
                                          Ainv_mpfr.re(j, i), Ainv_mpfr.im(j, i),
                                          MPFR_RNDN);
                    }
                }

            MpfrComplexMatrix prod(n, n, prec);
            mpfr_complex_mat_mul_simple(prod, A_mpfr, Ainv_mpfr);

            MpfrComplexMatrix I_mat(n, n, prec);
            mpfr_complex_mat_set_identity(I_mat);

            MpfrComplexMatrix R(n, n, prec);
            mpfr_complex_mat_sub(R, prod, I_mat);

            double norm_R = mpfr_complex_mat_norm1(R);
            double norm_A = mpfr_complex_mat_norm1(A_mpfr);
            double norm_Ainv = mpfr_complex_mat_norm1(Ainv_mpfr);

            double residual = 0.0;
            if (norm_A > 0.0 && norm_Ainv > 0.0)
                residual = norm_R / (norm_A * norm_Ainv * n * eps);

            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("CPOTRI", ps, lr, format);
        }

        std::free(A_buf); std::free(A_hpd);
    }
}
