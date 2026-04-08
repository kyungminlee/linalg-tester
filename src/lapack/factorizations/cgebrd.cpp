/* cgebrd.cpp -- LAPACK complex GEBRD accuracy tester (bidiagonal reduction) */
/* A = Q * B * P^H where B is real bidiagonal, Q and P are complex unitary */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"  // for get_eps

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Reconstruct Q from left Householder reflectors (stored below       */
/* diagonal of A with complex tauq).                                   */
/* H_i = I - tauq_i * v_i * v_i^H                                     */
/* For m >= n: reflectors in columns 0..n-1, below diagonal.          */
/*   v_i: v(j)=0 for j<i, v(i)=1, v(j)=A(j,i) for j>i               */
/* For m < n: reflectors in columns 0..m-2, below first subdiagonal.  */
/*   v_i: v(j)=0 for j<i+1, v(i+1)=1, v(j)=A(j,i) for j>i+1        */
/* ------------------------------------------------------------------ */

static void accumulate_Q_cgebrd(MpfrComplexMatrix &Q, const MpfrComplexMatrix &A,
                                 const MpfrComplexMatrix &tauq, int m, int n)
{
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    int k = (m >= n) ? n : (m - 1);  /* number of reflectors */
    if (k <= 0) {
        mpfr_complex_mat_set_identity(Q);
        return;
    }

    mpfr_complex_mat_set_identity(Q);
    MpfrComplexScalar dot(prec), scale(prec), tmp(prec);

    for (int i = k - 1; i >= 0; --i) {
        int v_start = (m >= n) ? i : (i + 1);

        /* Apply H_i = I - tauq_i * v_i * v_i^H to Q from the left */
        for (int j = 0; j < Q.cols(); ++j) {
            /* dot = conj(v_i)^T * Q(:,j) */
            /* v(v_start)=1, so conj(1)=1: dot starts with Q(v_start,j) */
            mpfr_set(dot.re(), Q.re(v_start, j), MPFR_RNDN);
            mpfr_set(dot.im(), Q.im(v_start, j), MPFR_RNDN);
            for (int r = v_start + 1; r < m; ++r) {
                /* dot += conj(A(r,i)) * Q(r,j) */
                MpfrComplexScalar cv(prec);
                mpfr_complex_conj(cv.re(), cv.im(),
                                  A.re(r, i), A.im(r, i),
                                  MPFR_RNDN);
                mpfr_complex_fma(dot.re(), dot.im(),
                                 cv.re(), cv.im(),
                                 Q.re(r, j), Q.im(r, j),
                                 MPFR_RNDN);
            }

            /* scale = tauq_i * dot */
            mpfr_complex_mul(scale.re(), scale.im(),
                             tauq.re(i, 0), tauq.im(i, 0),
                             dot.re(), dot.im(),
                             MPFR_RNDN);

            /* Q(v_start,j) -= scale * 1 */
            mpfr_sub(Q.re(v_start, j), Q.re(v_start, j), scale.re(), MPFR_RNDN);
            mpfr_sub(Q.im(v_start, j), Q.im(v_start, j), scale.im(), MPFR_RNDN);

            for (int r = v_start + 1; r < m; ++r) {
                /* Q(r,j) -= scale * A(r,i) */
                mpfr_complex_mul(tmp.re(), tmp.im(),
                                 scale.re(), scale.im(),
                                 A.re(r, i), A.im(r, i),
                                 MPFR_RNDN);
                mpfr_sub(Q.re(r, j), Q.re(r, j), tmp.re(), MPFR_RNDN);
                mpfr_sub(Q.im(r, j), Q.im(r, j), tmp.im(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Reconstruct P^H from right Householder reflectors (stored in rows  */
/* of A with complex taup).                                            */
/* H_i = I - conj(taup_i) * v_i * v_i^H  (complex reflectors)        */
/* P = H_0 H_1 ... H_{k-1}, so P^H = H_{k-1}^H ... H_0^H            */
/* Since H_i = I - conj(taup_i)*v*v^H, H_i^H = I - taup_i*v*v^H      */
/* We build P first (accumulate from right), then take adjoint.        */
/*                                                                     */
/* For m >= n: reflectors in rows 0..n-2, right of first superdiag.   */
/*   v_i: v(j)=0 for j<i+1, v(i+1)=1, v(j)=A(i,j) for j>i+1        */
/* For m < n: reflectors in rows 0..m-1, right of diagonal.           */
/*   v_i: v(j)=0 for j<i, v(i)=1, v(j)=A(i,j) for j>i               */
/* ------------------------------------------------------------------ */

static void accumulate_Ph_cgebrd(MpfrComplexMatrix &Ph, const MpfrComplexMatrix &A,
                                  const MpfrComplexMatrix &taup, int m, int n)
{
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    int k = (m >= n) ? (n - 1) : m;  /* number of right reflectors */
    if (k <= 0) {
        mpfr_complex_mat_set_identity(Ph);
        return;
    }

    /* Build P = H_0 H_1 ... H_{k-1} by applying from H_{k-1} backward.
       H_i = I - conj(taup_i) * v_i * v_i^H */
    MpfrComplexMatrix P(n, n, prec);
    mpfr_complex_mat_set_identity(P);
    MpfrComplexScalar dot(prec), scale(prec), tmp(prec), ctau(prec);

    /* LAPACK ZGEBRD: right reflectors G_i = I - taup_i * u * u^H.
       P = G_0 * G_1 * ... * G_{k-1}.
       The reflector vector u is stored in A rows. LAPACK ZGEBRD applies
       ZLACGV to conjugate the row before ZLARFG, then conjugates back.
       The NET effect is that A(i,r) stores conj(u(r)).
       So u(r) = conj(A(i,r)).
       To apply G_i = I - taup * u * u^H:
       Inner product: u^H * P(:,j) = sum conj(u(r))*P(r,j) = sum A(i,r)*P(r,j)
       Application: P(r,j) -= taup * u(r) * dot = taup * conj(A(i,r)) * dot */
    for (int i = k - 1; i >= 0; --i) {
        int v_start = (m >= n) ? (i + 1) : i;

        for (int j = 0; j < n; ++j) {
            /* dot = u^H * P(:,j) = P(v_start,j) + sum A(i,r)*P(r,j) */
            mpfr_set(dot.re(), P.re(v_start, j), MPFR_RNDN);
            mpfr_set(dot.im(), P.im(v_start, j), MPFR_RNDN);
            for (int r = v_start + 1; r < n; ++r) {
                mpfr_complex_fma(dot.re(), dot.im(),
                                 A.re(i, r), A.im(i, r),
                                 P.re(r, j), P.im(r, j),
                                 MPFR_RNDN);
            }

            /* scale = taup_i * dot (no conj!) */
            mpfr_complex_mul(scale.re(), scale.im(),
                             taup.re(i, 0), taup.im(i, 0),
                             dot.re(), dot.im(),
                             MPFR_RNDN);

            /* P(v_start,j) -= scale */
            mpfr_sub(P.re(v_start, j), P.re(v_start, j), scale.re(), MPFR_RNDN);
            mpfr_sub(P.im(v_start, j), P.im(v_start, j), scale.im(), MPFR_RNDN);

            /* P(r,j) -= scale * u(r) = scale * conj(A(i,r)) */
            for (int r = v_start + 1; r < n; ++r) {
                MpfrComplexScalar cv(prec);
                mpfr_complex_conj(cv.re(), cv.im(),
                                  A.re(i, r), A.im(i, r),
                                  MPFR_RNDN);
                mpfr_complex_mul(tmp.re(), tmp.im(),
                                 scale.re(), scale.im(),
                                 cv.re(), cv.im(),
                                 MPFR_RNDN);
                mpfr_sub(P.re(r, j), P.re(r, j), tmp.re(), MPFR_RNDN);
                mpfr_sub(P.im(r, j), P.im(r, j), tmp.im(), MPFR_RNDN);
            }
        }
    }

    /* P^H */
    mpfr_complex_mat_adjoint(Ph, P);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_cgebrd(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    int mn = std::min(m, n);

    unsigned seed_A = params.seed;

    /* Generate random complex m-by-n matrix A */
    void *A = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                       ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);

    /* Save A_orig in MPFR */
    MpfrComplexMatrix A_orig(m, n, prec);
    custom_to_mpfr_complex_mat(A_orig, A, lda, ctx);

    /* For complex GEBRD: d and e are REAL, tauq and taup are COMPLEX.
       Real typesize is half of complex typesize. */
    std::size_t real_typesize = ctx.typesize / 2;

    /* Allocate d, e (real), tauq, taup (complex) */
    void *d_arr = std::malloc(static_cast<std::size_t>(mn) * real_typesize);
    void *e_arr = std::malloc(static_cast<std::size_t>(std::max(mn - 1, 1)) * real_typesize);
    void *tauq  = std::malloc(static_cast<std::size_t>(mn) * ctx.typesize);
    void *taup  = std::malloc(static_cast<std::size_t>(mn) * ctx.typesize);

    /* Workspace query */
    int lwork_query = -1;
    int info = 0;
    void *work_query = std::malloc(ctx.typesize);
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, void *, const int *,
        void *, void *, void *, void *,
        void *, const int *, int *)>(load_sym(lib, sym));
    fn(&m, &n, A, &lda, d_arr, e_arr, tauq, taup,
       work_query, &lwork_query, &info);
    int lwork = query_lwork_complex(work_query, ctx);
    std::free(work_query);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

    /* Call GEBRD */
    info = 0;
    fn(&m, &n, A, &lda, d_arr, e_arr, tauq, taup,
       work, &lwork, &info);

    /* Convert factored A (complex) to MPFR */
    MpfrComplexMatrix A_fact(m, n, prec);
    custom_to_mpfr_complex_mat(A_fact, A, lda, ctx);

    /* Convert d (real) to MPFR */
    MpfrMatrix d_mpfr(mn, 1, prec);
    {
        const char *dp = static_cast<const char *>(d_arr);
        for (int i = 0; i < mn; ++i)
            ctx.to_mpfr(d_mpfr.at(i, 0), dp + static_cast<std::size_t>(i) * real_typesize);
    }

    /* Convert e (real) to MPFR */
    int ne = (mn > 1) ? (mn - 1) : 0;
    MpfrMatrix e_mpfr(std::max(ne, 1), 1, prec);
    if (ne > 0) {
        const char *ep = static_cast<const char *>(e_arr);
        for (int i = 0; i < ne; ++i)
            ctx.to_mpfr(e_mpfr.at(i, 0), ep + static_cast<std::size_t>(i) * real_typesize);
    }

    /* Convert tauq, taup (complex) to MPFR */
    MpfrComplexMatrix tauq_mpfr(mn, 1, prec);
    MpfrComplexMatrix taup_mpfr(mn, 1, prec);
    {
        const char *tqp = static_cast<const char *>(tauq);
        const char *tpp = static_cast<const char *>(taup);
        for (int i = 0; i < mn; ++i) {
            ctx.to_mpfr_complex(tauq_mpfr.re(i, 0), tauq_mpfr.im(i, 0),
                                tqp + static_cast<std::size_t>(i) * ctx.typesize);
            ctx.to_mpfr_complex(taup_mpfr.re(i, 0), taup_mpfr.im(i, 0),
                                tpp + static_cast<std::size_t>(i) * ctx.typesize);
        }
    }

    /* Build bidiagonal matrix B as a complex matrix with real d/e on diagonals
       and zero imaginary parts. B is m-by-n. */
    MpfrComplexMatrix B(m, n, prec);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(B.re(i, j), 0.0, MPFR_RNDN);
            mpfr_set_d(B.im(i, j), 0.0, MPFR_RNDN);
        }
    for (int i = 0; i < mn; ++i) {
        mpfr_set(B.re(i, i), d_mpfr.at(i, 0), MPFR_RNDN);
        /* im already 0 */
    }
    if (m >= n) {
        /* Superdiagonal */
        for (int i = 0; i < ne; ++i)
            mpfr_set(B.re(i, i + 1), e_mpfr.at(i, 0), MPFR_RNDN);
    } else {
        /* Subdiagonal */
        for (int i = 0; i < ne; ++i)
            mpfr_set(B.re(i + 1, i), e_mpfr.at(i, 0), MPFR_RNDN);
    }

    /* Reconstruct Q (m-by-m) and P^H (n-by-n) */
    MpfrComplexMatrix Q(m, m, prec);
    accumulate_Q_cgebrd(Q, A_fact, tauq_mpfr, m, n);

    MpfrComplexMatrix Ph(n, n, prec);
    accumulate_Ph_cgebrd(Ph, A_fact, taup_mpfr, m, n);

    /* Compute Q * B * P^H */
    MpfrComplexMatrix QB(m, n, prec);
    mpfr_complex_mat_mul_simple(QB, Q, B);

    MpfrComplexMatrix QBPh(m, n, prec);
    mpfr_complex_mat_mul_simple(QBPh, QB, Ph);

    /* Residual: ||A - Q*B*P^H||_1 / (||A||_1 * max(m,n) * eps) */
    MpfrComplexMatrix Resid(m, n, prec);
    mpfr_complex_mat_sub(Resid, A_orig, QBPh);
    double norm_resid = mpfr_complex_mat_norm1(Resid);
    double norm_A = mpfr_complex_mat_norm1(A_orig);

    int mn_max = std::max(m, n);
    double residual = (norm_A > 0.0) ? norm_resid / (norm_A * mn_max * eps) : 0.0;

    /* Orthogonality of Q */
    double ortho_Q = mpfr_complex_orthogonality(Q, eps);

    /* Orthogonality of P (from P^H): P = (P^H)^H */
    MpfrComplexMatrix P(n, n, prec);
    mpfr_complex_mat_adjoint(P, Ph);
    double ortho_P = mpfr_complex_orthogonality(P, eps);

    double ortho = std::max(ortho_Q, ortho_P);

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = ortho;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    report_lapack_result("CGEBRD", params_str, lr, format);

    std::free(A);
    std::free(d_arr);
    std::free(e_arr);
    std::free(tauq);
    std::free(taup);
    std::free(work);
}
