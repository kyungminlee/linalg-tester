/* gebrd.cpp -- LAPACK GEBRD accuracy tester (bidiagonal reduction) */
/* A = Q * B * P^T where B is bidiagonal */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Reconstruct Q from left Householder reflectors (stored below       */
/* diagonal of A with tauq).                                           */
/* For m >= n: reflectors are in columns 0..n-1, below the diagonal.  */
/*   v_i: v(j)=0 for j<i, v(i)=1, v(j)=A(j,i) for j>i               */
/* For m < n: reflectors are in columns 0..m-1, below first           */
/*   subdiagonal. v_i: v(j)=0 for j<i+1, v(i+1)=1,                   */
/*   v(j)=A(j,i) for j>i+1   (but only for i=0..m-2)                 */
/*   Actually GEBRD doc says: if m >= n, Q is m-by-n, tauq has n elts */
/*   if m < n, Q is m-by-m, tauq has m elements.                      */
/* ------------------------------------------------------------------ */

static void accumulate_Q_gebrd(MpfrMatrix &Q, const MpfrMatrix &A,
                                const MpfrMatrix &tauq, int m, int n)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    int mn = std::min(m, n);
    int k = (m >= n) ? n : (m - 1);  /* number of reflectors */
    if (k <= 0) {
        mpfr_mat_set_identity(Q);
        return;
    }

    mpfr_mat_set_identity(Q);
    MpfrScalar dot(prec), scale(prec);

    for (int i = k - 1; i >= 0; --i) {
        /* Reflector column index in A is i.
           For m >= n: v starts at row i (v(i)=1, v(j)=A(j,i) for j>i)
           For m < n:  v starts at row i+1 (v(i+1)=1, v(j)=A(j,i) for j>i+1) */
        int v_start = (m >= n) ? i : (i + 1);

        /* Apply H_i = I - tauq_i * v_i * v_i^T to Q from the left */
        for (int j = 0; j < Q.cols(); ++j) {
            /* dot = v_i^T * Q(:,j) */
            mpfr_set(dot.get(), Q.at(v_start, j), MPFR_RNDN);
            for (int r = v_start + 1; r < m; ++r)
                mpfr_fma(dot.get(), A.at(r, i), Q.at(r, j), dot.get(), MPFR_RNDN);

            mpfr_mul(scale.get(), tauq.at(i, 0), dot.get(), MPFR_RNDN);

            mpfr_sub(Q.at(v_start, j), Q.at(v_start, j), scale.get(), MPFR_RNDN);
            for (int r = v_start + 1; r < m; ++r) {
                MpfrScalar tmp(prec);
                mpfr_mul(tmp.get(), scale.get(), A.at(r, i), MPFR_RNDN);
                mpfr_sub(Q.at(r, j), Q.at(r, j), tmp.get(), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Reconstruct P^T from right Householder reflectors (stored above    */
/* diagonal of A with taup).                                           */
/* For m >= n: reflectors in rows 0..n-2, right of first superdiag.   */
/*   v_i: v(j)=0 for j<i+1, v(i+1)=1, v(j)=A(i,j) for j>i+1        */
/* For m < n: reflectors in rows 0..m-1, right of diagonal.           */
/*   v_i: v(j)=0 for j<i, v(i)=1, v(j)=A(i,j) for j>i               */
/* P = H_1 H_2 ... H_k, so P^T = H_k^T ... H_1^T                    */
/* Since H_i is symmetric (H_i = I - taup_i v_i v_i^T), P^T = H_k...H_1 */
/* ------------------------------------------------------------------ */

static void accumulate_Pt_gebrd(MpfrMatrix &Pt, const MpfrMatrix &A,
                                 const MpfrMatrix &taup, int m, int n)
{
    mpfr_prec_t prec = mpfr_get_prec(A.at(0, 0));
    int k = (m >= n) ? (n - 1) : m;  /* number of right reflectors */
    if (k <= 0) {
        mpfr_mat_set_identity(Pt);
        return;
    }

    /* P^T is n-by-n. Build P first, then transpose.
       P = H_0 H_1 ... H_{k-1}
       Build from right: start with I, apply H_{k-1}, ..., H_0 from left. */
    MpfrMatrix P(n, n, prec);
    mpfr_mat_set_identity(P);
    MpfrScalar dot(prec), scale(prec);

    for (int i = k - 1; i >= 0; --i) {
        int v_start = (m >= n) ? (i + 1) : i;

        for (int j = 0; j < n; ++j) {
            mpfr_set(dot.get(), P.at(v_start, j), MPFR_RNDN);
            for (int r = v_start + 1; r < n; ++r)
                mpfr_fma(dot.get(), A.at(i, r), P.at(r, j), dot.get(), MPFR_RNDN);

            mpfr_mul(scale.get(), taup.at(i, 0), dot.get(), MPFR_RNDN);

            mpfr_sub(P.at(v_start, j), P.at(v_start, j), scale.get(), MPFR_RNDN);
            for (int r = v_start + 1; r < n; ++r) {
                MpfrScalar tmp(prec);
                mpfr_mul(tmp.get(), scale.get(), A.at(i, r), MPFR_RNDN);
                mpfr_sub(P.at(r, j), P.at(r, j), tmp.get(), MPFR_RNDN);
            }
        }
    }

    /* P^T */
    mpfr_mat_transpose(Pt, P);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_gebrd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    int mn = std::min(m, n);

    unsigned seed_A = params.seed;

    /* Generate random m-by-n matrix A */
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    /* Save A_orig in MPFR */
    MpfrMatrix A_orig(m, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Allocate d, e, tauq, taup */
    void *d_arr = std::malloc(static_cast<std::size_t>(mn) * ctx.typesize);
    void *e_arr = std::malloc(static_cast<std::size_t>(std::max(mn - 1, 1)) * ctx.typesize);
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
    int lwork = query_lwork(work_query, ctx);
    std::free(work_query);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

    /* Call GEBRD */
    info = 0;
    fn(&m, &n, A, &lda, d_arr, e_arr, tauq, taup,
       work, &lwork, &info);

    /* Convert factored A, d, e, tauq, taup to MPFR */
    MpfrMatrix A_fact(m, n, prec);
    custom_to_mpfr_mat(A_fact, A, lda, ctx);

    MpfrMatrix d_mpfr(mn, 1, prec);
    MpfrMatrix tauq_mpfr(mn, 1, prec);
    MpfrMatrix taup_mpfr(mn, 1, prec);
    {
        const char *dp = static_cast<const char *>(d_arr);
        const char *tqp = static_cast<const char *>(tauq);
        const char *tpp = static_cast<const char *>(taup);
        for (int i = 0; i < mn; ++i) {
            ctx.to_mpfr(d_mpfr.at(i, 0), dp + static_cast<std::size_t>(i) * ctx.typesize);
            ctx.to_mpfr(tauq_mpfr.at(i, 0), tqp + static_cast<std::size_t>(i) * ctx.typesize);
            ctx.to_mpfr(taup_mpfr.at(i, 0), tpp + static_cast<std::size_t>(i) * ctx.typesize);
        }
    }

    int ne = (mn > 1) ? (mn - 1) : 0;
    MpfrMatrix e_mpfr(std::max(ne, 1), 1, prec);
    if (ne > 0) {
        const char *ep = static_cast<const char *>(e_arr);
        for (int i = 0; i < ne; ++i)
            ctx.to_mpfr(e_mpfr.at(i, 0), ep + static_cast<std::size_t>(i) * ctx.typesize);
    }

    /* Build bidiagonal matrix B (m-by-n) */
    MpfrMatrix B(m, n, prec);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            mpfr_set_d(B.at(i, j), 0.0, MPFR_RNDN);
    for (int i = 0; i < mn; ++i)
        mpfr_set(B.at(i, i), d_mpfr.at(i, 0), MPFR_RNDN);
    if (m >= n) {
        /* Superdiagonal */
        for (int i = 0; i < ne; ++i)
            mpfr_set(B.at(i, i + 1), e_mpfr.at(i, 0), MPFR_RNDN);
    } else {
        /* Subdiagonal */
        for (int i = 0; i < ne; ++i)
            mpfr_set(B.at(i + 1, i), e_mpfr.at(i, 0), MPFR_RNDN);
    }

    /* Reconstruct Q (m-by-m) and P^T (n-by-n) */
    MpfrMatrix Q(m, m, prec);
    accumulate_Q_gebrd(Q, A_fact, tauq_mpfr, m, n);

    MpfrMatrix Pt(n, n, prec);
    accumulate_Pt_gebrd(Pt, A_fact, taup_mpfr, m, n);

    /* Compute Q * B * P^T */
    MpfrMatrix QB(m, n, prec);
    mpfr_mat_mul_simple(QB, Q, B);

    MpfrMatrix QBPt(m, n, prec);
    mpfr_mat_mul_simple(QBPt, QB, Pt);

    /* Residual: ||A - Q*B*P^T||_1 / (||A||_1 * max(m,n) * eps) */
    MpfrMatrix Resid(m, n, prec);
    mpfr_mat_sub(Resid, A_orig, QBPt);
    double norm_resid = mpfr_mat_norm1(Resid);
    double norm_A = mpfr_mat_norm1(A_orig);

    int mn_max = std::max(m, n);
    double residual = (norm_A > 0.0) ? norm_resid / (norm_A * mn_max * eps) : 0.0;

    /* Orthogonality of Q */
    double ortho_Q = mpfr_orthogonality(Q, eps);

    /* Orthogonality of P (from P^T) -- P = (P^T)^T, so check P^T * P = I */
    /* Since P^T is already n-by-n: (P^T)^T * P^T should be I, same as
       checking columns of P^T are orthonormal, i.e., orthogonality(P^T) */
    MpfrMatrix P(n, n, prec);
    mpfr_mat_transpose(P, Pt);
    double ortho_P = mpfr_orthogonality(P, eps);

    double ortho = std::max(ortho_Q, ortho_P);

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = ortho;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    report_lapack_result("GEBRD", params_str, lr, format);

    std::free(A);
    std::free(d_arr);
    std::free(e_arr);
    std::free(tauq);
    std::free(taup);
    std::free(work);
}
