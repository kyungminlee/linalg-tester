#include "error_metrics.h"
#include "mpfr_complex.h"

#include <cstdlib>

/* ------------------------------------------------------------------ */
/* Internal helper: accumulate error for a single element               */
/* ------------------------------------------------------------------ */

static void accumulate_error(
    MpfrScalar &val, MpfrScalar &diff, MpfrScalar &absdiff,
    MpfrScalar &absref, MpfrScalar &ratio,
    MpfrScalar &sum_sq_diff, MpfrScalar &sum_sq_ref, MpfrScalar &max_rel,
    const mpfr_t &ref_elem, const void *out_elem,
    const TesterCtx &ctx,
    MpfrScalar &max_abs_at_zero, int &nan_inf_mismatches)
{
    ctx.to_mpfr(val.get(), out_elem);

    /* --- NaN/Inf handling --- */
    int ref_nan = mpfr_nan_p(ref_elem);
    int ref_inf = mpfr_inf_p(ref_elem);
    int val_nan = mpfr_nan_p(val.get());
    int val_inf = mpfr_inf_p(val.get());

    if (ref_nan) {
        if (!val_nan)
            ++nan_inf_mismatches;
        return; /* skip ratio computation */
    }
    if (ref_inf) {
        /* Check same sign of infinity */
        if (!val_inf || mpfr_sgn(ref_elem) != mpfr_sgn(val.get()))
            ++nan_inf_mismatches;
        return;
    }
    /* ref is finite */
    if (val_nan || val_inf) {
        ++nan_inf_mismatches;
        return;
    }

    /* --- Normal finite path --- */
    mpfr_sub(diff.get(), val.get(), ref_elem, MPFR_RNDN);
    mpfr_abs(absdiff.get(), diff.get(), MPFR_RNDN);
    mpfr_abs(absref.get(), ref_elem, MPFR_RNDN);

    mpfr_fma(sum_sq_diff.get(), diff.get(), diff.get(),
             sum_sq_diff.get(), MPFR_RNDN);
    mpfr_fma(sum_sq_ref.get(), ref_elem, ref_elem,
             sum_sq_ref.get(), MPFR_RNDN);

    if (mpfr_zero_p(absref.get())) {
        /* ref is zero: track max absolute error at zero positions */
        if (!mpfr_zero_p(absdiff.get())) {
            if (mpfr_sgn(max_abs_at_zero.get()) < 0 ||
                mpfr_cmp(absdiff.get(), max_abs_at_zero.get()) > 0)
                mpfr_set(max_abs_at_zero.get(), absdiff.get(), MPFR_RNDN);
        } else {
            /* Both zero, but mark that we've seen a zero ref */
            if (mpfr_sgn(max_abs_at_zero.get()) < 0)
                mpfr_set_d(max_abs_at_zero.get(), 0.0, MPFR_RNDN);
        }
        return;
    }
    mpfr_div(ratio.get(), absdiff.get(), absref.get(), MPFR_RNDN);
    if (mpfr_cmp(ratio.get(), max_rel.get()) > 0)
        mpfr_set(max_rel.get(), ratio.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Internal helper: finalize ErrorResult from accumulators              */
/* ------------------------------------------------------------------ */

static ErrorResult finalize_error(MpfrScalar &sum_sq_diff,
                                   MpfrScalar &sum_sq_ref,
                                   MpfrScalar &max_rel,
                                   MpfrScalar &ratio,
                                   MpfrScalar &max_abs_at_zero,
                                   int nan_inf_mismatches)
{
    ErrorResult result;
    result.max_relative = mpfr_get_d(max_rel.get(), MPFR_RNDN);

    if (mpfr_zero_p(sum_sq_ref.get())) {
        result.normwise_relative = 0.0;
    } else {
        mpfr_sqrt(sum_sq_diff.get(), sum_sq_diff.get(), MPFR_RNDN);
        mpfr_sqrt(sum_sq_ref.get(), sum_sq_ref.get(), MPFR_RNDN);
        mpfr_div(ratio.get(), sum_sq_diff.get(), sum_sq_ref.get(), MPFR_RNDN);
        result.normwise_relative = mpfr_get_d(ratio.get(), MPFR_RNDN);
    }

    result.max_absolute_at_zero = mpfr_get_d(max_abs_at_zero.get(), MPFR_RNDN);
    result.nan_inf_mismatches = nan_inf_mismatches;

    return result;
}

/* ------------------------------------------------------------------ */
/* Full matrix comparison                                               */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_matrix(const MpfrMatrix &ref, const void *out,
                                  int ld, const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrScalar val(prec), diff(prec), absdiff(prec), absref(prec);
    MpfrScalar ratio(prec), sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            accumulate_error(val, diff, absdiff, absref, ratio,
                             sum_sq_diff, sum_sq_ref, max_rel,
                             ref.at(i, j),
                             p + IDX(i, j, ld) * ctx.typesize,
                             ctx,
                             max_abs_at_zero, nan_inf_mismatches);
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Triangle-only matrix comparison                                      */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_matrix_triangle(const MpfrMatrix &ref, const void *out,
                                           int ld, char uplo, const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrScalar val(prec), diff(prec), absdiff(prec), absref(prec);
    MpfrScalar ratio(prec), sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j)) {
                accumulate_error(val, diff, absdiff, absref, ratio,
                                 sum_sq_diff, sum_sq_ref, max_rel,
                                 ref.at(i, j),
                                 p + IDX(i, j, ld) * ctx.typesize,
                                 ctx,
                                 max_abs_at_zero, nan_inf_mismatches);
            }
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Vector comparison                                                    */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_vector(const MpfrMatrix &ref, const void *out,
                                  int inc, const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrScalar val(prec), diff(prec), absdiff(prec), absref(prec);
    MpfrScalar ratio(prec), sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    int n = ref.rows();
    int abs_inc = (inc < 0) ? -inc : inc;

    for (int i = 0; i < n; ++i) {
        std::size_t offset;
        if (inc > 0) {
            offset = static_cast<std::size_t>(i) * abs_inc * ctx.typesize;
        } else {
            offset = static_cast<std::size_t>(n - 1 - i) * abs_inc * ctx.typesize;
        }
        accumulate_error(val, diff, absdiff, absref, ratio,
                         sum_sq_diff, sum_sq_ref, max_rel,
                         ref.at(i, 0),
                         p + offset,
                         ctx,
                         max_abs_at_zero, nan_inf_mismatches);
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Scalar comparison                                                    */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_scalar(const MpfrScalar &ref, const void *out,
                                  const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrScalar val(prec), diff(prec), absdiff(prec), absref(prec), ratio(prec);

    ctx.to_mpfr(val.get(), out);

    ErrorResult result;
    result.max_absolute_at_zero = -1.0;
    result.nan_inf_mismatches = 0;

    /* --- NaN/Inf handling --- */
    int ref_nan = mpfr_nan_p(ref.get());
    int ref_inf = mpfr_inf_p(ref.get());
    int val_nan = mpfr_nan_p(val.get());
    int val_inf = mpfr_inf_p(val.get());

    if (ref_nan) {
        if (!val_nan)
            result.nan_inf_mismatches = 1;
        result.max_relative = 0.0;
        result.normwise_relative = 0.0;
        return result;
    }
    if (ref_inf) {
        if (!val_inf || mpfr_sgn(ref.get()) != mpfr_sgn(val.get()))
            result.nan_inf_mismatches = 1;
        result.max_relative = 0.0;
        result.normwise_relative = 0.0;
        return result;
    }
    if (val_nan || val_inf) {
        result.nan_inf_mismatches = 1;
        result.max_relative = 0.0;
        result.normwise_relative = 0.0;
        return result;
    }

    /* --- Normal finite path --- */
    mpfr_sub(diff.get(), val.get(), ref.get(), MPFR_RNDN);
    mpfr_abs(absdiff.get(), diff.get(), MPFR_RNDN);
    mpfr_abs(absref.get(), ref.get(), MPFR_RNDN);

    if (mpfr_zero_p(absref.get())) {
        result.max_relative = mpfr_zero_p(absdiff.get()) ? 0.0
                              : mpfr_get_d(absdiff.get(), MPFR_RNDN);
        result.normwise_relative = result.max_relative;
        result.max_absolute_at_zero = mpfr_get_d(absdiff.get(), MPFR_RNDN);
    } else {
        mpfr_div(ratio.get(), absdiff.get(), absref.get(), MPFR_RNDN);
        result.max_relative = mpfr_get_d(ratio.get(), MPFR_RNDN);
        result.normwise_relative = result.max_relative;
    }

    return result;
}

/* ------------------------------------------------------------------ */
/* Integer comparison (exact)                                           */
/* ------------------------------------------------------------------ */

bool compute_error_index(int ref, int out)
{
    return ref == out;
}

/* ================================================================== */
/* COMPLEX ERROR METRICS                                               */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* Internal helper: accumulate complex error for a single element      */
/* ------------------------------------------------------------------ */

static void accumulate_complex_error(
    MpfrComplexScalar &val, MpfrScalar &diff_abs, MpfrScalar &ref_abs,
    MpfrScalar &ratio,
    MpfrScalar &sum_sq_diff, MpfrScalar &sum_sq_ref, MpfrScalar &max_rel,
    const mpfr_t &ref_re, const mpfr_t &ref_im,
    const void *out_elem, const TesterCtx &ctx,
    MpfrScalar &max_abs_at_zero, int &nan_inf_mismatches)
{
    ctx.to_mpfr_complex(val.re(), val.im(), out_elem);

    /* NaN/Inf check: any NaN/Inf component is flagged */
    int ref_bad = mpfr_nan_p(ref_re) || mpfr_inf_p(ref_re) ||
                  mpfr_nan_p(ref_im) || mpfr_inf_p(ref_im);
    int val_bad = mpfr_nan_p(val.re()) || mpfr_inf_p(val.re()) ||
                  mpfr_nan_p(val.im()) || mpfr_inf_p(val.im());

    if (ref_bad || val_bad) {
        if (ref_bad != val_bad)
            ++nan_inf_mismatches;
        return;
    }

    /* |out - ref| = sqrt((re_out-re_ref)^2 + (im_out-im_ref)^2) */
    mpfr_prec_t prec = ctx.prec;
    MpfrScalar d_re(prec), d_im(prec);
    mpfr_sub(d_re.get(), val.re(), ref_re, MPFR_RNDN);
    mpfr_sub(d_im.get(), val.im(), ref_im, MPFR_RNDN);

    mpfr_complex_abs(diff_abs.get(), d_re.get(), d_im.get(), MPFR_RNDN);
    mpfr_complex_abs(ref_abs.get(), ref_re, ref_im, MPFR_RNDN);

    /* Frobenius accumulation: sum |diff|^2, sum |ref|^2 */
    mpfr_fma(sum_sq_diff.get(), diff_abs.get(), diff_abs.get(),
             sum_sq_diff.get(), MPFR_RNDN);
    mpfr_fma(sum_sq_ref.get(), ref_abs.get(), ref_abs.get(),
             sum_sq_ref.get(), MPFR_RNDN);

    if (mpfr_zero_p(ref_abs.get())) {
        if (!mpfr_zero_p(diff_abs.get())) {
            if (mpfr_sgn(max_abs_at_zero.get()) < 0 ||
                mpfr_cmp(diff_abs.get(), max_abs_at_zero.get()) > 0)
                mpfr_set(max_abs_at_zero.get(), diff_abs.get(), MPFR_RNDN);
        } else {
            if (mpfr_sgn(max_abs_at_zero.get()) < 0)
                mpfr_set_d(max_abs_at_zero.get(), 0.0, MPFR_RNDN);
        }
        return;
    }
    mpfr_div(ratio.get(), diff_abs.get(), ref_abs.get(), MPFR_RNDN);
    if (mpfr_cmp(ratio.get(), max_rel.get()) > 0)
        mpfr_set(max_rel.get(), ratio.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Complex full matrix comparison                                      */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_complex_matrix(const MpfrComplexMatrix &ref,
                                          const void *out, int ld,
                                          const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrComplexScalar val(prec);
    MpfrScalar diff_abs(prec), ref_abs(prec), ratio(prec);
    MpfrScalar sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            accumulate_complex_error(val, diff_abs, ref_abs, ratio,
                                     sum_sq_diff, sum_sq_ref, max_rel,
                                     ref.re(i, j), ref.im(i, j),
                                     p + IDX(i, j, ld) * ctx.typesize,
                                     ctx,
                                     max_abs_at_zero, nan_inf_mismatches);
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Complex triangle-only matrix comparison                             */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_complex_matrix_triangle(const MpfrComplexMatrix &ref,
                                                   const void *out, int ld,
                                                   char uplo,
                                                   const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrComplexScalar val(prec);
    MpfrScalar diff_abs(prec), ref_abs(prec), ratio(prec);
    MpfrScalar sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j)) {
                accumulate_complex_error(val, diff_abs, ref_abs, ratio,
                                         sum_sq_diff, sum_sq_ref, max_rel,
                                         ref.re(i, j), ref.im(i, j),
                                         p + IDX(i, j, ld) * ctx.typesize,
                                         ctx,
                                         max_abs_at_zero, nan_inf_mismatches);
            }
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Complex vector comparison                                           */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_complex_vector(const MpfrComplexMatrix &ref,
                                          const void *out, int inc,
                                          const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrComplexScalar val(prec);
    MpfrScalar diff_abs(prec), ref_abs(prec), ratio(prec);
    MpfrScalar sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    int n = ref.rows();
    int abs_inc = (inc < 0) ? -inc : inc;

    for (int i = 0; i < n; ++i) {
        std::size_t offset;
        if (inc > 0) {
            offset = static_cast<std::size_t>(i) * abs_inc * ctx.typesize;
        } else {
            offset = static_cast<std::size_t>(n - 1 - i) * abs_inc * ctx.typesize;
        }
        accumulate_complex_error(val, diff_abs, ref_abs, ratio,
                                 sum_sq_diff, sum_sq_ref, max_rel,
                                 ref.re(i, 0), ref.im(i, 0),
                                 p + offset,
                                 ctx,
                                 max_abs_at_zero, nan_inf_mismatches);
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Complex scalar comparison                                           */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_complex_scalar(const MpfrComplexScalar &ref,
                                          const void *out,
                                          const TesterCtx &ctx)
{
    mpfr_prec_t prec = ctx.prec;
    MpfrComplexScalar val(prec);
    ctx.to_mpfr_complex(val.re(), val.im(), out);

    ErrorResult result;
    result.max_absolute_at_zero = -1.0;
    result.nan_inf_mismatches = 0;

    int ref_bad = mpfr_nan_p(ref.re()) || mpfr_inf_p(ref.re()) ||
                  mpfr_nan_p(ref.im()) || mpfr_inf_p(ref.im());
    int val_bad = mpfr_nan_p(val.re()) || mpfr_inf_p(val.re()) ||
                  mpfr_nan_p(val.im()) || mpfr_inf_p(val.im());

    if (ref_bad || val_bad) {
        if (ref_bad != val_bad)
            result.nan_inf_mismatches = 1;
        result.max_relative = 0.0;
        result.normwise_relative = 0.0;
        return result;
    }

    MpfrScalar d_re(prec), d_im(prec), diff_abs(prec), ref_abs(prec), ratio(prec);
    mpfr_sub(d_re.get(), val.re(), ref.re(), MPFR_RNDN);
    mpfr_sub(d_im.get(), val.im(), ref.im(), MPFR_RNDN);
    mpfr_complex_abs(diff_abs.get(), d_re.get(), d_im.get(), MPFR_RNDN);
    mpfr_complex_abs(ref_abs.get(), ref.re(), ref.im(), MPFR_RNDN);

    if (mpfr_zero_p(ref_abs.get())) {
        double da = mpfr_get_d(diff_abs.get(), MPFR_RNDN);
        result.max_relative = da;
        result.normwise_relative = da;
        result.max_absolute_at_zero = da;
    } else {
        mpfr_div(ratio.get(), diff_abs.get(), ref_abs.get(), MPFR_RNDN);
        result.max_relative = mpfr_get_d(ratio.get(), MPFR_RNDN);
        result.normwise_relative = result.max_relative;
    }

    return result;
}
