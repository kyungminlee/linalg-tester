/* mirror_error.cpp -- MPFR-vs-MPFR error comparison functions
 *
 * These follow the same error accumulation pattern as src/core/error_metrics.cpp
 * but compare two MPFR values directly instead of converting a native value
 * to MPFR first.
 */

#include "mirror_error.h"
#include "../src/core/mpfr_complex.h"

#include <cstdlib>

/* ------------------------------------------------------------------ */
/* Internal helper: accumulate error for a single real element pair     */
/* ------------------------------------------------------------------ */

static void accumulate_mpfr_error(
    MpfrScalar &diff, MpfrScalar &absdiff,
    MpfrScalar &absref, MpfrScalar &ratio,
    MpfrScalar &sum_sq_diff, MpfrScalar &sum_sq_ref, MpfrScalar &max_rel,
    const mpfr_t &ref_elem, const mpfr_t &test_elem,
    MpfrScalar &max_abs_at_zero, int &nan_inf_mismatches)
{
    /* --- NaN/Inf handling --- */
    int ref_nan = mpfr_nan_p(ref_elem);
    int ref_inf = mpfr_inf_p(ref_elem);
    int val_nan = mpfr_nan_p(test_elem);
    int val_inf = mpfr_inf_p(test_elem);

    if (ref_nan) {
        if (!val_nan)
            ++nan_inf_mismatches;
        return;
    }
    if (ref_inf) {
        if (!val_inf || mpfr_sgn(ref_elem) != mpfr_sgn(test_elem))
            ++nan_inf_mismatches;
        return;
    }
    /* ref is finite */
    if (val_nan || val_inf) {
        ++nan_inf_mismatches;
        return;
    }

    /* --- Normal finite path --- */
    mpfr_sub(diff.get(), test_elem, ref_elem, MPFR_RNDN);
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
/* Internal helper: accumulate error for a single complex element pair */
/* ------------------------------------------------------------------ */

static void accumulate_mpfr_complex_error(
    MpfrScalar &diff_abs, MpfrScalar &ref_abs,
    MpfrScalar &ratio,
    MpfrScalar &sum_sq_diff, MpfrScalar &sum_sq_ref, MpfrScalar &max_rel,
    const mpfr_t &ref_re, const mpfr_t &ref_im,
    const mpfr_t &test_re, const mpfr_t &test_im,
    mpfr_prec_t prec,
    MpfrScalar &max_abs_at_zero, int &nan_inf_mismatches)
{
    /* NaN/Inf check: any NaN/Inf component is flagged */
    int ref_bad = mpfr_nan_p(ref_re) || mpfr_inf_p(ref_re) ||
                  mpfr_nan_p(ref_im) || mpfr_inf_p(ref_im);
    int val_bad = mpfr_nan_p(test_re) || mpfr_inf_p(test_re) ||
                  mpfr_nan_p(test_im) || mpfr_inf_p(test_im);

    if (ref_bad || val_bad) {
        if (ref_bad != val_bad)
            ++nan_inf_mismatches;
        return;
    }

    /* |test - ref| = sqrt((re_test-re_ref)^2 + (im_test-im_ref)^2) */
    MpfrScalar d_re(prec), d_im(prec);
    mpfr_sub(d_re.get(), test_re, ref_re, MPFR_RNDN);
    mpfr_sub(d_im.get(), test_im, ref_im, MPFR_RNDN);

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

/* ================================================================== */
/* REAL ERROR METRICS                                                  */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* Full matrix comparison                                               */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_mpfr_matrix(const MpfrMatrix &ref,
                                       const MpfrMatrix &test,
                                       mpfr_prec_t prec)
{
    MpfrScalar diff(prec), absdiff(prec), absref(prec);
    MpfrScalar ratio(prec), sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            accumulate_mpfr_error(diff, absdiff, absref, ratio,
                                  sum_sq_diff, sum_sq_ref, max_rel,
                                  ref.at(i, j), test.at(i, j),
                                  max_abs_at_zero, nan_inf_mismatches);
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Vector comparison                                                    */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_mpfr_vector(const MpfrMatrix &ref,
                                       const MpfrMatrix &test,
                                       mpfr_prec_t prec)
{
    MpfrScalar diff(prec), absdiff(prec), absref(prec);
    MpfrScalar ratio(prec), sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    int n = ref.rows();
    for (int i = 0; i < n; ++i) {
        accumulate_mpfr_error(diff, absdiff, absref, ratio,
                              sum_sq_diff, sum_sq_ref, max_rel,
                              ref.at(i, 0), test.at(i, 0),
                              max_abs_at_zero, nan_inf_mismatches);
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Scalar comparison                                                    */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_mpfr_scalar(const MpfrScalar &ref,
                                       const MpfrScalar &test,
                                       mpfr_prec_t prec)
{
    MpfrScalar diff(prec), absdiff(prec), absref(prec), ratio(prec);

    ErrorResult result;
    result.max_absolute_at_zero = -1.0;
    result.nan_inf_mismatches = 0;

    /* --- NaN/Inf handling --- */
    int ref_nan = mpfr_nan_p(ref.get());
    int ref_inf = mpfr_inf_p(ref.get());
    int val_nan = mpfr_nan_p(test.get());
    int val_inf = mpfr_inf_p(test.get());

    if (ref_nan) {
        if (!val_nan)
            result.nan_inf_mismatches = 1;
        result.max_relative = 0.0;
        result.normwise_relative = 0.0;
        return result;
    }
    if (ref_inf) {
        if (!val_inf || mpfr_sgn(ref.get()) != mpfr_sgn(test.get()))
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
    mpfr_sub(diff.get(), test.get(), ref.get(), MPFR_RNDN);
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

/* ================================================================== */
/* COMPLEX ERROR METRICS                                               */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* Complex full matrix comparison                                      */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_mpfr_complex_matrix(const MpfrComplexMatrix &ref,
                                               const MpfrComplexMatrix &test,
                                               mpfr_prec_t prec)
{
    MpfrScalar diff_abs(prec), ref_abs(prec), ratio(prec);
    MpfrScalar sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            accumulate_mpfr_complex_error(diff_abs, ref_abs, ratio,
                                          sum_sq_diff, sum_sq_ref, max_rel,
                                          ref.re(i, j), ref.im(i, j),
                                          test.re(i, j), test.im(i, j),
                                          prec,
                                          max_abs_at_zero, nan_inf_mismatches);
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Complex vector comparison                                           */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_mpfr_complex_vector(const MpfrComplexMatrix &ref,
                                               const MpfrComplexMatrix &test,
                                               mpfr_prec_t prec)
{
    MpfrScalar diff_abs(prec), ref_abs(prec), ratio(prec);
    MpfrScalar sum_sq_diff(prec), sum_sq_ref(prec), max_rel(prec);
    MpfrScalar max_abs_at_zero(prec);
    int nan_inf_mismatches = 0;

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);
    mpfr_set_d(max_abs_at_zero.get(), -1.0, MPFR_RNDN);

    int n = ref.rows();
    for (int i = 0; i < n; ++i) {
        accumulate_mpfr_complex_error(diff_abs, ref_abs, ratio,
                                      sum_sq_diff, sum_sq_ref, max_rel,
                                      ref.re(i, 0), ref.im(i, 0),
                                      test.re(i, 0), test.im(i, 0),
                                      prec,
                                      max_abs_at_zero, nan_inf_mismatches);
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio,
                          max_abs_at_zero, nan_inf_mismatches);
}

/* ------------------------------------------------------------------ */
/* Complex scalar comparison                                           */
/* ------------------------------------------------------------------ */

ErrorResult compute_error_mpfr_complex_scalar(const MpfrComplexScalar &ref,
                                               const MpfrComplexScalar &test,
                                               mpfr_prec_t prec)
{
    ErrorResult result;
    result.max_absolute_at_zero = -1.0;
    result.nan_inf_mismatches = 0;

    int ref_bad = mpfr_nan_p(ref.re()) || mpfr_inf_p(ref.re()) ||
                  mpfr_nan_p(ref.im()) || mpfr_inf_p(ref.im());
    int val_bad = mpfr_nan_p(test.re()) || mpfr_inf_p(test.re()) ||
                  mpfr_nan_p(test.im()) || mpfr_inf_p(test.im());

    if (ref_bad || val_bad) {
        if (ref_bad != val_bad)
            result.nan_inf_mismatches = 1;
        result.max_relative = 0.0;
        result.normwise_relative = 0.0;
        return result;
    }

    MpfrScalar d_re(prec), d_im(prec), diff_abs(prec), ref_abs(prec), ratio(prec);
    mpfr_sub(d_re.get(), test.re(), ref.re(), MPFR_RNDN);
    mpfr_sub(d_im.get(), test.im(), ref.im(), MPFR_RNDN);
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
