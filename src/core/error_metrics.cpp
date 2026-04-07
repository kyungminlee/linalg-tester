#include "error_metrics.h"

#include <cstdlib>

/* ------------------------------------------------------------------ */
/* Internal helper: accumulate error for a single element               */
/* ------------------------------------------------------------------ */

static void accumulate_error(
    MpfrScalar &val, MpfrScalar &diff, MpfrScalar &absdiff,
    MpfrScalar &absref, MpfrScalar &ratio,
    MpfrScalar &sum_sq_diff, MpfrScalar &sum_sq_ref, MpfrScalar &max_rel,
    const mpfr_t &ref_elem, const void *out_elem,
    const TesterCtx &ctx)
{
    ctx.to_mpfr(val.get(), out_elem);

    mpfr_sub(diff.get(), val.get(), ref_elem, MPFR_RNDN);
    mpfr_abs(absdiff.get(), diff.get(), MPFR_RNDN);
    mpfr_abs(absref.get(), ref_elem, MPFR_RNDN);

    mpfr_fma(sum_sq_diff.get(), diff.get(), diff.get(),
             sum_sq_diff.get(), MPFR_RNDN);
    mpfr_fma(sum_sq_ref.get(), ref_elem, ref_elem,
             sum_sq_ref.get(), MPFR_RNDN);

    if (mpfr_zero_p(absref.get()))
        return;
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
                                   MpfrScalar &ratio)
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

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            accumulate_error(val, diff, absdiff, absref, ratio,
                             sum_sq_diff, sum_sq_ref, max_rel,
                             ref.at(i, j),
                             p + IDX(i, j, ld) * ctx.typesize,
                             ctx);
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio);
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

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);

    const char *p = static_cast<const char *>(out);
    for (int j = 0; j < ref.cols(); ++j) {
        for (int i = 0; i < ref.rows(); ++i) {
            if ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j)) {
                accumulate_error(val, diff, absdiff, absref, ratio,
                                 sum_sq_diff, sum_sq_ref, max_rel,
                                 ref.at(i, j),
                                 p + IDX(i, j, ld) * ctx.typesize,
                                 ctx);
            }
        }
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio);
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

    mpfr_set_d(sum_sq_diff.get(), 0.0, MPFR_RNDN);
    mpfr_set_d(sum_sq_ref.get(),  0.0, MPFR_RNDN);
    mpfr_set_d(max_rel.get(),     0.0, MPFR_RNDN);

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
                         ctx);
    }

    return finalize_error(sum_sq_diff, sum_sq_ref, max_rel, ratio);
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

    mpfr_sub(diff.get(), val.get(), ref.get(), MPFR_RNDN);
    mpfr_abs(absdiff.get(), diff.get(), MPFR_RNDN);
    mpfr_abs(absref.get(), ref.get(), MPFR_RNDN);

    ErrorResult result;
    if (mpfr_zero_p(absref.get())) {
        result.max_relative = mpfr_zero_p(absdiff.get()) ? 0.0
                              : mpfr_get_d(absdiff.get(), MPFR_RNDN);
        result.normwise_relative = result.max_relative;
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
