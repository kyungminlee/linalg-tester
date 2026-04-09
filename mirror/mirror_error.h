/* mirror_error.h -- MPFR-vs-MPFR error comparison */

#pragma once

#include "../src/core/tester_ctx.h"
#include "../src/core/mpfr_types.h"
#include "../src/core/mpfr_complex_types.h"

/* Compare two MPFR matrices element-wise */
ErrorResult compute_error_mpfr_matrix(const MpfrMatrix &ref,
                                       const MpfrMatrix &test,
                                       mpfr_prec_t prec);

/* Compare two MPFR vectors element-wise */
ErrorResult compute_error_mpfr_vector(const MpfrMatrix &ref,
                                       const MpfrMatrix &test,
                                       mpfr_prec_t prec);

/* Compare two MPFR scalars */
ErrorResult compute_error_mpfr_scalar(const MpfrScalar &ref,
                                       const MpfrScalar &test,
                                       mpfr_prec_t prec);

/* Complex variants */
ErrorResult compute_error_mpfr_complex_matrix(const MpfrComplexMatrix &ref,
                                               const MpfrComplexMatrix &test,
                                               mpfr_prec_t prec);

ErrorResult compute_error_mpfr_complex_vector(const MpfrComplexMatrix &ref,
                                               const MpfrComplexMatrix &test,
                                               mpfr_prec_t prec);

ErrorResult compute_error_mpfr_complex_scalar(const MpfrComplexScalar &ref,
                                               const MpfrComplexScalar &test,
                                               mpfr_prec_t prec);
