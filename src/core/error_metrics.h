#pragma once

#include "mpfr_types.h"
#include "tester_ctx.h"

// Full matrix comparison
ErrorResult compute_error_matrix(const MpfrMatrix &ref, const void *out,
                                  int ld, const TesterCtx &ctx);

// Triangle-only matrix comparison (for SYRK, SYR2K, SYR, etc.)
ErrorResult compute_error_matrix_triangle(const MpfrMatrix &ref, const void *out,
                                           int ld, char uplo, const TesterCtx &ctx);

// Vector comparison (ref is MpfrMatrix(n,1))
ErrorResult compute_error_vector(const MpfrMatrix &ref, const void *out,
                                  int inc, const TesterCtx &ctx);

// Scalar comparison
ErrorResult compute_error_scalar(const MpfrScalar &ref, const void *out,
                                  const TesterCtx &ctx);

// Integer comparison (exact)
bool compute_error_index(int ref, int out);
