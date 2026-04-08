/* lapack_common.h -- Shared LAPACK testing infrastructure */

#pragma once

#include "../core/tester_ctx.h"
#include "../core/mpfr_types.h"
#include "../core/report.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/* ------------------------------------------------------------------ */
/* Residual-based error result for LAPACK routines                      */
/* ------------------------------------------------------------------ */

struct LapackResult {
    double residual;         // ||A - reconstruct(factors)|| / (||A|| * n * eps)
    double orthogonality;    // ||Q^T*Q - I|| / (max(m,n) * eps)  (-1 if N/A)
    int    info;             // INFO return value from LAPACK
};

/* ------------------------------------------------------------------ */
/* Report a LAPACK result using the standard report_result             */
/* ------------------------------------------------------------------ */

inline void report_lapack_result(const char *routine, const char *params_str,
                                  const LapackResult &lr, const std::string &format)
{
    /* Pack residual into ErrorResult for compatibility with report_result */
    ErrorResult err;
    err.max_relative = lr.residual;
    err.normwise_relative = (lr.orthogonality >= 0.0) ? lr.orthogonality : lr.residual;

    if (lr.info != 0) {
        char info_params[256];
        std::snprintf(info_params, sizeof(info_params),
                      "%s INFO=%d", params_str, lr.info);
        report_result(routine, info_params, err, format);
    } else {
        report_result(routine, params_str, err, format);
    }
}

/* ------------------------------------------------------------------ */
/* Workspace query helper                                              */
/* Calls a LAPACK routine with LWORK=-1 to get optimal workspace size  */
/* Returns the optimal LWORK as an integer                             */
/* ------------------------------------------------------------------ */

inline int query_lwork(const void *work_val, const TesterCtx &ctx)
{
    MpfrScalar tmp(ctx.prec);
    ctx.to_mpfr(tmp.get(), work_val);
    return static_cast<int>(mpfr_get_d(tmp.get(), MPFR_RNDN));
}
