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
/* Report a LAPACK result with proper labeled fields                    */
/* ------------------------------------------------------------------ */

inline void report_lapack_result(const char *routine, const char *params_str,
                                  const LapackResult &lr, const std::string &format)
{
    if (format == "json") {
        std::printf("{\"routine\":\"%s\",\"params\":\"%s\",\"residual\":%.6e",
                    routine, params_str, lr.residual);
        if (lr.orthogonality >= 0.0)
            std::printf(",\"orthogonality\":%.6e", lr.orthogonality);
        if (lr.info != 0)
            std::printf(",\"info\":%d", lr.info);
        std::printf("}\n");
    } else if (format == "csv") {
        std::printf("%s,%s,%.6e,", routine, params_str, lr.residual);
        if (lr.orthogonality >= 0.0)
            std::printf("%.6e,", lr.orthogonality);
        else
            std::printf(",");
        std::printf("%d\n", lr.info);
    } else {
        /* default: text */
        std::printf("[%s %s] residual=%.6e", routine, params_str, lr.residual);
        if (lr.orthogonality >= 0.0)
            std::printf("  orthogonality=%.6e", lr.orthogonality);
        if (lr.info != 0)
            std::printf("  INFO=%d", lr.info);
        std::printf("\n");
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
