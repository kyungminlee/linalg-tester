/* mirror_lapack_common.h -- Shared utilities for LAPACK mirror tests */

#pragma once

#include "../mirror_ctx.h"
#include "../mirror_gen.h"
#include "../mirror_error.h"
#include "../mirror_report.h"
#include "../../src/core/mpfr_types.h"
#include "../../src/core/mpfr_complex_types.h"
#include "../../src/core/loader.h"

#include <mpfr.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

/* Query optimal workspace size from WORK(1) after lwork=-1 call */
inline int mirror_query_lwork(const void *work_val, const TesterCtx &ctx)
{
    MpfrScalar tmp(ctx.prec);
    ctx.to_mpfr(tmp.get(), work_val);
    return static_cast<int>(mpfr_get_d(tmp.get(), MPFR_RNDN));
}

/* Complex variant: lwork is in the real part of WORK(1) */
inline int mirror_query_lwork_complex(const void *work_val, const TesterCtx &ctx)
{
    /* In complex mode, to_mpfr reads only the first (real) component */
    MpfrScalar tmp(ctx.prec);
    ctx.to_mpfr(tmp.get(), work_val);
    return static_cast<int>(mpfr_get_d(tmp.get(), MPFR_RNDN));
}

/* Helper to convert a native array of n scalars to an MpfrMatrix (n x 1) */
inline void native_array_to_mpfr(MpfrMatrix &dst, const void *src, int n,
                                   const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    for (int i = 0; i < n; ++i)
        ctx.to_mpfr(dst.at(i, 0), p + static_cast<std::size_t>(i) * ctx.typesize);
}

/* Helper to convert a native complex array of n elements to MpfrComplexMatrix (n x 1) */
inline void native_complex_array_to_mpfr(MpfrComplexMatrix &dst, const void *src,
                                           int n, const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    for (int i = 0; i < n; ++i)
        ctx.to_mpfr_complex(dst.re(i, 0), dst.im(i, 0),
                            p + static_cast<std::size_t>(i) * ctx.typesize);
}

/* Helper to convert a native real array (from complex context, typesize/2 per element) */
inline void native_real_array_to_mpfr(MpfrMatrix &dst, const void *src, int n,
                                        const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    std::size_t real_ts = ctx.typesize / 2;
    for (int i = 0; i < n; ++i)
        ctx.to_mpfr(dst.at(i, 0), p + static_cast<std::size_t>(i) * real_ts);
}
