/* mirror_pblas_common.h -- Shared PBLAS mirror testing infrastructure */

#pragma once

#include "../mirror_ctx.h"
#include "../mirror_gen.h"
#include "../mirror_error.h"
#include "../mirror_report.h"
#include "../../src/core/mpfr_types.h"
#include "../../src/core/mpfr_complex_types.h"
#include "../../src/core/loader.h"
#include "../../src/blacs/blacs_common.h"
#include "../../src/pblas/pblas_common.h"

#include <mpfr.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>

/* MirrorPblasCtx: PblasCtx that initializes from a MirrorSide's library.
   Each side gets its own BLACS grid, initialized independently. */
struct MirrorPblasCtx {
    PblasCtx pc;

    bool init(const MirrorSide &side, int mb, int nb) {
        return pc.init(side.lib, mb, nb);
    }

    void finalize() { pc.finalize(); }

    int local_rows(int m) const { return pc.local_rows(m); }
    int local_cols(int n) const { return pc.local_cols(n); }
    void make_desc(int desc[9], int m, int n, int lld) const {
        pc.make_desc(desc, m, n, lld);
    }
    bool is_root() const { return pc.bc.is_root(); }
};

/* Scatter MPFR global matrix to native local storage for a side */
inline void *scatter_mpfr_to_local(const MpfrMatrix &global, int m, int n,
                                     int lld, const MirrorPblasCtx &mpc,
                                     const TesterCtx &ctx)
{
    /* First materialize global to native with ld=m */
    void *g_native = mpfr_mat_to_native(global, m, ctx);

    /* Allocate local storage */
    int loc_n = mpc.local_cols(n);
    void *local = std::calloc(
        static_cast<std::size_t>(lld) * std::max(1, loc_n), ctx.typesize);

    scatter_global_to_local(local, lld, g_native, m,
                             m, n, mpc.pc.mb, mpc.pc.nb,
                             mpc.pc.bc.myrow, mpc.pc.bc.mycol,
                             mpc.pc.bc.nprow, mpc.pc.bc.npcol,
                             ctx.typesize);
    std::free(g_native);
    return local;
}

/* Scatter MPFR complex global matrix to native local storage */
inline void *scatter_mpfr_complex_to_local(const MpfrComplexMatrix &global,
                                             int m, int n, int lld,
                                             const MirrorPblasCtx &mpc,
                                             const TesterCtx &ctx)
{
    void *g_native = mpfr_complex_mat_to_native(global, m, ctx);
    int loc_n = mpc.local_cols(n);
    void *local = std::calloc(
        static_cast<std::size_t>(lld) * std::max(1, loc_n), ctx.typesize);

    scatter_global_to_local(local, lld, g_native, m,
                             m, n, mpc.pc.mb, mpc.pc.nb,
                             mpc.pc.bc.myrow, mpc.pc.bc.mycol,
                             mpc.pc.bc.nprow, mpc.pc.bc.npcol,
                             ctx.typesize);
    std::free(g_native);
    return local;
}

/* Gather native local matrix back to MPFR for comparison.
   Reads from local native storage using ctx.to_mpfr. */
inline void gather_local_to_mpfr(MpfrMatrix &result, const void *local,
                                   int lld, int m, int n,
                                   const MirrorPblasCtx &mpc,
                                   const TesterCtx &ctx)
{
    int loc_m = mpc.local_rows(m);
    int loc_n = mpc.local_cols(n);
    const char *p = static_cast<const char *>(local);

    /* First fill result with zeros */
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            mpfr_set_d(result.at(i, j), 0.0, MPFR_RNDN);

    /* Copy local elements to their global positions */
    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, mpc.pc.nb, mpc.pc.bc.mycol, 0, mpc.pc.bc.npcol);
        for (int il = 0; il < loc_m; ++il) {
            int ig = indxl2g(il, mpc.pc.mb, mpc.pc.bc.myrow, 0, mpc.pc.bc.nprow);
            std::size_t off = (static_cast<std::size_t>(jl) * lld + il) * ctx.typesize;
            ctx.to_mpfr(result.at(ig, jg), p + off);
        }
    }
}

/* Gather native local complex matrix back to MPFR */
inline void gather_local_complex_to_mpfr(MpfrComplexMatrix &result,
                                           const void *local, int lld,
                                           int m, int n,
                                           const MirrorPblasCtx &mpc,
                                           const TesterCtx &ctx)
{
    int loc_m = mpc.local_rows(m);
    int loc_n = mpc.local_cols(n);
    const char *p = static_cast<const char *>(local);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            mpfr_set_d(result.re(i, j), 0.0, MPFR_RNDN);
            mpfr_set_d(result.im(i, j), 0.0, MPFR_RNDN);
        }
    }

    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, mpc.pc.nb, mpc.pc.bc.mycol, 0, mpc.pc.bc.npcol);
        for (int il = 0; il < loc_m; ++il) {
            int ig = indxl2g(il, mpc.pc.mb, mpc.pc.bc.myrow, 0, mpc.pc.bc.nprow);
            std::size_t off = (static_cast<std::size_t>(jl) * lld + il) * ctx.typesize;
            ctx.to_mpfr_complex(result.re(ig, jg), result.im(ig, jg), p + off);
        }
    }
}
