/* pblas_common.h -- Shared PBLAS testing infrastructure */
/* Provides block-cyclic distribution utilities, PblasCtx for BLACS context
   and process grid management, and scatter/extract helpers for distributed
   matrix generation and MPFR reference comparison. */

#pragma once

#include "../blacs/blacs_common.h"
#include "../core/mpfr_types.h"
#include "../core/tester_ctx.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Block-cyclic distribution utilities (0-based indexing)              */
/* Implements the ScaLAPACK NUMROC, INDXG2P, INDXG2L, INDXL2G.       */
/* ------------------------------------------------------------------ */

/* Number of rows/cols of a distributed matrix owned by process iproc */
inline int numroc(int n, int nb, int iproc, int isrcproc, int nprocs)
{
    if (n <= 0 || nb <= 0 || nprocs <= 0) return 0;
    int mydist = (iproc - isrcproc + nprocs) % nprocs;
    int nblocks = n / nb;
    int result = (nblocks / nprocs) * nb;
    int extra = nblocks % nprocs;
    if (mydist < extra)
        result += nb;
    else if (mydist == extra)
        result += n % nb;
    return result;
}

/* Global index -> process that owns it */
inline int indxg2p(int indxglob, int nb, int isrcproc, int nprocs)
{
    return (isrcproc + indxglob / nb) % nprocs;
}

/* Global index -> local index on the owning process */
inline int indxg2l(int indxglob, int nb, int nprocs)
{
    return (indxglob / (nb * nprocs)) * nb + indxglob % nb;
}

/* Local index -> global index */
inline int indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs)
{
    return nprocs * nb * (indxloc / nb) +
           ((iproc - isrcproc + nprocs) % nprocs) * nb +
           (indxloc % nb);
}

/* ------------------------------------------------------------------ */
/* PblasCtx: BLACS context + block-cyclic parameters                  */
/* ------------------------------------------------------------------ */

struct PblasCtx {
    BlacsCtx bc;
    int mb = 1, nb = 1;  /* Block sizes for row and column distribution */

    /* Initialize BLACS context and process grid.
       Automatically determines the best grid topology.
       Returns false if BLACS symbols cannot be loaded. */
    bool init(void *lib, int mb_, int nb_)
    {
        if (!bc.load(lib)) return false;
        mb = std::max(1, mb_);
        nb = std::max(1, nb_);

        bc.fn_pinfo(&bc.mypnum, &bc.nprocs);

        int nprow, npcol;
        compute_grid(bc.nprocs, nprow, npcol);
        bc.init_grid(nprow, npcol, 'R');
        return bc.in_grid();
    }

    void finalize() { bc.finalize(); }

    /* Local dimensions for this process */
    int local_rows(int m) const { return numroc(m, mb, bc.myrow, 0, bc.nprow); }
    int local_cols(int n) const { return numroc(n, nb, bc.mycol, 0, bc.npcol); }

    /* Fill a 9-element ScaLAPACK array descriptor */
    void make_desc(int desc[9], int m, int n, int lld) const
    {
        desc[0] = 1;          /* DTYPE_ = dense */
        desc[1] = bc.ictxt;   /* CTXT_          */
        desc[2] = m;          /* M_ (global)    */
        desc[3] = n;          /* N_ (global)    */
        desc[4] = mb;         /* MB_            */
        desc[5] = nb;         /* NB_            */
        desc[6] = 0;          /* RSRC_          */
        desc[7] = 0;          /* CSRC_          */
        desc[8] = lld;        /* LLD_           */
    }

private:
    /* Find a near-square nprow x npcol grid for nprocs processes */
    static void compute_grid(int nprocs, int &nprow, int &npcol)
    {
        nprow = 1;
        npcol = nprocs;
        for (int r = static_cast<int>(std::sqrt(static_cast<double>(nprocs)));
             r >= 1; --r) {
            if (nprocs % r == 0) {
                nprow = r;
                npcol = nprocs / r;
                break;
            }
        }
    }
};

/* ------------------------------------------------------------------ */
/* Scatter: global column-major matrix -> local block-cyclic storage   */
/* ------------------------------------------------------------------ */

inline void scatter_global_to_local(void *local, int lld,
                                     const void *global, int ldg,
                                     int m, int n, int mb, int nb,
                                     int myrow, int mycol,
                                     int nprow, int npcol,
                                     std::size_t typesize)
{
    int loc_m = numroc(m, mb, myrow, 0, nprow);
    int loc_n = numroc(n, nb, mycol, 0, npcol);

    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, nb, mycol, 0, npcol);
        for (int il = 0; il < loc_m; ++il) {
            int ig = indxl2g(il, mb, myrow, 0, nprow);
            std::memcpy(
                static_cast<char *>(local) +
                    (static_cast<std::size_t>(jl) * lld + il) * typesize,
                static_cast<const char *>(global) +
                    (static_cast<std::size_t>(jg) * ldg + ig) * typesize,
                typesize);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Extract: local portion of global MPFR matrix for this process       */
/* ------------------------------------------------------------------ */

inline void extract_local_mpfr(MpfrMatrix &local, const MpfrMatrix &global,
                                int m, int n, int mb, int nb,
                                int myrow, int mycol,
                                int nprow, int npcol)
{
    int loc_m = numroc(m, mb, myrow, 0, nprow);
    int loc_n = numroc(n, nb, mycol, 0, npcol);

    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, nb, mycol, 0, npcol);
        for (int il = 0; il < loc_m; ++il) {
            int ig = indxl2g(il, mb, myrow, 0, nprow);
            mpfr_set(local.at(il, jl), global.at(ig, jg), MPFR_RNDN);
        }
    }
}

/* Build a local reference for triangular PBLAS results (PSYRK, PSYR2K).
   Elements in the uplo triangle use global_ref (the computed reference);
   elements outside use global_in (the initial C, which PBLAS leaves unchanged). */
inline void extract_local_mpfr_sym(MpfrMatrix &local,
                                    const MpfrMatrix &global_ref,
                                    const MpfrMatrix &global_in,
                                    int n, int mb, int nb,
                                    int myrow, int mycol,
                                    int nprow, int npcol,
                                    char uplo)
{
    int loc_m = numroc(n, mb, myrow, 0, nprow);
    int loc_n = numroc(n, nb, mycol, 0, npcol);

    for (int jl = 0; jl < loc_n; ++jl) {
        int jg = indxl2g(jl, nb, mycol, 0, npcol);
        for (int il = 0; il < loc_m; ++il) {
            int ig = indxl2g(il, mb, myrow, 0, nprow);
            if ((uplo == 'U' && ig <= jg) || (uplo == 'L' && ig >= jg))
                mpfr_set(local.at(il, jl), global_ref.at(ig, jg), MPFR_RNDN);
            else
                mpfr_set(local.at(il, jl), global_in.at(ig, jg), MPFR_RNDN);
        }
    }
}
