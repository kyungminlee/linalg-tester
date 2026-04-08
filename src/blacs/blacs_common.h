/* blacs_common.h -- Shared BLACS testing infrastructure */

#pragma once

#include "../core/tester_ctx.h"
#include "../core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <dlfcn.h>

/* ------------------------------------------------------------------ */
/* Non-fatal symbol loader (returns nullptr on failure)                */
/* ------------------------------------------------------------------ */

inline void *try_load_sym(void *lib, const char *name)
{
    dlerror();  /* clear */
    void *sym = dlsym(lib, name);
    return dlerror() ? nullptr : sym;
}

/* ------------------------------------------------------------------ */
/* BLACS context helper                                                */
/* ------------------------------------------------------------------ */

struct BlacsCtx {
    int ictxt  = -1;
    int nprow  = 0, npcol  = 0;
    int myrow  = -1, mycol  = -1;
    int mypnum = -1, nprocs = 0;

    /* Function pointer types (Fortran interface) */
    using pinfo_fn    = void (*)(int *, int *);
    using get_fn      = void (*)(const int *, const int *, int *);
    using gridinit_fn = void (*)(int *, const char *, const int *, const int *, std::size_t);
    using gridinfo_fn = void (*)(const int *, int *, int *, int *, int *);
    using gridexit_fn = void (*)(const int *);
    using exit_fn     = void (*)(const int *);

    pinfo_fn    fn_pinfo    = nullptr;
    get_fn      fn_get      = nullptr;
    gridinit_fn fn_gridinit = nullptr;
    gridinfo_fn fn_gridinfo = nullptr;
    gridexit_fn fn_gridexit = nullptr;
    exit_fn     fn_exit     = nullptr;

    /* Load all BLACS context symbols from lib. Returns false if any fail. */
    bool load(void *lib)
    {
        fn_pinfo    = reinterpret_cast<pinfo_fn>(try_load_sym(lib, "blacs_pinfo_"));
        fn_get      = reinterpret_cast<get_fn>(try_load_sym(lib, "blacs_get_"));
        fn_gridinit = reinterpret_cast<gridinit_fn>(try_load_sym(lib, "blacs_gridinit_"));
        fn_gridinfo = reinterpret_cast<gridinfo_fn>(try_load_sym(lib, "blacs_gridinfo_"));
        fn_gridexit = reinterpret_cast<gridexit_fn>(try_load_sym(lib, "blacs_gridexit_"));
        fn_exit     = reinterpret_cast<exit_fn>(try_load_sym(lib, "blacs_exit_"));
        return fn_pinfo && fn_get && fn_gridinit && fn_gridinfo && fn_gridexit;
    }

    /* Initialize a process grid.
       Calls PINFO, GET(-1,0), GRIDINIT, GRIDINFO in sequence. */
    void init_grid(int nprow_req, int npcol_req, char order = 'R')
    {
        fn_pinfo(&mypnum, &nprocs);

        int neg1 = -1, zero = 0;
        fn_get(&neg1, &zero, &ictxt);
        fn_gridinit(&ictxt, &order, &nprow_req, &npcol_req, (std::size_t)1);
        fn_gridinfo(&ictxt, &nprow, &npcol, &myrow, &mycol);
    }

    /* Destroy the current grid context. */
    void finalize()
    {
        if (ictxt >= 0) {
            fn_gridexit(&ictxt);
            ictxt = -1;
        }
    }

    bool is_root() const { return myrow == 0 && mycol == 0; }

    /* Check if this process is part of the grid (GRIDINFO returns -1 if not). */
    bool in_grid() const { return myrow >= 0 && mycol >= 0; }
};

/* ------------------------------------------------------------------ */
/* BLACS result structure                                               */
/* ------------------------------------------------------------------ */

struct BlacsResult {
    bool   passed;           /* Overall pass/fail */
    double max_error;        /* For combine ops (-1 if N/A) */
    int    data_mismatches;  /* For communication tests (0 if N/A) */
};

/* ------------------------------------------------------------------ */
/* Report a BLACS result                                               */
/* ------------------------------------------------------------------ */

inline void report_blacs_result(const char *routine, const char *params_str,
                                 const BlacsResult &br, const std::string &format)
{
    if (format == "json") {
        std::printf("{\"routine\":\"%s\",\"params\":\"%s\",\"passed\":%s",
                    routine, params_str, br.passed ? "true" : "false");
        if (br.max_error >= 0.0)
            std::printf(",\"max_error\":%.6e", br.max_error);
        if (br.data_mismatches > 0)
            std::printf(",\"data_mismatches\":%d", br.data_mismatches);
        std::printf("}\n");
    } else if (format == "csv") {
        std::printf("%s,%s,%s,%.6e,%d\n",
                    routine, params_str,
                    br.passed ? "true" : "false",
                    br.max_error,
                    br.data_mismatches);
    } else {
        /* default: text */
        std::printf("[%s %s] %s", routine, params_str,
                    br.passed ? "PASS" : "FAIL");
        if (br.max_error >= 0.0)
            std::printf("  max_error=%.6e", br.max_error);
        if (br.data_mismatches > 0)
            std::printf("  data_mismatches=%d", br.data_mismatches);
        std::printf("\n");
    }
}
