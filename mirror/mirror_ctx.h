/* mirror_ctx.h -- Core structs for the mirror tester */

#pragma once

#include "../src/core/tester_ctx.h"
#include <mpfr.h>
#include <string>

/* Represents one side (A or B) of a mirror comparison */
struct MirrorSide {
    TesterCtx ctx;
    void *lib;
    std::string sym;
    std::string label; /* "A" or "B" */
};

/* Shared configuration for the mirror comparison */
struct MirrorConfig {
    mpfr_prec_t prec;      /* shared MPFR precision */
    std::string reference;  /* "a" or "b" */
    double threshold;       /* -1 = no threshold */
    std::string format;     /* text/json/csv */
};

/* Mirror test function signature */
using mirror_test_fn = void (*)(const MirrorSide &a, const MirrorSide &b,
                                 const TestParams &params,
                                 const MirrorConfig &config);

/* Dispatch table entry for mirror routines */
struct MirrorRoutineEntry {
    const char *name;
    const char *fortran_name; /* base for derive_sym (NULL = use name) */
    const char *category;
    const char *description;
    const char *formula;
    mirror_test_fn test_fn;
};
