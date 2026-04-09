/* derive_sym.h -- Fortran symbol name derivation for mirror tester */

#pragma once

#include "mirror_ctx.h"
#include <cstring>
#include <string>

inline std::string derive_sym(const std::string &prefix,
                               const MirrorRoutineEntry &r) {
    const char *routine_name = r.name;
    const char *base = r.fortran_name ? r.fortran_name : r.name;

    /* IAMAX: i<prefix>amax_ */
    if (std::strcmp(routine_name, "iamax") == 0)
        return "i" + prefix + "amax_";

    /* CROT: csrot_/zdrot_ */
    if (std::strcmp(routine_name, "crot") == 0) {
        if (prefix == "c") return "csrot_";
        if (prefix == "z") return "zdrot_";
        return prefix + "rot_";
    }

    /* CRSCAL: csscal_/zdscal_ */
    if (std::strcmp(routine_name, "crscal") == 0) {
        if (prefix == "c") return "csscal_";
        if (prefix == "z") return "zdscal_";
        return prefix + "scal_";
    }

    /* PBLAS: p<prefix><basename>_ (e.g. pgemm -> pdgemm_) */
    if (r.category &&
        (std::strncmp(r.category, "pblas", 5) == 0 ||
         std::strncmp(r.category, "cpblas", 6) == 0)) {
        if (base[0] == 'p')
            return std::string("p") + prefix + (base + 1) + "_";
    }

    /* Default: <prefix><base>_ */
    return prefix + base + "_";
}
