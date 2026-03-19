#pragma once

#include "reference.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <mpfr.h>

/* ------------------------------------------------------------------ */
/* Library loading                                                      */
/* ------------------------------------------------------------------ */

/* Open a list of libraries with RTLD_NOW|RTLD_GLOBAL (for preloads)
   and return handles. Exits on failure.                               */
inline std::vector<void *> preload_libs(const std::vector<std::string> &paths)
{
    std::vector<void *> handles;
    for (const auto &p : paths) {
        void *h = dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            std::fprintf(stderr, "dlopen preload(%s): %s\n", p.c_str(), dlerror());
            std::exit(EXIT_FAILURE);
        }
        handles.push_back(h);
    }
    return handles;
}

inline void close_libs(std::vector<void *> &handles)
{
    for (void *h : handles) dlclose(h);
    handles.clear();
}

/* Load a symbol from lib; exits on failure. */
inline void *load_sym(void *lib, const char *name)
{
    dlerror(); /* clear */
    void *sym = dlsym(lib, name);
    const char *err = dlerror();
    if (err) {
        std::fprintf(stderr, "dlsym(%s): %s\n", name, err);
        std::exit(EXIT_FAILURE);
    }
    return sym;
}

/* ------------------------------------------------------------------ */
/* Random matrix generation                                             */
/* ------------------------------------------------------------------ */

/* Generate 'count' values in (-1, 1); returns malloc'd count*typesize bytes. */
inline void *gen_random_array(int count, std::size_t typesize,
                               mpfr_to_custom_fn from_mpfr, mpfr_prec_t prec,
                               unsigned *seed)
{
    void *arr = std::malloc(static_cast<std::size_t>(count) * typesize);
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    char *p = static_cast<char *>(arr);
    for (int i = 0; i < count; ++i) {
        double d = static_cast<double>(rand_r(seed)) /
                   static_cast<double>(RAND_MAX) * 2.0 - 1.0;
        mpfr_set_d(val, d, MPFR_RNDN);
        from_mpfr(p, val, MPFR_RNDN);
        p += typesize;
    }

    mpfr_clear(val);
    return arr;
}

/* Generate a k×k triangular matrix (column-major, zero-padded).
   Diagonal is diagonally dominant: |A[i,i]| ~ k.                    */
inline void *gen_triangular_array(int k, char uplo, char diag,
                                   std::size_t typesize,
                                   mpfr_to_custom_fn from_mpfr,
                                   mpfr_prec_t prec, unsigned *seed)
{
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(k) * k, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            char *elem = arr + (static_cast<std::size_t>(j) * k + i) * typesize;
            if (i == j) {
                if (diag == 'U') {
                    mpfr_set_d(val, 1.0, MPFR_RNDN);
                } else {
                    double d = static_cast<double>(k) +
                               static_cast<double>(rand_r(seed)) /
                               static_cast<double>(RAND_MAX);
                    if (rand_r(seed) & 1) d = -d;
                    mpfr_set_d(val, d, MPFR_RNDN);
                }
                from_mpfr(elem, val, MPFR_RNDN);
            } else if ((uplo == 'U' && j > i) || (uplo == 'L' && i > j)) {
                double d = static_cast<double>(rand_r(seed)) /
                           static_cast<double>(RAND_MAX) * 2.0 - 1.0;
                mpfr_set_d(val, d, MPFR_RNDN);
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    return static_cast<void *>(arr);
}
