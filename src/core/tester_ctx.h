#pragma once

#include <cstddef>
#include <mpfr.h>

extern "C" typedef void (*custom_to_mpfr_fn)(mpfr_t dst, const void *src);
extern "C" typedef void (*mpfr_to_custom_fn)(void *dst, mpfr_t src, mpfr_rnd_t rnd);

struct TesterCtx {
    mpfr_prec_t prec;
    std::size_t typesize;
    custom_to_mpfr_fn to_mpfr;
    mpfr_to_custom_fn from_mpfr;
};

struct ErrorResult {
    double max_relative;      // max_ij |e_ij| / |ref_ij|
    double normwise_relative; // ||E||_F / ||ref||_F
};

struct TestParams {
    int m = 64, n = 64, k = 64;
    int kl = 2, ku = 2;
    int incx = 1, incy = 1;
    int ld_pad = 0;
    unsigned seed = 42;
};
