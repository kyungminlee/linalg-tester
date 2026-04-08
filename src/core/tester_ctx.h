#pragma once

#include <cstddef>
#include <mpfr.h>

extern "C" typedef void (*custom_to_mpfr_fn)(mpfr_t dst, const void *src);
extern "C" typedef void (*mpfr_to_custom_fn)(void *dst, mpfr_t src, mpfr_rnd_t rnd);
extern "C" typedef void (*custom_to_mpfr_complex_fn)(mpfr_t re, mpfr_t im, const void *src);
extern "C" typedef void (*mpfr_to_custom_complex_fn)(void *dst, mpfr_t re, mpfr_t im, mpfr_rnd_t rnd);

enum class ComplexReturnABI { Hidden, Register };

struct TesterCtx {
    mpfr_prec_t prec;
    std::size_t typesize;
    custom_to_mpfr_fn to_mpfr;
    mpfr_to_custom_fn from_mpfr;
    /* Complex support */
    bool complex_mode = false;
    custom_to_mpfr_complex_fn to_mpfr_complex = nullptr;
    mpfr_to_custom_complex_fn from_mpfr_complex = nullptr;
    ComplexReturnABI complex_return_abi = ComplexReturnABI::Hidden;
};

struct ErrorResult {
    double max_relative;          // max_ij |e_ij| / |ref_ij|
    double normwise_relative;     // ||E||_F / ||ref||_F
    double max_absolute_at_zero;  // max absolute error where ref == 0 (-1 if no zeros)
    int    nan_inf_mismatches;    // count of NaN/Inf mismatches
};

struct TestParams {
    int m = 64, n = 64, k = 64;
    int kl = 2, ku = 2;
    int incx = 1, incy = 1;
    int ld_pad = 0;
    int mb = 0, nb = 0;  /* PBLAS block sizes (0 = auto) */
    unsigned seed = 42;
};
