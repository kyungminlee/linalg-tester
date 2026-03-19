/*
 * quad_conv.c — conversion library for quad-precision (IEEE 754 binary128).
 *
 * Exports the two symbols required by linalg-tester:
 *   custom_to_mpfr(mpfr_t dst, const void *src)
 *   mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o quad_conv.so quad_conv.c -lmpfr
 */

#define MPFR_WANT_FLOAT128 1
#include <mpfr.h>
#include <string.h>

/* REAL(16) in gfortran on x86_64 is IEEE 754 binary128.
   mpfr_set_float128 and mpfr_get_float128 are available since MPFR 4.0.0. */

void custom_to_mpfr(mpfr_t dst, const void *src)
{
    __float128 v;
    memcpy(&v, src, sizeof(__float128));
    mpfr_set_float128(dst, v, MPFR_RNDN);
}

void mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
{
    __float128 v = mpfr_get_float128(src, rnd);
    memcpy(dst, &v, sizeof(__float128));
}
