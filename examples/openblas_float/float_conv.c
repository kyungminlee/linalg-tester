/*
 * float_conv.c — conversion library for single-precision (IEEE 754 float).
 *
 * Exports the two symbols required by linalg-tester:
 *   custom_to_mpfr(mpfr_t dst, const void *src)
 *   mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o float_conv.so float_conv.c -lmpfr
 */

#include <mpfr.h>
#include <string.h>

void custom_to_mpfr(mpfr_t dst, const void *src)
{
    float v;
    memcpy(&v, src, sizeof(float));
    mpfr_set_flt(dst, v, MPFR_RNDN);
}

void mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
{
    float v = mpfr_get_flt(src, rnd);
    memcpy(dst, &v, sizeof(float));
}
