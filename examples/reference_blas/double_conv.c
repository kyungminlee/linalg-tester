/*
 * double_conv.c — conversion library for double-precision (IEEE 754 double).
 *
 * Exports the two symbols required by linalg-tester:
 *   custom_to_mpfr(mpfr_t dst, const void *src)
 *   mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o double_conv.so double_conv.c -lmpfr
 */

#include <mpfr.h>
#include <string.h>

void custom_to_mpfr(mpfr_t dst, const void *src)
{
    double v;
    memcpy(&v, src, sizeof(double));
    mpfr_set_d(dst, v, MPFR_RNDN);
}

void mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
{
    double v = mpfr_get_d(src, rnd);
    memcpy(dst, &v, sizeof(double));
}
