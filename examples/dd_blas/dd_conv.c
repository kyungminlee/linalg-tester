/*
 * dd_conv.c — conversion library for double-double precision.
 *
 * A double-double (DD) scalar is stored as two consecutive IEEE 754 doubles
 * (hi, lo) in memory, representing the exact value  hi + lo  where
 * |lo| < ulp(hi)/2  (the Dekker/Knuth non-overlapping condition).
 * sizeof(dd) == 16 bytes; pass --typesize 16 to the testers.
 *
 * Exports the two symbols required by linalg-tester:
 *   custom_to_mpfr(mpfr_t dst, const void *src)
 *   mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o dd_conv.so dd_conv.c -lmpfr
 */

#include <mpfr.h>
#include <string.h>

/* Read a (hi, lo) pair from src and set dst = hi + lo in full precision. */
void custom_to_mpfr(mpfr_t dst, const void *src)
{
    double hi, lo;
    memcpy(&hi, (const char *)src,                 sizeof(double));
    memcpy(&lo, (const char *)src + sizeof(double), sizeof(double));

    /* dst = hi */
    mpfr_set_d(dst, hi, MPFR_RNDN);
    /* dst += lo  (exact if prec >= 107 bits, as the caller should ensure) */
    mpfr_add_d(dst, dst, lo, MPFR_RNDN);
}

/*
 * Round src to the nearest double-double and write the (hi, lo) pair to dst.
 *
 * Algorithm (Dekker split / Fast2Sum):
 *   hi = round(src) as double
 *   lo = round(src - hi) as double
 * Both roundings use the caller-supplied rounding mode.
 */
void mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
{
    mpfr_t tmp;
    mpfr_prec_t prec = mpfr_get_prec(src);

    /* hi = src rounded to double */
    double hi = mpfr_get_d(src, rnd);

    /* lo = src - hi, then rounded to double */
    mpfr_init2(tmp, prec);
    mpfr_set_d(tmp, hi, MPFR_RNDN);       /* tmp = hi (exact) */
    mpfr_sub(tmp, src, tmp, MPFR_RNDN);   /* tmp = src - hi   */
    double lo = mpfr_get_d(tmp, rnd);
    mpfr_clear(tmp);

    memcpy((char *)dst,                 &hi, sizeof(double));
    memcpy((char *)dst + sizeof(double), &lo, sizeof(double));
}
