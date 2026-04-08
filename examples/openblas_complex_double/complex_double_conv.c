/*
 * complex_double_conv.c -- conversion library for complex double-precision.
 *
 * Exports all four symbols required by linalg-tester in complex mode:
 *   custom_to_mpfr(mpfr_t dst, const void *src)
 *   mpfr_to_custom(void *dst, mpfr_t src, mpfr_rnd_t rnd)
 *   custom_to_mpfr_complex(mpfr_t re, mpfr_t im, const void *src)
 *   mpfr_to_custom_complex(void *dst, mpfr_t re, mpfr_t im, mpfr_rnd_t rnd)
 *
 * Memory layout: complex double is two consecutive doubles (real, imaginary),
 * total typesize = 16.  The real-only functions operate on a single double
 * (for routines like HERK where alpha/beta are real scalars; use --typesize 16
 * for complex elements and the tester will call the complex variants for
 * matrix/vector data).
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o complex_double_conv.so complex_double_conv.c -lmpfr
 */

#include <mpfr.h>
#include <string.h>

/* Real scalar conversion (for real-valued parameters in complex routines) */

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

/* Complex scalar conversion */

void custom_to_mpfr_complex(mpfr_t re, mpfr_t im, const void *src)
{
    double parts[2];
    memcpy(parts, src, sizeof(parts));
    mpfr_set_d(re, parts[0], MPFR_RNDN);
    mpfr_set_d(im, parts[1], MPFR_RNDN);
}

void mpfr_to_custom_complex(void *dst, mpfr_t re, mpfr_t im, mpfr_rnd_t rnd)
{
    double parts[2];
    parts[0] = mpfr_get_d(re, rnd);
    parts[1] = mpfr_get_d(im, rnd);
    memcpy(dst, parts, sizeof(parts));
}
