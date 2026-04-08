/* mpfr_complex.h -- Complex arithmetic helpers using pairs of mpfr_t */

#pragma once

#include <mpfr.h>

/* ------------------------------------------------------------------ */
/* Complex addition: (re_out, im_out) = (re_a, im_a) + (re_b, im_b)   */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_add(mpfr_t re_out, mpfr_t im_out,
                              const mpfr_t re_a, const mpfr_t im_a,
                              const mpfr_t re_b, const mpfr_t im_b,
                              mpfr_rnd_t rnd)
{
    mpfr_add(re_out, re_a, re_b, rnd);
    mpfr_add(im_out, im_a, im_b, rnd);
}

/* ------------------------------------------------------------------ */
/* Complex subtraction: (re_out, im_out) = (re_a, im_a) - (re_b, im_b)*/
/* ------------------------------------------------------------------ */

inline void mpfr_complex_sub(mpfr_t re_out, mpfr_t im_out,
                              const mpfr_t re_a, const mpfr_t im_a,
                              const mpfr_t re_b, const mpfr_t im_b,
                              mpfr_rnd_t rnd)
{
    mpfr_sub(re_out, re_a, re_b, rnd);
    mpfr_sub(im_out, im_a, im_b, rnd);
}

/* ------------------------------------------------------------------ */
/* Complex multiply: (re_out, im_out) = (re_a, im_a) * (re_b, im_b)   */
/*   re = re_a*re_b - im_a*im_b                                       */
/*   im = re_a*im_b + im_a*re_b                                       */
/* Uses temporaries to allow aliasing (out == a or out == b).          */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mul(mpfr_t re_out, mpfr_t im_out,
                              const mpfr_t re_a, const mpfr_t im_a,
                              const mpfr_t re_b, const mpfr_t im_b,
                              mpfr_rnd_t rnd)
{
    mpfr_prec_t prec = mpfr_get_prec(re_a);
    mpfr_t t1, t2, t3, t4;
    mpfr_init2(t1, prec); mpfr_init2(t2, prec);
    mpfr_init2(t3, prec); mpfr_init2(t4, prec);

    mpfr_mul(t1, re_a, re_b, rnd);   /* re_a * re_b */
    mpfr_mul(t2, im_a, im_b, rnd);   /* im_a * im_b */
    mpfr_mul(t3, re_a, im_b, rnd);   /* re_a * im_b */
    mpfr_mul(t4, im_a, re_b, rnd);   /* im_a * re_b */

    mpfr_sub(re_out, t1, t2, rnd);   /* re = t1 - t2 */
    mpfr_add(im_out, t3, t4, rnd);   /* im = t3 + t4 */

    mpfr_clear(t1); mpfr_clear(t2);
    mpfr_clear(t3); mpfr_clear(t4);
}

/* ------------------------------------------------------------------ */
/* Complex FMA: (re_acc, im_acc) += (re_a, im_a) * (re_b, im_b)       */
/*   re_acc += re_a*re_b - im_a*im_b                                  */
/*   im_acc += re_a*im_b + im_a*re_b                                  */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_fma(mpfr_t re_acc, mpfr_t im_acc,
                              const mpfr_t re_a, const mpfr_t im_a,
                              const mpfr_t re_b, const mpfr_t im_b,
                              mpfr_rnd_t rnd)
{
    mpfr_prec_t prec = mpfr_get_prec(re_acc);
    mpfr_t t;
    mpfr_init2(t, prec);

    /* re_acc += re_a*re_b */
    mpfr_fma(re_acc, re_a, re_b, re_acc, rnd);
    /* re_acc -= im_a*im_b */
    mpfr_mul(t, im_a, im_b, rnd);
    mpfr_sub(re_acc, re_acc, t, rnd);

    /* im_acc += re_a*im_b */
    mpfr_fma(im_acc, re_a, im_b, im_acc, rnd);
    /* im_acc += im_a*re_b */
    mpfr_fma(im_acc, im_a, re_b, im_acc, rnd);

    mpfr_clear(t);
}

/* ------------------------------------------------------------------ */
/* Complex conjugate: (re_out, im_out) = (re_in, -im_in)              */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_conj(mpfr_t re_out, mpfr_t im_out,
                               const mpfr_t re_in, const mpfr_t im_in,
                               mpfr_rnd_t rnd)
{
    mpfr_set(re_out, re_in, rnd);
    mpfr_neg(im_out, im_in, rnd);
}

/* ------------------------------------------------------------------ */
/* Complex absolute value: out = sqrt(re^2 + im^2)                     */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_abs(mpfr_t out,
                              const mpfr_t re_in, const mpfr_t im_in,
                              mpfr_rnd_t rnd)
{
    mpfr_prec_t prec = mpfr_get_prec(re_in);
    mpfr_t t1, t2;
    mpfr_init2(t1, prec); mpfr_init2(t2, prec);

    mpfr_mul(t1, re_in, re_in, rnd);
    mpfr_fma(t1, im_in, im_in, t1, rnd);
    mpfr_sqrt(out, t1, rnd);

    mpfr_clear(t1); mpfr_clear(t2);
}

/* ------------------------------------------------------------------ */
/* Complex absolute value squared: out = re^2 + im^2                   */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_abs_sq(mpfr_t out,
                                 const mpfr_t re_in, const mpfr_t im_in,
                                 mpfr_rnd_t rnd)
{
    mpfr_mul(out, re_in, re_in, rnd);
    mpfr_fma(out, im_in, im_in, out, rnd);
}

/* ------------------------------------------------------------------ */
/* Complex multiply by real: (re_out, im_out) = (re_a, im_a) * real_b */
/* ------------------------------------------------------------------ */

inline void mpfr_complex_mul_real(mpfr_t re_out, mpfr_t im_out,
                                   const mpfr_t re_a, const mpfr_t im_a,
                                   const mpfr_t real_b,
                                   mpfr_rnd_t rnd)
{
    mpfr_mul(re_out, re_a, real_b, rnd);
    mpfr_mul(im_out, im_a, real_b, rnd);
}
