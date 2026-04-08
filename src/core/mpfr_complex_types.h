/* mpfr_complex_types.h -- RAII wrappers for complex MPFR types */

#pragma once

#include "mpfr_types.h"
#include "mpfr_complex.h"
#include "tester_ctx.h"
#include <cstddef>

/* ------------------------------------------------------------------ */
/* RAII wrapper for a complex mpfr scalar (re + im)                    */
/* ------------------------------------------------------------------ */

class MpfrComplexScalar {
public:
    explicit MpfrComplexScalar(mpfr_prec_t prec) {
        mpfr_init2(re_, prec);
        mpfr_init2(im_, prec);
    }
    ~MpfrComplexScalar() {
        mpfr_clear(re_);
        mpfr_clear(im_);
    }

    MpfrComplexScalar(const MpfrComplexScalar &) = delete;
    MpfrComplexScalar &operator=(const MpfrComplexScalar &) = delete;

    mpfr_t &re() { return re_; }
    mpfr_t &im() { return im_; }
    const mpfr_t &re() const { return re_; }
    const mpfr_t &im() const { return im_; }

private:
    mpfr_t re_;
    mpfr_t im_;
};

/* ------------------------------------------------------------------ */
/* RAII wrapper for a complex column-major matrix                       */
/* Stores separate real and imaginary MpfrMatrix arrays.               */
/* ------------------------------------------------------------------ */

class MpfrComplexMatrix {
public:
    MpfrComplexMatrix(int rows, int cols, mpfr_prec_t prec)
        : re_(rows, cols, prec), im_(rows, cols, prec) {}

    MpfrComplexMatrix(const MpfrComplexMatrix &) = delete;
    MpfrComplexMatrix &operator=(const MpfrComplexMatrix &) = delete;

    mpfr_t &re(int i, int j)             { return re_.at(i, j); }
    const mpfr_t &re(int i, int j) const { return re_.at(i, j); }
    mpfr_t &im(int i, int j)             { return im_.at(i, j); }
    const mpfr_t &im(int i, int j) const { return im_.at(i, j); }

    int rows() const { return re_.rows(); }
    int cols() const { return re_.cols(); }

    MpfrMatrix &real_part() { return re_; }
    MpfrMatrix &imag_part() { return im_; }
    const MpfrMatrix &real_part() const { return re_; }
    const MpfrMatrix &imag_part() const { return im_; }

private:
    MpfrMatrix re_;
    MpfrMatrix im_;
};

/* ------------------------------------------------------------------ */
/* Conversion helpers for complex types                                */
/* ------------------------------------------------------------------ */

/* Convert a column-major custom complex matrix to MpfrComplexMatrix.
   Each element occupies ctx.typesize bytes and is decomposed via
   ctx.to_mpfr_complex into separate real and imaginary parts. */
inline void custom_to_mpfr_complex_mat(MpfrComplexMatrix &dst,
                                        const void *src, int ld,
                                        const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    for (int j = 0; j < dst.cols(); ++j)
        for (int i = 0; i < dst.rows(); ++i)
            ctx.to_mpfr_complex(dst.re(i, j), dst.im(i, j),
                                p + IDX(i, j, ld) * ctx.typesize);
}

/* Convert a strided custom complex vector to MpfrComplexMatrix(n,1).
   Follows BLAS convention for negative inc. */
inline void custom_to_mpfr_complex_vec(MpfrComplexMatrix &dst,
                                        const void *src, int inc,
                                        const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    int n = dst.rows();
    int abs_inc = (inc < 0) ? -inc : inc;
    for (int i = 0; i < n; ++i) {
        std::size_t offset;
        if (inc > 0) {
            offset = static_cast<std::size_t>(i) * abs_inc * ctx.typesize;
        } else {
            offset = static_cast<std::size_t>(n - 1 - i) * abs_inc * ctx.typesize;
        }
        ctx.to_mpfr_complex(dst.re(i, 0), dst.im(i, 0), p + offset);
    }
}
