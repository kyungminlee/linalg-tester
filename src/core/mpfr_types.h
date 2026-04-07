#pragma once

#include <cstddef>
#include <cstdlib>
#include <mpfr.h>
#include "tester_ctx.h"

#define IDX(i, j, ld) (static_cast<std::size_t>(j) * (ld) + (i))

/* ------------------------------------------------------------------ */
/* RAII wrapper for an MPFR matrix (column-major)                       */
/* ------------------------------------------------------------------ */

class MpfrMatrix {
public:
    MpfrMatrix(int rows, int cols, mpfr_prec_t prec)
        : rows_(rows), cols_(cols)
    {
        std::size_t n = static_cast<std::size_t>(rows) * cols;
        data_ = new mpfr_t[n];
        for (std::size_t i = 0; i < n; ++i)
            mpfr_init2(data_[i], prec);
    }

    ~MpfrMatrix() {
        std::size_t n = static_cast<std::size_t>(rows_) * cols_;
        for (std::size_t i = 0; i < n; ++i)
            mpfr_clear(data_[i]);
        delete[] data_;
    }

    MpfrMatrix(const MpfrMatrix &) = delete;
    MpfrMatrix &operator=(const MpfrMatrix &) = delete;

    mpfr_t &at(int i, int j)             { return data_[IDX(i, j, rows_)]; }
    const mpfr_t &at(int i, int j) const { return data_[IDX(i, j, rows_)]; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

private:
    mpfr_t *data_;
    int     rows_;
    int     cols_;
};

/* ------------------------------------------------------------------ */
/* RAII wrapper for a single mpfr_t                                     */
/* ------------------------------------------------------------------ */

class MpfrScalar {
public:
    explicit MpfrScalar(mpfr_prec_t prec) { mpfr_init2(val_, prec); }
    ~MpfrScalar() { mpfr_clear(val_); }

    MpfrScalar(const MpfrScalar &) = delete;
    MpfrScalar &operator=(const MpfrScalar &) = delete;

    mpfr_t &get() { return val_; }
    const mpfr_t &get() const { return val_; }

private:
    mpfr_t val_;
};

/* ------------------------------------------------------------------ */
/* Conversion helpers                                                   */
/* ------------------------------------------------------------------ */

/* Convert a column-major custom matrix to an MpfrMatrix. */
inline void custom_to_mpfr_mat(MpfrMatrix &dst, const void *src,
                                int ld, const TesterCtx &ctx)
{
    const char *p = static_cast<const char *>(src);
    for (int j = 0; j < dst.cols(); ++j)
        for (int i = 0; i < dst.rows(); ++i)
            ctx.to_mpfr(dst.at(i, j),
                        p + IDX(i, j, ld) * ctx.typesize);
}

/* Convert a strided custom vector to an MpfrMatrix(n,1).
   Follows BLAS convention for negative inc:
     inc > 0: element i at src[i * inc]
     inc < 0: element i at src[(n-1-i) * |inc|]                       */
inline void custom_to_mpfr_vec(MpfrMatrix &dst, const void *src,
                                int inc, const TesterCtx &ctx)
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
        ctx.to_mpfr(dst.at(i, 0), p + offset);
    }
}
