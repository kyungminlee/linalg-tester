#pragma once

#include "tester_ctx.h"
#include <cstddef>
#include <mpfr.h>

void *gen_random_array(int count, std::size_t typesize,
                        mpfr_to_custom_fn from_mpfr, mpfr_prec_t prec,
                        unsigned *seed);

void *gen_triangular_array(int k, char uplo, char diag,
                            std::size_t typesize,
                            mpfr_to_custom_fn from_mpfr,
                            mpfr_prec_t prec, unsigned *seed);

void *gen_symmetric_array(int n, char uplo,
                           std::size_t typesize,
                           mpfr_to_custom_fn from_mpfr,
                           mpfr_prec_t prec, unsigned *seed);

void *gen_band_array(int m, int n, int kl, int ku,
                      std::size_t typesize,
                      mpfr_to_custom_fn from_mpfr,
                      mpfr_prec_t prec, unsigned *seed);

void *gen_symmetric_band_array(int n, int k, char uplo,
                                std::size_t typesize,
                                mpfr_to_custom_fn from_mpfr,
                                mpfr_prec_t prec, unsigned *seed);

void *gen_triangular_band_array(int n, int k, char uplo, char diag,
                                 std::size_t typesize,
                                 mpfr_to_custom_fn from_mpfr,
                                 mpfr_prec_t prec, unsigned *seed);

void *gen_packed_symmetric_array(int n, char uplo,
                                  std::size_t typesize,
                                  mpfr_to_custom_fn from_mpfr,
                                  mpfr_prec_t prec, unsigned *seed);

void *gen_packed_triangular_array(int n, char uplo, char diag,
                                   std::size_t typesize,
                                   mpfr_to_custom_fn from_mpfr,
                                   mpfr_prec_t prec, unsigned *seed);
