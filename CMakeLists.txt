cmake_minimum_required(VERSION 3.27)
project(linalg-tester)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)

set(CORE_SOURCES
    src/core/error_metrics.cpp
    src/core/generators.cpp
    src/core/report.cpp
    src/core/sentinel.cpp
)

set(BLAS3_SOURCES
    src/blas/level3/gemm.cpp
    src/blas/level3/trsm.cpp
    src/blas/level3/symm.cpp
    src/blas/level3/syrk.cpp
    src/blas/level3/syr2k.cpp
    src/blas/level3/trmm.cpp
    src/blas/level3/hemm.cpp
    src/blas/level3/herk.cpp
    src/blas/level3/her2k.cpp
)

set(BLAS2_SOURCES
    src/blas/level2/gemv.cpp
    src/blas/level2/symv.cpp
    src/blas/level2/trmv.cpp
    src/blas/level2/trsv.cpp
    src/blas/level2/ger.cpp
    src/blas/level2/syr.cpp
    src/blas/level2/syr2.cpp
    src/blas/level2/gbmv.cpp
    src/blas/level2/sbmv.cpp
    src/blas/level2/tbmv.cpp
    src/blas/level2/tbsv.cpp
    src/blas/level2/spmv.cpp
    src/blas/level2/tpmv.cpp
    src/blas/level2/tpsv.cpp
    src/blas/level2/spr.cpp
    src/blas/level2/spr2.cpp
    src/blas/level2/hemv.cpp
    src/blas/level2/hbmv.cpp
    src/blas/level2/hpmv.cpp
    src/blas/level2/geru.cpp
    src/blas/level2/gerc.cpp
    src/blas/level2/her.cpp
    src/blas/level2/hpr.cpp
    src/blas/level2/her2.cpp
    src/blas/level2/hpr2.cpp
)

set(BLAS1_SOURCES
    src/blas/level1/rotg.cpp
    src/blas/level1/rotmg.cpp
    src/blas/level1/rot.cpp
    src/blas/level1/rotm.cpp
    src/blas/level1/swap.cpp
    src/blas/level1/scal.cpp
    src/blas/level1/copy.cpp
    src/blas/level1/axpy.cpp
    src/blas/level1/dot.cpp
    src/blas/level1/nrm2.cpp
    src/blas/level1/asum.cpp
    src/blas/level1/iamax.cpp
    src/blas/level1/dotc.cpp
    src/blas/level1/dotu.cpp
    src/blas/level1/crot.cpp
    src/blas/level1/crscal.cpp
)

set(LAPACK_FACT_SOURCES
    src/lapack/factorizations/getrf.cpp
    src/lapack/factorizations/potrf.cpp
    src/lapack/factorizations/sytrf.cpp
    src/lapack/factorizations/geqrf.cpp
    src/lapack/factorizations/gelqf.cpp
    src/lapack/factorizations/gebrd.cpp
    src/lapack/factorizations/sytrd.cpp
)

set(LAPACK_SOLVE_SOURCES
    src/lapack/solvers/gesv.cpp
    src/lapack/solvers/posv.cpp
    src/lapack/solvers/sysv.cpp
    src/lapack/solvers/gbsv.cpp
    src/lapack/solvers/gtsv.cpp
    src/lapack/solvers/gels.cpp
    src/lapack/solvers/gelsd.cpp
)

set(LAPACK_EIG_SOURCES
    src/lapack/eigenvalues/syev.cpp
    src/lapack/eigenvalues/syevd.cpp
    src/lapack/eigenvalues/syevr.cpp
    src/lapack/eigenvalues/geev.cpp
    src/lapack/eigenvalues/gees.cpp
    src/lapack/eigenvalues/gesvd.cpp
    src/lapack/eigenvalues/gesdd.cpp
    src/lapack/eigenvalues/sygv.cpp
)

set(LAPACK_AUX_SOURCES
    src/lapack/auxiliary/getrs.cpp
    src/lapack/auxiliary/getri.cpp
    src/lapack/auxiliary/potrs.cpp
    src/lapack/auxiliary/potri.cpp
    src/lapack/auxiliary/orgqr.cpp
    src/lapack/auxiliary/ormqr.cpp
    src/lapack/auxiliary/gecon.cpp
    src/lapack/auxiliary/lange.cpp
    src/lapack/auxiliary/lansy.cpp
    src/lapack/auxiliary/lacpy.cpp
    src/lapack/auxiliary/laswp.cpp
)

set(CLAPACK_FACT_SOURCES
    src/lapack/factorizations/cgetrf.cpp
    src/lapack/factorizations/cpotrf.cpp
    src/lapack/factorizations/cgeqrf.cpp
    src/lapack/factorizations/cgelqf.cpp
    src/lapack/factorizations/cgebrd.cpp
    src/lapack/factorizations/hetrf.cpp
    src/lapack/factorizations/hetrd.cpp
    src/lapack/factorizations/csytrf.cpp
)

set(CLAPACK_SOLVE_SOURCES
    src/lapack/solvers/cgesv.cpp
    src/lapack/solvers/cposv.cpp
    src/lapack/solvers/cgbsv.cpp
    src/lapack/solvers/cgtsv.cpp
    src/lapack/solvers/cgels.cpp
    src/lapack/solvers/cgelsd.cpp
    src/lapack/solvers/hesv.cpp
    src/lapack/solvers/csysv.cpp
)

set(CLAPACK_EIG_SOURCES
    src/lapack/eigenvalues/heev.cpp
    src/lapack/eigenvalues/heevd.cpp
    src/lapack/eigenvalues/heevr.cpp
    src/lapack/eigenvalues/hegv.cpp
    src/lapack/eigenvalues/cgesvd.cpp
    src/lapack/eigenvalues/cgesdd.cpp
    src/lapack/eigenvalues/cgeev.cpp
    src/lapack/eigenvalues/cgees.cpp
)

set(CLAPACK_AUX_SOURCES
    src/lapack/auxiliary/cgetrs.cpp
    src/lapack/auxiliary/cgetri.cpp
    src/lapack/auxiliary/cpotrs.cpp
    src/lapack/auxiliary/cpotri.cpp
    src/lapack/auxiliary/ungqr.cpp
    src/lapack/auxiliary/unmqr.cpp
    src/lapack/auxiliary/cgecon.cpp
    src/lapack/auxiliary/clange.cpp
    src/lapack/auxiliary/lanhe.cpp
    src/lapack/auxiliary/clacpy.cpp
    src/lapack/auxiliary/claswp.cpp
    src/lapack/auxiliary/clansy.cpp
)

set(PBLAS_SOURCES
    src/pblas/level3/pgemm.cpp
    src/pblas/level3/ptrsm.cpp
    src/pblas/level3/psymm.cpp
    src/pblas/level3/psyrk.cpp
    src/pblas/level3/psyr2k.cpp
    src/pblas/level3/ptrmm.cpp
    src/pblas/level3/phemm.cpp
    src/pblas/level3/pherk.cpp
    src/pblas/level3/pher2k.cpp
    src/pblas/level3/ptran.cpp
    src/pblas/level3/ptranc.cpp
    src/pblas/level3/ptranu.cpp
    src/pblas/level2/pgemv.cpp
    src/pblas/level2/psymv.cpp
    src/pblas/level2/ptrmv.cpp
    src/pblas/level2/ptrsv.cpp
    src/pblas/level2/pger.cpp
    src/pblas/level2/psyr.cpp
    src/pblas/level2/psyr2.cpp
    src/pblas/level2/phemv.cpp
    src/pblas/level2/pgeru.cpp
    src/pblas/level2/pgerc.cpp
    src/pblas/level2/pher.cpp
    src/pblas/level2/pher2.cpp
    src/pblas/level1/pswap.cpp
    src/pblas/level1/pscal.cpp
    src/pblas/level1/pcopy.cpp
    src/pblas/level1/paxpy.cpp
    src/pblas/level1/pdot.cpp
    src/pblas/level1/pnrm2.cpp
    src/pblas/level1/pasum.cpp
    src/pblas/level1/pamax.cpp
    src/pblas/level1/pdotc.cpp
    src/pblas/level1/pdotu.cpp
)

set(BLACS_SOURCES
    src/blacs/context/blacs_setup.cpp
    src/blacs/p2p/blacs_gesd2d.cpp
    src/blacs/p2p/blacs_trsd2d.cpp
    src/blacs/broadcast/blacs_gebs2d.cpp
    src/blacs/broadcast/blacs_trbs2d.cpp
    src/blacs/combine/blacs_gsum2d.cpp
    src/blacs/combine/blacs_gamx2d.cpp
    src/blacs/combine/blacs_gamn2d.cpp
)

set(SCALAPACK_SOURCES
    src/scalapack/factorizations/pgetrf.cpp
    src/scalapack/factorizations/ppotrf.cpp
    src/scalapack/factorizations/pgeqrf.cpp
    src/scalapack/factorizations/cpgetrf.cpp
    src/scalapack/factorizations/cppotrf.cpp
    src/scalapack/factorizations/cpgeqrf.cpp
    src/scalapack/solvers/pgesv.cpp
    src/scalapack/solvers/pposv.cpp
    src/scalapack/solvers/cpgesv.cpp
    src/scalapack/solvers/cpposv.cpp
    src/scalapack/auxiliary/pgetrs.cpp
    src/scalapack/auxiliary/ppotrs.cpp
    src/scalapack/auxiliary/ptrtri.cpp
    src/scalapack/auxiliary/placpy.cpp
    src/scalapack/auxiliary/plange.cpp
    src/scalapack/auxiliary/plansy.cpp
    src/scalapack/auxiliary/cpgetrs.cpp
    src/scalapack/auxiliary/cppotrs.cpp
    src/scalapack/auxiliary/cptrtri.cpp
    src/scalapack/auxiliary/cplacpy.cpp
    src/scalapack/auxiliary/cplange.cpp
    src/scalapack/auxiliary/planhe.cpp
    src/scalapack/auxiliary/cplansy.cpp
    src/scalapack/eigenvalues/psyev.cpp
    src/scalapack/eigenvalues/psyevd.cpp
    src/scalapack/eigenvalues/pgesvd.cpp
    src/scalapack/eigenvalues/pheev.cpp
    src/scalapack/eigenvalues/pheevd.cpp
    src/scalapack/eigenvalues/cpgesvd.cpp
    src/scalapack/redistribution/pgemr2d.cpp
    src/scalapack/redistribution/cpgemr2d.cpp
)

add_executable(linalg_tester
    src/main.cpp
    ${CORE_SOURCES}
    ${BLAS3_SOURCES}
    ${BLAS2_SOURCES}
    ${BLAS1_SOURCES}
    ${LAPACK_FACT_SOURCES}
    ${LAPACK_SOLVE_SOURCES}
    ${LAPACK_EIG_SOURCES}
    ${LAPACK_AUX_SOURCES}
    ${CLAPACK_FACT_SOURCES}
    ${CLAPACK_SOLVE_SOURCES}
    ${CLAPACK_EIG_SOURCES}
    ${CLAPACK_AUX_SOURCES}
    ${BLACS_SOURCES}
    ${PBLAS_SOURCES}
    ${SCALAPACK_SOURCES}
)

target_link_libraries(linalg_tester PRIVATE PkgConfig::GMP PkgConfig::MPFR)
target_include_directories(linalg_tester PRIVATE third_party)

# ---------------------------------------------------------------------------
# Conversion libraries (shared libraries for testing)
# ---------------------------------------------------------------------------

add_library(float_conv SHARED examples/openblas_float/float_conv.c)
target_link_libraries(float_conv PRIVATE PkgConfig::GMP PkgConfig::MPFR)

add_library(double_conv SHARED examples/openblas_double/double_conv.c)
target_link_libraries(double_conv PRIVATE PkgConfig::GMP PkgConfig::MPFR)

add_library(complex_double_conv SHARED examples/openblas_complex_double/complex_double_conv.c)
target_link_libraries(complex_double_conv PRIVATE PkgConfig::GMP PkgConfig::MPFR)

# ---------------------------------------------------------------------------
# Mirror tester
# ---------------------------------------------------------------------------

set(MIRROR_CORE_SOURCES
    mirror/mirror_gen.cpp
    mirror/mirror_error.cpp
    mirror/mirror_report.cpp
)

set(MIRROR_BLAS3_SOURCES
    mirror/blas/level3/gemm.cpp
    mirror/blas/level3/trsm.cpp
    mirror/blas/level3/symm.cpp
    mirror/blas/level3/syrk.cpp
    mirror/blas/level3/syr2k.cpp
    mirror/blas/level3/trmm.cpp
    mirror/blas/level3/hemm.cpp
    mirror/blas/level3/herk.cpp
    mirror/blas/level3/her2k.cpp
)

set(MIRROR_BLAS2_SOURCES
    mirror/blas/level2/gemv.cpp
    mirror/blas/level2/symv.cpp
    mirror/blas/level2/trmv.cpp
    mirror/blas/level2/trsv.cpp
    mirror/blas/level2/ger.cpp
    mirror/blas/level2/syr.cpp
    mirror/blas/level2/syr2.cpp
    mirror/blas/level2/gbmv.cpp
    mirror/blas/level2/sbmv.cpp
    mirror/blas/level2/tbmv.cpp
    mirror/blas/level2/tbsv.cpp
    mirror/blas/level2/spmv.cpp
    mirror/blas/level2/tpmv.cpp
    mirror/blas/level2/tpsv.cpp
    mirror/blas/level2/spr.cpp
    mirror/blas/level2/spr2.cpp
    mirror/blas/level2/hemv.cpp
    mirror/blas/level2/hbmv.cpp
    mirror/blas/level2/hpmv.cpp
    mirror/blas/level2/geru.cpp
    mirror/blas/level2/gerc.cpp
    mirror/blas/level2/her.cpp
    mirror/blas/level2/hpr.cpp
    mirror/blas/level2/her2.cpp
    mirror/blas/level2/hpr2.cpp
)

set(MIRROR_BLAS1_SOURCES
    mirror/blas/level1/rotg.cpp
    mirror/blas/level1/rotmg.cpp
    mirror/blas/level1/rot.cpp
    mirror/blas/level1/rotm.cpp
    mirror/blas/level1/swap.cpp
    mirror/blas/level1/scal.cpp
    mirror/blas/level1/copy.cpp
    mirror/blas/level1/axpy.cpp
    mirror/blas/level1/dot.cpp
    mirror/blas/level1/nrm2.cpp
    mirror/blas/level1/asum.cpp
    mirror/blas/level1/iamax.cpp
    mirror/blas/level1/dotc.cpp
    mirror/blas/level1/dotu.cpp
    mirror/blas/level1/crot.cpp
    mirror/blas/level1/crscal.cpp
)

add_executable(mirror_tester
    mirror/main.cpp
    ${MIRROR_CORE_SOURCES}
    ${MIRROR_BLAS3_SOURCES}
    ${MIRROR_BLAS2_SOURCES}
    ${MIRROR_BLAS1_SOURCES}
    ${CORE_SOURCES}
)

target_link_libraries(mirror_tester PRIVATE PkgConfig::GMP PkgConfig::MPFR)
target_include_directories(mirror_tester PRIVATE third_party ${CMAKE_SOURCE_DIR})
