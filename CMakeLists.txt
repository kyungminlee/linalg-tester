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
)

set(BLAS3_SOURCES
    src/blas/level3/gemm.cpp
    src/blas/level3/trsm.cpp
    src/blas/level3/symm.cpp
    src/blas/level3/syrk.cpp
    src/blas/level3/syr2k.cpp
    src/blas/level3/trmm.cpp
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
)

target_link_libraries(linalg_tester PRIVATE PkgConfig::GMP PkgConfig::MPFR)
target_include_directories(linalg_tester PRIVATE third_party)
