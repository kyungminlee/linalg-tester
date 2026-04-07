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

add_executable(linalg_tester
    src/main.cpp
    ${CORE_SOURCES}
    ${BLAS3_SOURCES}
    ${BLAS2_SOURCES}
    ${BLAS1_SOURCES}
)

target_link_libraries(linalg_tester PRIVATE PkgConfig::GMP PkgConfig::MPFR)
target_include_directories(linalg_tester PRIVATE third_party)
