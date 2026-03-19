cmake_minimum_required(VERSION 3.27)
project(linalg-tester)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)

add_library(reference OBJECT src/reference.cpp)
target_link_libraries(reference PUBLIC PkgConfig::GMP PkgConfig::MPFR)

foreach(routine IN ITEMS gemm trsm)
  add_executable(${routine}_tester src/${routine}_tester.cpp)
  target_link_libraries(${routine}_tester PRIVATE reference)
  target_include_directories(${routine}_tester PRIVATE third_party)
endforeach()
