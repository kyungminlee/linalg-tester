/* scalapack.h -- ScaLAPACK test function declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

/* Real ScaLAPACK factorizations */
void test_pgetrf(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_ppotrf(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_pgeqrf(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Real ScaLAPACK solvers */
void test_pgesv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_pposv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Real ScaLAPACK auxiliary */
void test_pgetrs(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_ppotrs(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_ptrtri(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_placpy(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_plange(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_plansy(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Real ScaLAPACK eigenvalue/SVD */
void test_psyev(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_psyevd(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_pgesvd(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Real ScaLAPACK redistribution */
void test_pgemr2d(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex ScaLAPACK factorizations */
void test_cpgetrf(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cppotrf(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cpgeqrf(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex ScaLAPACK solvers */
void test_cpgesv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cpposv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex ScaLAPACK auxiliary */
void test_cpgetrs(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cppotrs(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cptrtri(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cplacpy(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cplange(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_planhe(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cplansy(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex ScaLAPACK eigenvalue/SVD */
void test_pheev(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_pheevd(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_cpgesvd(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex ScaLAPACK redistribution */
void test_cpgemr2d(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
