/* solvers.h -- LAPACK solver tester declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

void test_gesv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_posv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_sysv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_gbsv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_gtsv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_gels(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_gelsd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);

/* Complex LAPACK solvers */
void test_cgesv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_cposv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_cgbsv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_cgtsv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_cgels(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_cgelsd(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_hesv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_csysv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
