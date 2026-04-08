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
