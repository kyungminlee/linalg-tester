/* auxiliary.h -- LAPACK auxiliary tester declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

void test_getrs(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_getri(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_potrs(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_potri(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_orgqr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_ormqr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_gecon(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_lange(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_lansy(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_lacpy(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_laswp(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
