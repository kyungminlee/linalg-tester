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

/* Complex LAPACK auxiliary */
void test_cgetrs(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_cgetri(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_cpotrs(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_cpotri(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_ungqr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_unmqr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_cgecon(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_clange(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_lanhe(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_clacpy(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_claswp(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_clansy(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
