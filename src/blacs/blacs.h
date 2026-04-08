/* blacs.h -- BLACS test function declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

void test_blacs_setup(const TesterCtx &ctx, void *lib, const char *sym,
                      const TestParams &params, const std::string &format);

void test_blacs_gesd2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format);

void test_blacs_gebs2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format);

void test_blacs_gsum2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format);

void test_blacs_gamx2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format);

void test_blacs_gamn2d(const TesterCtx &ctx, void *lib, const char *sym,
                       const TestParams &params, const std::string &format);

void test_blacs_trsd2d(const TesterCtx &ctx, void *lib, const char *sym,
                        const TestParams &params, const std::string &format);

void test_blacs_trbs2d(const TesterCtx &ctx, void *lib, const char *sym,
                        const TestParams &params, const std::string &format);
