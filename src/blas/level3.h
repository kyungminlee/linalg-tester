#pragma once

#include "../core/tester_ctx.h"

#include <string>

void test_gemm(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_trsm(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_symm(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_syrk(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_syr2k(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_trmm(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex-only Level 3 */
void test_hemm(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_herk(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_her2k(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
