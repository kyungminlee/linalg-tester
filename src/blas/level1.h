#pragma once

#include "../core/tester_ctx.h"

#include <string>

void test_rotg(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_rotmg(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_rot(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_rotm(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_swap(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_scal(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_copy(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_axpy(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_dot(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_nrm2(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_asum(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_iamax(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex-only Level 1 */
void test_dotc(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_dotu(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_crot(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_crscal(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
