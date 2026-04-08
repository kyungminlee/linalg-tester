#pragma once

#include "../core/tester_ctx.h"

#include <string>

void test_gemv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_symv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_trmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_trsv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_ger(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_syr(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_syr2(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_gbmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_sbmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_tbmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_tbsv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_spmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_tpmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_tpsv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_spr(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_spr2(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);

/* Complex-only Level 2 */
void test_hemv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_hbmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_hpmv(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_geru(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_gerc(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_her(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_hpr(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_her2(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
void test_hpr2(const TesterCtx &ctx, void *lib, const char *sym, const TestParams &params, const std::string &format);
