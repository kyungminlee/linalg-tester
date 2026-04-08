/* factorizations.h -- LAPACK factorization tester declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

void test_getrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_potrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_sytrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_geqrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_gelqf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_gebrd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_sytrd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
