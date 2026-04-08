/* eigenvalues.h -- LAPACK eigenvalue/SVD tester declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

void test_syev(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_syevd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_syevr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_geev(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_gees(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_gesvd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_gesdd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_sygv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
