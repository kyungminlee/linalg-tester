/* pblas.h -- PBLAS test function declarations */

#pragma once

#include "../core/tester_ctx.h"
#include <string>

/* Level 3 */
void test_pgemm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_ptrsm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_psymm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_psyrk(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_psyr2k(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format);
void test_ptrmm(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);

/* Level 2 */
void test_pgemv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_psymv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_ptrmv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_ptrsv(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_pger(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_psyr(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_psyr2(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);

/* Level 1 */
void test_pswap(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_pscal(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_pcopy(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_paxpy(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_pdot(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format);
void test_pnrm2(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
void test_pasum(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format);
