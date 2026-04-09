/* mirror_pblas1.h -- Forward declarations for PBLAS Level 1 mirror tests */
#pragma once
#include "../mirror_ctx.h"

void mirror_test_pswap(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pscal(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pcopy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_paxpy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pdot(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pnrm2(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pasum(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pamax(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pdotc(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pdotu(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
