/* mirror_blacs.h -- Forward declarations for BLACS mirror tests */
#pragma once
#include "../mirror_ctx.h"

void mirror_test_blacs_setup(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_gesd2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_trsd2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_gebs2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_trbs2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_gsum2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_gamx2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_blacs_gamn2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
