/* solvers.h -- Mirror test forward declarations for LAPACK solvers */
#pragma once
#include "../mirror_ctx.h"

/* Real */
void mirror_test_gesv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_posv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_sysv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gbsv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gtsv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gels(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gelsd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Complex */
void mirror_test_cgesv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cposv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgbsv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgtsv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgels(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgelsd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_hesv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_csysv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
