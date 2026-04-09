/* auxiliary.h -- Mirror test forward declarations for LAPACK auxiliary */
#pragma once
#include "../mirror_ctx.h"

/* Real */
void mirror_test_getrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_getri(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_potrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_potri(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_orgqr(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_ormqr(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gecon(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_lange(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_lansy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_lacpy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_laswp(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Complex */
void mirror_test_cgetrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgetri(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpotrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpotri(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_ungqr(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_unmqr(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgecon(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_clange(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_lanhe(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_clacpy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_claswp(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_clansy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
