/* mirror_blas2.h -- Mirror tester Level 2 BLAS declarations */

#pragma once

#include "../mirror_ctx.h"

void mirror_test_gemv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_symv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_trmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_trsv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_ger(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_syr(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_syr2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_gbmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_sbmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_tbmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_tbsv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_spmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_tpmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_tpsv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_spr(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_spr2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);

/* Complex-only Level 2 */
void mirror_test_hemv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_hbmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_hpmv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_geru(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_gerc(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_her(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_hpr(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_her2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_hpr2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
