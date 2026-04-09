/* mirror_blas1.h -- Mirror tester Level 1 BLAS declarations */

#pragma once

#include "../mirror_ctx.h"

void mirror_test_rotg(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_rotmg(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config);
void mirror_test_rot(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_rotm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_swap(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_scal(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_copy(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_axpy(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_dot(const MirrorSide &a, const MirrorSide &b,
                      const TestParams &params, const MirrorConfig &config);
void mirror_test_nrm2(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_asum(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_iamax(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config);

/* Complex-only Level 1 */
void mirror_test_dotc(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_dotu(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_crot(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_crscal(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config);
