/* mirror_blas3.h -- Mirror tester Level 3 BLAS declarations */

#pragma once

#include "../mirror_ctx.h"

void mirror_test_gemm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_trsm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_symm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_syrk(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_syr2k(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config);
void mirror_test_trmm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);

/* Complex-only Level 3 */
void mirror_test_hemm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_herk(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config);
void mirror_test_her2k(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config);
