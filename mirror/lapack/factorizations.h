/* factorizations.h -- Mirror test forward declarations for LAPACK factorizations */
#pragma once
#include "../mirror_ctx.h"

/* Real */
void mirror_test_getrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_potrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_sytrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_geqrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gelqf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_gebrd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_sytrd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Complex */
void mirror_test_cgetrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpotrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgeqrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgelqf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cgebrd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_hetrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_hetrd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_csytrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
