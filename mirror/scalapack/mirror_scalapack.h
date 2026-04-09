/* mirror_scalapack.h -- Forward declarations for ScaLAPACK mirror tests */
#pragma once
#include "../mirror_ctx.h"

/* Factorizations (real) */
void mirror_test_pgetrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_ppotrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pgeqrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
/* Factorizations (complex) */
void mirror_test_cpgetrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cppotrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpgeqrf(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Solvers (real) */
void mirror_test_pgesv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pposv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
/* Solvers (complex) */
void mirror_test_cpgesv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpposv(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Auxiliary (real) */
void mirror_test_pgetrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_ppotrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_ptrtri(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_placpy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_plange(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_plansy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
/* Auxiliary (complex) */
void mirror_test_cpgetrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cppotrs(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cptrtri(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cplacpy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cplange(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_planhe(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cplansy(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Eigenvalues (real) */
void mirror_test_psyev(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_psyevd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pgesvd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
/* Eigenvalues (complex) */
void mirror_test_pheev(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_pheevd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpgesvd(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);

/* Redistribution */
void mirror_test_pgemr2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
void mirror_test_cpgemr2d(const MirrorSide &a, const MirrorSide &b, const TestParams &params, const MirrorConfig &config);
