/*
 * gfortran_stubs.c — minimal stubs for the two libgfortran symbols that
 * OpenBLAS references but never calls during DGEMM / DTRSM computation:
 *
 *   _gfortran_concat_string  — Fortran string concatenation (used only in
 *                               error/debug paths inside OpenBLAS).
 *   _gfortran_etime          — Fortran CPU-time query (used only in
 *                               optional timing/benchmark paths).
 *
 * This file is compiled into gfortran_stubs.so and passed as --preload so
 * that dlopen(libopenblas.so) can resolve its NEEDED libgfortran.so.5
 * entries even when libgfortran is not installed on the system.
 *
 * On systems where libgfortran.so.5 is already present (i.e. the normal
 * case after `apt install libopenblas-dev`), this file is not needed.
 *
 * Build:
 *   gcc -O0 -shared -fPIC -o gfortran_stubs.so gfortran_stubs.c
 */

#include <stdint.h>
#include <string.h>
#include <sys/time.h>

/*
 * void _gfortran_concat_string(int32_t destlen, char *dest,
 *                               int32_t len1,   const char *s1,
 *                               int32_t len2,   const char *s2)
 */
void _gfortran_concat_string(int32_t destlen, char *dest,
                              int32_t len1,   const char *s1,
                              int32_t len2,   const char *s2)
{
    int32_t copy1 = len1 < destlen ? len1 : destlen;
    memcpy(dest, s1, (size_t)copy1);
    int32_t copy2 = (len2 < destlen - copy1) ? len2 : destlen - copy1;
    if (copy2 > 0)
        memcpy(dest + copy1, s2, (size_t)copy2);
    if (copy1 + copy2 < destlen)
        memset(dest + copy1 + copy2, ' ', (size_t)(destlen - copy1 - copy2));
}

/*
 * float _gfortran_etime(float tarray[2])
 * Returns user + system CPU time; tarray[0]=user, tarray[1]=system.
 */
float _gfortran_etime(float *tarray)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    float t = (float)tv.tv_sec + (float)tv.tv_usec * 1e-6f;
    if (tarray) { tarray[0] = t; tarray[1] = 0.0f; }
    return t;
}
