/* mirror_report.cpp -- Reporting for mirror tester */

#include "mirror_report.h"
#include <cstdio>

void mirror_report_result(const char *routine, const char *params_str,
                           const ErrorResult &err,
                           const SentinelResult *sentinel_a,
                           const SentinelResult *sentinel_b,
                           const MirrorConfig &config)
{
    bool sa_ok = (!sentinel_a || sentinel_a->passed);
    bool sb_ok = (!sentinel_b || sentinel_b->passed);
    const char *ref_label = (config.reference == "a") ? "A" : "B";

    if (config.format == "json") {
        std::printf("{\"routine\":\"%s\",\"params\":\"%s\","
                    "\"reference\":\"%s\","
                    "\"max_relative\":%.6e,\"normwise_relative\":%.6e,"
                    "\"sentinel_a_passed\":%s,\"sentinel_b_passed\":%s",
                    routine, params_str,
                    ref_label,
                    err.max_relative, err.normwise_relative,
                    sa_ok ? "true" : "false",
                    sb_ok ? "true" : "false");
        if (err.max_absolute_at_zero >= 0)
            std::printf(",\"max_absolute_at_zero\":%.6e", err.max_absolute_at_zero);
        if (err.nan_inf_mismatches > 0)
            std::printf(",\"nan_inf_mismatches\":%d", err.nan_inf_mismatches);
        std::printf("}\n");
    } else if (config.format == "csv") {
        std::printf("%s,%s,%s,%.6e,%.6e,%s,%s\n",
                    routine, params_str, ref_label,
                    err.max_relative, err.normwise_relative,
                    sa_ok ? "true" : "false",
                    sb_ok ? "true" : "false");
    } else {
        /* text */
        std::printf("[%s %s ref=%s] max_rel=%.6e  normwise=%.6e",
                    routine, params_str, ref_label,
                    err.max_relative, err.normwise_relative);
        if (err.max_absolute_at_zero > 0)
            std::printf("  max_abs_at_zero=%.6e", err.max_absolute_at_zero);
        if (err.nan_inf_mismatches > 0)
            std::printf(" NaN/Inf_MISMATCH(%d)", err.nan_inf_mismatches);
        if (sentinel_a && !sentinel_a->passed)
            std::printf(" SENTINEL_A_FAIL(%d)", sentinel_a->corrupted_count);
        if (sentinel_b && !sentinel_b->passed)
            std::printf(" SENTINEL_B_FAIL(%d)", sentinel_b->corrupted_count);
        std::printf("\n");
    }
}
