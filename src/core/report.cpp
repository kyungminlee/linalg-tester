#include "report.h"

#include <cstdio>
#include <string>

void report_result(const char *routine, const char *params_str,
                   const ErrorResult &err, const std::string &format)
{
    report_result(routine, params_str, err, nullptr, format);
}

void report_result(const char *routine, const char *params_str,
                   const ErrorResult &err, const SentinelResult *sentinel,
                   const std::string &format)
{
    bool sentinel_ok = (!sentinel || sentinel->passed);

    if (format == "json") {
        std::printf("{\"routine\":\"%s\",\"params\":\"%s\","
                    "\"max_relative\":%.6e,\"normwise_relative\":%.6e,"
                    "\"sentinel_passed\":%s,\"sentinel_corrupted\":%d",
                    routine, params_str,
                    err.max_relative, err.normwise_relative,
                    sentinel_ok ? "true" : "false",
                    sentinel ? sentinel->corrupted_count : 0);
        if (err.max_absolute_at_zero >= 0)
            std::printf(",\"max_absolute_at_zero\":%.6e", err.max_absolute_at_zero);
        if (err.nan_inf_mismatches > 0)
            std::printf(",\"nan_inf_mismatches\":%d", err.nan_inf_mismatches);
        std::printf("}\n");
    } else if (format == "csv") {
        std::printf("%s,%s,%.6e,%.6e,%s,%d\n",
                    routine, params_str,
                    err.max_relative, err.normwise_relative,
                    sentinel_ok ? "true" : "false",
                    sentinel ? sentinel->corrupted_count : 0);
    } else {
        /* default: text */
        std::printf("[%s %s] max_rel=%.6e  normwise=%.6e",
                    routine, params_str,
                    err.max_relative, err.normwise_relative);
        if (err.max_absolute_at_zero > 0)
            std::printf("  max_abs_at_zero=%.6e", err.max_absolute_at_zero);
        if (err.nan_inf_mismatches > 0)
            std::printf(" NaN/Inf_MISMATCH(%d)", err.nan_inf_mismatches);
        if (sentinel && !sentinel->passed) {
            std::printf(" SENTINEL_FAIL(%d corrupted)", sentinel->corrupted_count);
        }
        std::printf("\n");
    }
}
