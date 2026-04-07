#include "report.h"

#include <cstdio>
#include <string>

void report_result(const char *routine, const char *params_str,
                   const ErrorResult &err, const std::string &format)
{
    if (format == "json") {
        std::printf("{\"routine\":\"%s\",\"params\":\"%s\","
                    "\"max_relative\":%.6e,\"normwise_relative\":%.6e}\n",
                    routine, params_str,
                    err.max_relative, err.normwise_relative);
    } else if (format == "csv") {
        std::printf("%s,%s,%.6e,%.6e\n",
                    routine, params_str,
                    err.max_relative, err.normwise_relative);
    } else {
        /* default: text */
        std::printf("[%s %s] max_rel=%.6e  normwise=%.6e\n",
                    routine, params_str,
                    err.max_relative, err.normwise_relative);
    }
}
