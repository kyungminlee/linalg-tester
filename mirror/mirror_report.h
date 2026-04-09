/* mirror_report.h -- Reporting for mirror tester */

#pragma once

#include "mirror_ctx.h"
#include "../src/core/tester_ctx.h"
#include "../src/core/sentinel.h"
#include <string>

void mirror_report_result(const char *routine, const char *params_str,
                           const ErrorResult &err,
                           const SentinelResult *sentinel_a,
                           const SentinelResult *sentinel_b,
                           const MirrorConfig &config);
