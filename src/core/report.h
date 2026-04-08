#pragma once

#include "sentinel.h"
#include "tester_ctx.h"
#include <cstdio>
#include <string>

void report_result(const char *routine, const char *params_str,
                   const ErrorResult &err, const std::string &format);

void report_result(const char *routine, const char *params_str,
                   const ErrorResult &err, const SentinelResult *sentinel,
                   const std::string &format);
